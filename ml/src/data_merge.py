"""Merge cleaned district yields with season-aligned weather and district soil into the training table.

Join keys
  yield  x weather : (region, district) -> daily series; features computed over the crop's SEASON WINDOW of the crop year
  yield  x soil    : (region, district)  (soil is static; a regional estimate at the district centroid)
  yield  x centroid: (region, district)  (latitude/longitude as spatial features)
Rows that cannot be matched are DROPPED and counted: a missing weather record is never filled with a guess.
"""
import hashlib
import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from .real_config import PROCESSED_CSV, PROCESSED_META, PROVENANCE, SEASON_WINDOWS, SOIL_COLS, load_config

WEATHER_COLS = ["rainfall_mm", "temperature_c", "temp_max_c", "humidity_pct"]


def season_window(season: str, crop_year: int) -> tuple[date, date]:
    sm, sd, so, em, ed, eo = SEASON_WINDOWS[season]
    return date(crop_year + so, sm, sd), date(crop_year + eo, em, ed)


def season_weather_features(daily: pd.DataFrame, season: str, crop_year: int, min_cov: float = 0.9) -> dict | None:
    """Season totals/means from a daily frame (index=date; cols rain_mm,tmean_c,tmax_c,rh_pct). None if coverage is too thin."""
    s, e = season_window(season, crop_year)
    w = daily.loc[pd.Timestamp(s):pd.Timestamp(e)]
    days = (e - s).days + 1
    rain, tmean = w["rain_mm"].dropna(), w["tmean_c"].dropna()
    if len(rain) < min_cov * days or len(tmean) < min_cov * days:
        return None
    tmax, rh = w["tmax_c"].dropna(), w["rh_pct"].dropna() if "rh_pct" in w else pd.Series(dtype=float)
    return {"rainfall_mm": float(rain.sum() * days / len(rain)),            # scale a few missing days up to the full window
            "temperature_c": float(tmean.mean()),
            "temp_max_c": float(tmax.mean()) if len(tmax) >= min_cov * days else np.nan,
            "humidity_pct": float(rh.mean()) if len(rh) >= min_cov * days else np.nan}


def merge_dataset(clean: pd.DataFrame, centroids: pd.DataFrame, weather_loader, soil: pd.DataFrame | None, provider: str, soil_source: str,
                  cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """weather_loader(region, district) -> daily DataFrame | None."""
    cfg = cfg or load_config()
    min_cov = cfg["weather"]["min_season_coverage"]
    rep = {"rows_clean": len(clean), "dropped": {}}

    cen = centroids.dropna(subset=["latitude", "longitude"])[["region", "district", "latitude", "longitude"]].drop_duplicates(["region", "district"])
    df = clean.merge(cen, on=["region", "district"], how="left")
    miss = df["latitude"].isna()
    rep["dropped"]["no_district_centroid"] = int(miss.sum())
    rep["districts_without_centroid"] = sorted(set(map(tuple, df.loc[miss, ["region", "district"]].drop_duplicates().values.tolist())))[:50]
    df = df[~miss]

    feats, no_wx, thin = [], 0, 0
    for (st, d), g in df.groupby(["region", "district"]):
        daily = weather_loader(st, d)
        if daily is None or len(daily) == 0:
            no_wx += len(g)
            continue
        daily = daily.sort_index()
        cache: dict = {}
        for idx, season, year in zip(g.index, g["season"], g["year"], strict=True):
            f = cache.get((season, year))
            if (season, year) not in cache:
                f = cache[(season, year)] = season_weather_features(daily, season, int(year), min_cov)
            if f is None:
                thin += 1
            else:
                feats.append({"_idx": idx, **f})
    rep["dropped"]["no_weather_file_for_district"] = no_wx
    rep["dropped"]["insufficient_weather_coverage_in_season"] = thin
    wx = pd.DataFrame(feats).set_index("_idx") if feats else pd.DataFrame(columns=WEATHER_COLS)
    df = df.join(wx, how="inner")

    if soil is not None and len(soil):
        sc = [c for c in SOIL_COLS if c in soil.columns]
        df = df.merge(soil[["region", "district", *sc]].drop_duplicates(["region", "district"]), on=["region", "district"], how="left")
        if cfg["soil"].get("require_soil", True) and sc:
            nosoil = df[sc].isna().any(axis=1)
            rep["dropped"]["no_soil_record"] = int(nosoil.sum())
            df = df[~nosoil]
    for c in SOIL_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df["weather_source"] = f"{provider}_district_centroid_season_aggregate"
    df["soil_source"] = soil_source if soil is not None and len(soil) else "none"
    df["yield_source"] = "apy_observed"
    cols = ["region", "district", "crop", "season", "year", "yield_t_ha", "area_ha", "production_t", *WEATHER_COLS, *SOIL_COLS, "latitude", "longitude",
            "weather_source", "soil_source", "yield_source"]
    df = df[cols].sort_values(["year", "region", "district", "crop", "season"]).reset_index(drop=True)
    rep["rows_out"] = len(df)
    rep["districts_out"] = int(df.groupby(["region", "district"]).ngroups) if len(df) else 0
    return df, rep


def save_processed(df: pd.DataFrame, report: dict, extra: dict | None = None, csv=PROCESSED_CSV, meta=PROCESSED_META) -> dict:
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    version = hashlib.sha256(csv.read_bytes()).hexdigest()[:12]
    m = {"dataset_version": version, "built_at": datetime.now(timezone.utc).isoformat(), "rows": len(df),
         "years": [int(df["year"].min()), int(df["year"].max())] if len(df) else [], "provenance_classes": PROVENANCE,
         "sources": {"yield": "Government of India district-wise APY statistics (observed)", "weather": df["weather_source"].iloc[0] if len(df) else "",
                     "soil": df["soil_source"].iloc[0] if len(df) else ""}, "merge_report": report, **(extra or {})}
    meta.write_text(json.dumps(m, indent=2, default=str))
    return m


def load_processed(csv=PROCESSED_CSV, meta=PROCESSED_META) -> tuple[pd.DataFrame, dict]:
    if not csv.exists():
        raise FileNotFoundError(f"{csv} not found. Build it first: see docs/real-data.md (python -m ml.src.build_real_dataset).")
    return pd.read_csv(csv), (json.loads(meta.read_text()) if meta.exists() else {})