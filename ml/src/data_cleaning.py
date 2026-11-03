"""Clean raw district-level crop statistics (APY) into one tidy, unit-consistent table.

Supported raw layouts (auto-detected by column names, see COLUMN_ALIASES):
  * Kaggle / data.gov.in copy : State_Name, District_Name, Crop_Year, Season, Crop, Area, Production   (no unit columns)
  * India Data Portal export  : year, state_name, district_name, crop_name, season, area, area_unit, production, production_unit, yield, yield_unit
  * AgML-CY-Bench style       : Dist Name, State Name, Year, Area [ha], Production [t], Yield [kg/ha]  (single crop; add a `crop` column)
Yield is ALWAYS recomputed as production_tonnes / area_ha; a reported yield column is only used as a cross-check.
"""
import logging
import re

import numpy as np
import pandas as pd

from .real_config import (
    AREA_UNIT_TO_HA,
    CROP_ALIASES,
    CROP_PROD_UNIT_TO_T,
    DISTRICT_ALIASES,
    PROD_UNIT_TO_T,
    SEASON_ALIASES,
    STATE_ALIASES,
    TARGET,
    YIELD_BOUNDS,
    load_config,
)

log = logging.getLogger("data_cleaning")

COLUMN_ALIASES = {
    "state": ["state_name", "state", "state name"],
    "district": ["district_name", "district", "dist name", "district name", "district_as_per_source"],
    "crop_year": ["crop_year", "year", "fiscal_year", "crop year"],
    "season": ["season", "crop_season", "crop season"],
    "crop": ["crop", "crop_name", "crop name"],
    "area": ["area", "area_ha", "area [ha]", "area_in_hectares"],
    "production": ["production", "production_tonnes", "production [t]", "production_in_tonnes"],
    "area_unit": ["area_unit", "area unit"],
    "production_unit": ["production_unit", "production unit"],
    "yield_reported": ["yield", "yield_reported", "crop_yield"],
    "yield_unit": ["yield_unit", "yield unit"],
}
REQUIRED_RAW = ["state", "district", "crop_year", "crop", "area", "production"]


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to canonical names. Raises ValueError listing what is missing."""
    lookup = {re.sub(r"\s+", " ", c.strip().lower()): c for c in df.columns}
    ren = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in lookup and lookup[a] not in ren:
                ren[lookup[a]] = canon
                break
    out = df.rename(columns=ren)
    missing = [c for c in REQUIRED_RAW if c not in out.columns]
    if "season" not in out.columns:
        out["season"] = "Whole Year"  # source without a season column: cannot be matched to a weather window
    if missing:
        raise ValueError(f"Raw APY file is missing required column(s) {missing}. Found: {list(df.columns)}")
    return out


# ---------------------------------------------------------------- normalisers
def _clean_text(s) -> str:
    return re.sub(r"\s+", " ", str(s).replace("\u00a0", " ").strip())


def normalize_state(s) -> str | None:
    if pd.isna(s) or not str(s).strip():
        return None
    t = _clean_text(s).lower().rstrip(".")
    if t in STATE_ALIASES:
        return STATE_ALIASES[t]
    return " ".join(w if w in ("and", "of") else w.capitalize() for w in t.split(" "))


def normalize_district(s) -> str | None:
    if pd.isna(s) or not str(s).strip():
        return None
    t = _clean_text(s).lower()
    t = re.sub(r"\(.*?\)", "", t)                      # drop bracketed notes
    t = re.sub(r"\bdistrict\b", "", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return DISTRICT_ALIASES.get(t, t) or None


def normalize_crop(s) -> str | None:
    if pd.isna(s):
        return None
    return CROP_ALIASES.get(_clean_text(s).lower())


def normalize_season(s) -> str | None:
    if pd.isna(s):
        return None
    return SEASON_ALIASES.get(_clean_text(s).lower())


def parse_crop_year(v) -> float:
    """1997 -> 1997 ; '2015-16' / '2015-2016' -> 2015 (crop-year START year) ; invalid -> NaN."""
    if pd.isna(v):
        return np.nan
    m = re.match(r"^\s*(\d{4})\s*(?:[-/]\s*\d{2,4})?\s*$", str(v))
    if not m:
        return np.nan
    y = int(m.group(1))
    return float(y) if 1900 < y < 2100 else np.nan


# ---------------------------------------------------------------- units
def _unit_key(u) -> str:
    return _clean_text(u).lower() if not pd.isna(u) else ""


def convert_units(df: pd.DataFrame, assumptions: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Return area_ha and production_t. Rows whose unit cannot be converted are DROPPED and counted, never guessed."""
    assumptions = assumptions or {}
    rep = {"dropped_unknown_area_unit": 0, "dropped_unknown_production_unit": 0, "dropped_no_unit_info": 0, "assumed_units_used": {}}
    df = df.copy()
    n0 = len(df)
    if "area_unit" in df.columns:
        f = df["area_unit"].map(lambda u: AREA_UNIT_TO_HA.get(_unit_key(u)))
        bad = f.isna()
        rep["dropped_unknown_area_unit"] = int(bad.sum())
        df["area_ha"] = df["area"] * f
        df = df[~bad]
    else:
        df["area_ha"] = df["area"]  # Kaggle/data.gov.in copies state hectares; verify against your source

    if "production_unit" in df.columns:
        def factor(row):
            u = _unit_key(row["production_unit"])
            return CROP_PROD_UNIT_TO_T.get((row["crop"], u), PROD_UNIT_TO_T.get(u))
        f = df.apply(factor, axis=1)
        bad = f.isna()
        rep["dropped_unknown_production_unit"] = int(bad.sum())
        df["production_t"] = df["production"] * f
        df = df[~bad]
    else:
        # no unit column: tonnes assumed for all crops EXCEPT those known to use other units unless the user gave an explicit assumption
        known_odd = {c for c, _u in CROP_PROD_UNIT_TO_T}
        fac = pd.Series(1.0, index=df.index)
        odd = df["crop"].isin(known_odd)
        for crop in df.loc[odd, "crop"].unique():
            if assumptions.get(crop):
                fac[df["crop"] == crop] = float(assumptions[crop])
                rep["assumed_units_used"][crop] = float(assumptions[crop])
        undecided = odd & ~df["crop"].isin(assumptions.keys())
        rep["dropped_no_unit_info"] = int(undecided.sum())
        df["production_t"] = df["production"] * fac
        df = df[~undecided]
    rep["rows_after_units"] = len(df)
    rep["rows_before_units"] = n0
    return df, rep


# ---------------------------------------------------------------- main entry
def clean_apy(raw: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Raw APY frame -> (clean frame, report). Report lists every drop with a count so nothing disappears silently."""
    cfg = cfg or load_config()
    a, c = cfg["apy"], cfg["cleaning"]
    rep: dict = {"rows_in": len(raw), "steps": []}

    def step(name, df_before, df_after):
        rep["steps"].append({"step": name, "dropped": int(len(df_before) - len(df_after)), "remaining": int(len(df_after))})

    df = standardise_columns(raw)
    df["state"] = df["state"].map(normalize_state)
    df["district"] = df["district"].map(normalize_district)
    df["crop_raw"] = df["crop"]
    df["crop"] = df["crop"].map(normalize_crop)
    df["season"] = df["season"].map(normalize_season)
    df["year"] = df["crop_year"].map(parse_crop_year)
    for col in ("area", "production"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    b = df; df = df.dropna(subset=["state", "district"]); step("missing state/district", b, df)
    if a.get("states"):
        wanted = {normalize_state(x) for x in a["states"]}
        b = df; df = df[df["state"].isin(wanted)]; step(f"state not in {sorted(wanted)}", b, df)
    b = df; df = df.dropna(subset=["crop"]); step("crop not in supported crop list", b, df)
    b = df; df = df[df["season"].isin(a["seasons"])]; step(f"season not in {a['seasons']}", b, df)
    b = df; df = df.dropna(subset=["year"]); step("unparseable crop year", b, df)
    b = df; df = df[(df["year"] >= a["min_year"]) & (df["year"] <= a["max_year"])]; step(f"year outside {a['min_year']}-{a['max_year']}", b, df)
    df["year"] = df["year"].astype(int)
    b = df; df = df.dropna(subset=["area", "production"]); step("missing area or production", b, df)

    df, urep = convert_units(df, c.get("unit_assumptions"))
    rep["units"] = urep
    step("unit conversion", pd.DataFrame(index=range(urep["rows_before_units"])), pd.DataFrame(index=range(urep["rows_after_units"])))

    b = df; df = df[df["area_ha"] > 0]; step("zero or negative area", b, df)
    b = df; df = df[df["production_t"] >= 0]; step("negative production", b, df)
    if a.get("drop_zero_production", True):
        b = df; df = df[df["production_t"] > 0]; step("zero production (treated as not reported)", b, df)
    b = df; df = df[df["area_ha"] >= a["min_area_ha"]]; step(f"area below {a['min_area_ha']} ha", b, df)

    # duplicates: identical keys are summed (e.g. a district reported in two rows); report how many keys were merged
    key = ["state", "district", "crop", "season", "year"]
    dup = df.duplicated(key, keep=False)
    rep["duplicate_key_rows_merged"] = int(dup.sum())
    keep_cols = {"area_ha": "sum", "production_t": "sum"}
    if "yield_reported" in df.columns:
        keep_cols["yield_reported"] = "mean"
    df = df.groupby(key, as_index=False).agg(keep_cols)

    df[TARGET] = df["production_t"] / df["area_ha"]
    b = df
    lo = df["crop"].map(lambda k: YIELD_BOUNDS[k][0]); hi = df["crop"].map(lambda k: YIELD_BOUNDS[k][1])
    df = df[(df[TARGET] >= lo) & (df[TARGET] <= hi)]; step("yield outside physical bounds for the crop", b, df)

    # cross-check against the source's own yield column when present (kg/ha vs t/ha handled by ratio test)
    if "yield_reported" in df.columns and df["yield_reported"].notna().any():
        r = df["yield_reported"] / df[TARGET]
        ok = r.between(0.95, 1.05) | r.between(950, 1050)  # same units, or kg/ha vs t/ha
        rep["yield_crosscheck_agree_pct"] = round(float(ok[df["yield_reported"].notna()].mean() * 100), 1)
        df = df.drop(columns="yield_reported")

    df = df.rename(columns={"state": "region"}).sort_values(["year", "region", "district", "crop", "season"]).reset_index(drop=True)
    rep["rows_out"] = len(df)
    rep["years"] = [int(df["year"].min()), int(df["year"].max())] if len(df) else []
    rep["districts"] = int(df.groupby(["region", "district"]).ngroups) if len(df) else 0
    rep["by_crop"] = df["crop"].value_counts().to_dict() if len(df) else {}
    return df, rep