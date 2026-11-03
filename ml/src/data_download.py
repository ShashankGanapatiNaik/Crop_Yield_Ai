"""Download or import the raw datasets. NOTHING is fabricated: if a source is unreachable the step fails loudly,
and every raw file gets a provenance record (source, time, checksum).

Usage (from repo root; weather and soil are slow and resumable):
  python -m ml.src.data_download apy --src path/or/url/to/apy.csv
  python -m ml.src.data_download centroids [--csv my_district_centroids.csv]
  python -m ml.src.data_download weather --provider nasa_power --start 1997 --end 2022
  python -m ml.src.data_download soil --source soilgrids        # or:  --source csv --csv my_soil_by_district.csv

Endpoint response shapes were written from the providers' public documentation and are exercised in tests with MOCKED
payloads; on your first live run, check one district by hand (see docs/real-data.md).
"""
import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import time
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from .data_cleaning import normalize_district, normalize_state
from .real_config import RAW_DIR, SOIL_COLS, load_config

log = logging.getLogger("data_download")
NOMINATIM = "https://nominatim.openstreetmap.org/search"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
INDIA_BBOX = (6.0, 38.0, 68.0, 98.0)  # lat_min, lat_max, lon_min, lon_max


def _ua() -> str:
    return f"CropYieldAI/1.0 (academic research; contact: {os.environ.get('CROPAI_CONTACT', 'set CROPAI_CONTACT env var')})"


def _sleep(seconds: float) -> None:  # separate so tests can patch it out
    time.sleep(seconds)


def _get_json(url: str, params: dict | None = None, timeout: float = 90, retries: int = 3, backoff: float = 6.0):
    """The ONLY network function in the pipeline (tests monkeypatch it)."""
    last: Exception | None = None
    for i in range(retries):
        try:
            r = httpx.get(url, params=params, headers={"User-Agent": _ua()}, timeout=timeout)
            if r.status_code == 429:
                _sleep(backoff * (i + 1) * 2)
                continue
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            last = e
            _sleep(backoff * (i + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def district_key(region: str, district: str) -> str:
    return f"{region}|{district}"


# ------------------------------------------------------------------------------------------------ APY (yields)
def import_apy(src: str | Path, raw_dir: Path = RAW_DIR) -> Path:
    """Copy or download the district-level APY file into data/real/raw and record its provenance."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "apy_raw.csv"
    s = str(src)
    if s.startswith(("http://", "https://")):
        with httpx.stream("GET", s, headers={"User-Agent": _ua()}, timeout=300, follow_redirects=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    else:
        if not Path(s).exists():
            raise FileNotFoundError(f"{s} not found. Download the APY CSV yourself (see docs/real-data.md) and pass its path.")
        shutil.copyfile(s, dest)
    prov = {"source": s, "retrieved_at": datetime.now(timezone.utc).isoformat(), "sha256": sha256_file(dest), "bytes": dest.stat().st_size,
            "note": "Government of India district-wise APY statistics; check the licence of the exact copy you used."}
    (raw_dir / "apy_provenance.json").write_text(json.dumps(prov, indent=2))
    log.info("APY imported: %s (%s bytes)", dest, prov["bytes"])
    return dest


def district_pairs_from_raw(raw_path: Path | None = None) -> list[tuple[str, str]]:
    from .data_cleaning import clean_apy
    raw = pd.read_csv(raw_path or RAW_DIR / "apy_raw.csv", low_memory=False)
    clean, _ = clean_apy(raw)
    return sorted(set(map(tuple, clean[["region", "district"]].drop_duplicates().values)))


# ------------------------------------------------------------------------------------------------ district centroids
def geocode_district(region: str, district: str, get=None) -> dict | None:
    """Nominatim lookup with validation: inside India AND the returned state must match the APY state."""
    get = get or _get_json
    res = get(NOMINATIM, {"q": f"{district} district, {region}, India", "format": "jsonv2", "limit": 1, "addressdetails": 1, "countrycodes": "in"})
    if not res:
        return None
    r = res[0]
    lat, lon = float(r["lat"]), float(r["lon"])
    if not (INDIA_BBOX[0] <= lat <= INDIA_BBOX[1] and INDIA_BBOX[2] <= lon <= INDIA_BBOX[3]):
        return None
    if normalize_state((r.get("address") or {}).get("state")) != region:
        return None  # same district name in another state: refuse rather than mis-assign weather
    return {"latitude": lat, "longitude": lon}


def fetch_centroids(pairs, cache: Path | None = None, sleep: float = 1.1, get=None) -> pd.DataFrame:
    """Resumable. Unresolved districts are stored with NaN so they are not retried every run."""
    cache = cache or RAW_DIR / "district_centroids.csv"
    have = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["region", "district", "latitude", "longitude", "source"])
    done = set(zip(have["region"], have["district"], strict=False))
    rows = have.to_dict("records")
    for i, (st, d) in enumerate(p for p in pairs if tuple(p) not in done):
        hit = geocode_district(st, d, get)
        rows.append({"region": st, "district": d, "latitude": hit["latitude"] if hit else np.nan, "longitude": hit["longitude"] if hit else np.nan,
                     "source": "nominatim_search_approx" if hit else "unresolved"})
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(cache, index=False)
        _sleep(sleep)
    out = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def import_centroids_csv(path: str | Path, cache: Path | None = None) -> pd.DataFrame:
    """Preferred: your own district centroid table (columns: state|region, district, latitude, longitude)."""
    df = pd.read_csv(path)
    df = df.rename(columns={"state": "region", "lat": "latitude", "lon": "longitude", "lng": "longitude"})
    need = {"region", "district", "latitude", "longitude"}
    if need - set(df.columns):
        raise ValueError(f"centroid CSV needs columns {sorted(need)}; has {list(df.columns)}")
    df["region"] = df["region"].map(normalize_state)
    df["district"] = df["district"].map(normalize_district)
    df["source"] = "user_supplied_csv"
    df = df.dropna(subset=["region", "district", "latitude", "longitude"]).drop_duplicates(["region", "district"])
    cache = cache or RAW_DIR / "district_centroids.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


# ------------------------------------------------------------------------------------------------ weather
def parse_power(payload: dict) -> pd.DataFrame:
    p = payload["properties"]["parameter"]
    d = pd.DataFrame({"rain_mm": pd.Series(p["PRECTOTCORR"]), "tmean_c": pd.Series(p["T2M"]), "tmax_c": pd.Series(p["T2M_MAX"]),
                      "rh_pct": pd.Series(p.get("RH2M", {}))})
    d.index = pd.to_datetime(d.index, format="%Y%m%d")
    d = d.where(d > -900)  # NASA POWER fill value is -999
    d.index.name = "date"
    return d.sort_index()


def parse_open_meteo(payload: dict) -> pd.DataFrame:
    x = payload["daily"]
    d = pd.DataFrame({"rain_mm": x["precipitation_sum"], "tmean_c": x["temperature_2m_mean"], "tmax_c": x["temperature_2m_max"]},
                     index=pd.to_datetime(x["time"]))
    d["rh_pct"] = np.nan  # not requested from this provider: humidity is simply unavailable, never invented
    d.index.name = "date"
    return d.sort_index()


def fetch_weather_daily(lat: float, lon: float, start: date, end: date, provider: str, get=None) -> pd.DataFrame:
    get = get or _get_json
    if provider == "nasa_power":
        payload = get(POWER_URL, {"parameters": "PRECTOTCORR,T2M,T2M_MAX,RH2M", "community": "AG", "longitude": lon, "latitude": lat,
                                  "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"), "format": "JSON"})
        return parse_power(payload)
    if provider == "open_meteo":
        payload = get(OPEN_METEO_ARCHIVE, {"latitude": lat, "longitude": lon, "start_date": start.isoformat(), "end_date": end.isoformat(),
                                           "daily": "precipitation_sum,temperature_2m_mean,temperature_2m_max", "timezone": "UTC"})
        return parse_open_meteo(payload)
    raise ValueError(f"unknown weather provider {provider!r}")


def weather_cache_path(provider: str, region: str, district: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / "weather" / provider / f"{_slug(region)}__{_slug(district)}.csv"


def load_weather(provider: str, region: str, district: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame | None:
    p = weather_cache_path(provider, region, district, raw_dir)
    return pd.read_csv(p, parse_dates=["date"], index_col="date") if p.exists() else None


def download_weather(centroids: pd.DataFrame, start_year: int, end_year: int, provider: str, sleep: float = 1.0, raw_dir: Path = RAW_DIR, get=None) -> dict:
    """Resumable: districts already cached are skipped."""
    start, end = date(start_year, 1, 1), min(date(end_year + 1, 6, 30), date.today())
    stats = {"fetched": 0, "cached": 0, "failed": []}
    for r in centroids.dropna(subset=["latitude", "longitude"]).itertuples():
        p = weather_cache_path(provider, r.region, r.district, raw_dir)
        if p.exists():
            stats["cached"] += 1
            continue
        try:
            d = fetch_weather_daily(r.latitude, r.longitude, start, end, provider, get)
        except Exception as e:  # noqa: BLE001 - record and continue so a long run is not lost
            stats["failed"].append({"district": f"{r.region}/{r.district}", "error": str(e)[:200]})
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(p)
        stats["fetched"] += 1
        _sleep(sleep)
    return stats


# ------------------------------------------------------------------------------------------------ soil
SOILGRIDS_MAP = {"phh2o": "soil_ph", "soc": "soil_organic_carbon", "sand": "soil_sand_pct", "clay": "soil_clay_pct"}
SOILGRIDS_DEPTHS = ["0-5cm", "5-15cm", "15-30cm"]


def parse_soilgrids(payload: dict) -> dict:
    """Mean over the top 30 cm of each property, converted with the API's own d_factor (pH, g/kg, %, %)."""
    out = {}
    for layer in payload["properties"]["layers"]:
        col = SOILGRIDS_MAP.get(layer["name"])
        if not col:
            continue
        f = (layer.get("unit_measure") or {}).get("d_factor") or 1
        vals = [dp["values"]["mean"] / f for dp in layer["depths"] if (dp.get("values") or {}).get("mean") is not None]
        out[col] = float(np.mean(vals)) if vals else np.nan
    return out


def fetch_soil(lat: float, lon: float, get=None) -> dict:
    get = get or _get_json
    return parse_soilgrids(get(SOILGRIDS_URL, {"lon": lon, "lat": lat, "property": list(SOILGRIDS_MAP), "depth": SOILGRIDS_DEPTHS, "value": "mean"}))


def import_soil_csv(path: str | Path, out: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns={"state": "region"})
    miss = {"region", "district"} - set(df.columns)
    if miss:
        raise ValueError(f"soil CSV needs columns region/state and district (+ any of {SOIL_COLS})")
    df["region"] = df["region"].map(normalize_state)
    df["district"] = df["district"].map(normalize_district)
    df["soil_source"] = "user_supplied_csv"
    out = out or RAW_DIR / "soil_district.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def download_soil(centroids: pd.DataFrame, sleep: float = 13.0, out: Path | None = None, get=None) -> pd.DataFrame:
    """Resumable SoilGrids point queries at each district centroid (a regional ESTIMATE, not a field measurement)."""
    out = out or RAW_DIR / "soil_district.csv"
    have = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["region", "district", *SOIL_COLS, "soil_source"])
    done = set(zip(have["region"], have["district"], strict=False))
    rows = have.to_dict("records")
    for i, r in enumerate(x for x in centroids.dropna(subset=["latitude"]).itertuples() if (x.region, x.district) not in done):
        try:
            rows.append({"region": r.region, "district": r.district, **fetch_soil(r.latitude, r.longitude, get), "soil_source": "soilgrids_v2_point_0_30cm"})
        except Exception as e:  # noqa: BLE001
            log.warning("soil failed for %s/%s: %s", r.region, r.district, e)
        if i % 10 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)
        _sleep(sleep)
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("apy"); a.add_argument("--src", required=True, help="path or URL of the district-level APY CSV")
    c = sub.add_parser("centroids"); c.add_argument("--csv", help="your own centroid CSV (recommended); otherwise Nominatim is used")
    w = sub.add_parser("weather"); w.add_argument("--provider", default=cfg["weather"]["provider"], choices=["nasa_power", "open_meteo"])
    w.add_argument("--start", type=int, default=cfg["apy"]["min_year"]); w.add_argument("--end", type=int, default=cfg["apy"]["max_year"])
    s = sub.add_parser("soil"); s.add_argument("--source", default=cfg["soil"]["source"], choices=["soilgrids", "csv"]); s.add_argument("--csv")
    args = ap.parse_args(argv)

    if args.cmd == "apy":
        import_apy(args.src)
    elif args.cmd == "centroids":
        df = import_centroids_csv(args.csv) if args.csv else fetch_centroids(district_pairs_from_raw())
        print(f"{df['latitude'].notna().sum()} of {len(df)} districts have coordinates; unresolved ones are dropped from training.")
    elif args.cmd == "weather":
        cen = pd.read_csv(RAW_DIR / "district_centroids.csv")
        print(download_weather(cen, args.start, args.end, args.provider, cfg["weather"]["request_sleep_s"]))
    elif args.cmd == "soil":
        if args.source == "csv":
            import_soil_csv(args.csv)
        else:
            download_soil(pd.read_csv(RAW_DIR / "district_centroids.csv"), cfg["soil"]["request_sleep_s"])


if __name__ == "__main__":
    main()