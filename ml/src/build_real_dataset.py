"""Clean -> validate -> merge -> validate -> save the processed real dataset.  python -m ml.src.build_real_dataset

Prerequisites (see docs/real-data.md): raw APY file, district centroids, cached weather, and soil (unless [soil].source = "none").
This step needs NO internet: it only reads data/real/raw.
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from . import data_download as dd
from .data_cleaning import clean_apy
from .data_merge import merge_dataset, save_processed
from .data_validation import validate_clean, validate_processed, validate_raw_apy
from .real_config import PROCESSED_DIR, RAW_DIR, load_config

log = logging.getLogger("build_real_dataset")


def build(cfg: dict | None = None, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR, provider: str | None = None, min_rows: int = 200) -> dict:
    cfg = cfg or load_config()
    provider = provider or cfg["weather"]["provider"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    apy_path = raw_dir / "apy_raw.csv"
    if not apy_path.exists():
        raise FileNotFoundError(f"{apy_path} missing. Run: python -m ml.src.data_download apy --src <your APY csv>")

    raw = pd.read_csv(apy_path, low_memory=False)
    r_raw = validate_raw_apy(raw); r_raw.raise_if_errors()
    clean, clean_rep = clean_apy(raw, cfg)
    r_clean = validate_clean(clean); r_clean.raise_if_errors()
    clean.to_csv(processed_dir / "apy_clean.csv", index=False)

    cen_path = raw_dir / "district_centroids.csv"
    if not cen_path.exists():
        raise FileNotFoundError(f"{cen_path} missing. Run: python -m ml.src.data_download centroids [--csv ...]")
    centroids = pd.read_csv(cen_path)

    soil_source = cfg["soil"]["source"]
    soil = None
    if soil_source != "none":
        sp = raw_dir / "soil_district.csv"
        if not sp.exists():
            raise FileNotFoundError(f"{sp} missing. Run: python -m ml.src.data_download soil (or set [soil].source = \"none\" to train without soil)")
        soil = pd.read_csv(sp)
        soil_source = str(soil["soil_source"].iloc[0]) if "soil_source" in soil.columns and len(soil) else soil_source

    loader = lambda st, d: dd.load_weather(provider, st, d, raw_dir)  # noqa: E731
    merged, merge_rep = merge_dataset(clean, centroids, loader, soil, provider, soil_source, cfg)
    r_proc = validate_processed(merged, min_rows=min_rows); r_proc.raise_if_errors()

    reports = {"raw": r_raw.to_dict(), "clean": r_clean.to_dict(), "processed": r_proc.to_dict()}
    (processed_dir / "validation_report.json").write_text(json.dumps(reports, indent=2, default=float))
    prov = json.loads((raw_dir / "apy_provenance.json").read_text()) if (raw_dir / "apy_provenance.json").exists() else {}
    meta = save_processed(merged, merge_rep, {"cleaning_report": clean_rep, "apy_provenance": prov, "weather_provider": provider, "config": cfg},
                          csv=processed_dir / "district_yield.csv", meta=processed_dir / "district_yield.meta.json")
    log.info("processed dataset v%s: %s rows, %s districts, years %s", meta["dataset_version"], meta["rows"], merge_rep["districts_out"], meta["years"])
    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    m = build()
    print(json.dumps({k: m[k] for k in ("dataset_version", "rows", "years")}, indent=2))