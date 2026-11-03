"""Validation of raw, cleaned and processed datasets. Produces a report; errors abort the pipeline, warnings are kept."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data_cleaning import standardise_columns
from .real_config import KEY_COLS, MIN_FEATURE_COVERAGE, NUMERIC_CANDIDATES, RANGES, TARGET, YIELD_BOUNDS


class ValidationError(Exception):
    pass


@dataclass
class Report:
    name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self):
        if self.errors:
            raise ValidationError(f"{self.name} validation failed:\n  - " + "\n  - ".join(self.errors))

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "errors": self.errors, "warnings": self.warnings, "stats": self.stats}

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=float))


def validate_raw_apy(df: pd.DataFrame) -> Report:
    r = Report("raw APY")
    r.stats["rows"] = len(df)
    if len(df) == 0:
        r.errors.append("file has no rows")
        return r
    try:
        std = standardise_columns(df)
    except ValueError as e:
        r.errors.append(str(e))
        return r
    for c in ("area", "production"):
        num = pd.to_numeric(std[c], errors="coerce")
        r.stats[f"{c}_non_numeric_pct"] = round(float(num.isna().mean() * 100), 2)
        if num.isna().mean() > 0.5:
            r.errors.append(f"more than half of '{c}' is missing or non-numeric: wrong file or column mapping?")
    r.stats["exact_duplicate_rows"] = int(df.duplicated().sum())
    if r.stats["exact_duplicate_rows"]:
        r.warnings.append(f"{r.stats['exact_duplicate_rows']} exact duplicate rows in the raw file")
    has_units = "area_unit" in std.columns and "production_unit" in std.columns
    r.stats["has_unit_columns"] = has_units
    if not has_units:
        r.warnings.append("no unit columns: area is assumed to be hectares and production tonnes. Cotton (bales) is excluded unless "
                          "[cleaning].unit_assumptions says otherwise. Verify against the source documentation.")
    return r


def validate_clean(df: pd.DataFrame) -> Report:
    r = Report("cleaned APY")
    r.stats["rows"] = len(df)
    need = ["region", "district", "crop", "season", "year", "area_ha", "production_t", TARGET]
    miss = [c for c in need if c not in df.columns]
    if miss:
        r.errors.append(f"missing columns {miss}")
        return r
    if len(df) == 0:
        r.errors.append("no rows survived cleaning")
        return r
    if df[need].isna().any().any():
        r.errors.append("NaN present in required columns after cleaning")
    if (df["area_ha"] <= 0).any():
        r.errors.append("non-positive cultivated area remains")
    if not np.allclose(df[TARGET], df["production_t"] / df["area_ha"], rtol=1e-6):
        r.errors.append("yield_t_ha != production_t / area_ha")
    if df.duplicated(["region", "district", "crop", "season", "year"]).any():
        r.errors.append("duplicate (region, district, crop, season, year) keys remain")
    for crop, (lo, hi) in YIELD_BOUNDS.items():
        g = df.loc[df["crop"] == crop, TARGET]
        if len(g) and ((g < lo).any() or (g > hi).any()):
            r.errors.append(f"{crop}: yield outside physical bounds [{lo}, {hi}] t/ha")
    per_year = df.groupby("year").size()
    r.stats["rows_per_year"] = per_year.to_dict()
    thin = per_year[per_year < 50]
    if len(thin):
        r.warnings.append(f"very few rows in years {list(thin.index)}: weak evidence for chronological evaluation")
    return r


def validate_processed(df: pd.DataFrame, min_rows: int = 200) -> Report:
    r = Report("processed (merged) dataset")
    r.stats["rows"] = len(df)
    need = KEY_COLS + [TARGET, "area_ha", "rainfall_mm", "temperature_c", "weather_source"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        r.errors.append(f"missing columns {miss}")
        return r
    if len(df) < min_rows:
        r.errors.append(f"only {len(df)} rows (< {min_rows}): too few to train and evaluate chronologically")
    if df.duplicated(KEY_COLS).any():
        r.errors.append("duplicate keys in merged dataset (join fan-out?)")
    for col, (lo, hi) in RANGES.items():
        if col in df.columns:
            bad = int(((df[col] < lo) | (df[col] > hi)).sum())
            if bad:
                r.errors.append(f"{col}: {bad} values outside plausible range [{lo}, {hi}]")
    cov = {c: round(float(df[c].notna().mean()), 3) for c in NUMERIC_CANDIDATES if c in df.columns}
    r.stats["feature_coverage"] = cov
    for c, v in cov.items():
        if v < MIN_FEATURE_COVERAGE:
            r.warnings.append(f"{c}: only {v:.0%} coverage: excluded from the real feature set")
    for c in ("rainfall_mm", "temperature_c"):
        if df[c].isna().any():
            r.errors.append(f"{c} has missing values: weather merge incomplete")
    r.stats["years"] = [int(df["year"].min()), int(df["year"].max())] if len(df) else []
    r.stats["rows_by_crop"] = df["crop"].value_counts().to_dict()
    # a timeline needs enough distinct years for train/val/test blocks
    if df["year"].nunique() < 8:
        r.errors.append(f"only {df['year'].nunique()} distinct years: need >= 8 for train/validation/test blocks")
    return r