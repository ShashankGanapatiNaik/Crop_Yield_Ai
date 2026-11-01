"""Metadata helpers shared by full training and quick_train."""
import pandas as pd

from .config import NUMERIC, TARGET


def make_benchmarks(full: pd.DataFrame) -> dict:
    """Per-crop yield percentiles and what top-quartile farms look like (feeds the data-driven recommendations)."""
    bench = {}
    for crop, g in full.groupby("crop"):
        top_q = g[g[TARGET] >= g[TARGET].quantile(.75)]
        bench[crop] = {"median_yield": float(g[TARGET].median()), "p75_yield": float(g[TARGET].quantile(.75)),
                       "p90_yield": float(g[TARGET].quantile(.9)),
                       "top_quartile_median": {c: float(top_q[c].median()) for c in ["ph", "nitrogen", "phosphorus", "potassium", "rainfall_mm", "temperature_c", "fertilizer_kg_ha"]},
                       "best_irrigation": top_q["irrigation"].mode().iat[0], "best_soil": top_q["soil_type"].mode().iat[0]}
    return bench


def make_ranges(df: pd.DataFrame) -> dict:
    return {c: {"min": float(df[c].quantile(.005)), "max": float(df[c].quantile(.995)), "median": float(df[c].median())}
            for c in NUMERIC if c != "year"}
