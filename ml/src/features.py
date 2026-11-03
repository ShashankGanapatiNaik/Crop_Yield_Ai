"""Feature engineering as a scikit-learn transformer so it ships inside the saved pipeline.

Doing this inside the pipeline (not in a notebook) prevents train/serve skew: the API applies
exactly the same imputation, clipping and feature construction that training used.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import CATEGORICAL, IRRIGATION_MM, NUMERIC, SEASON_ORDINAL

ENGINEERED = ["rain_temp_interaction", "total_water_mm", "aridity_index", "heat_humidity_index",
              "n_p_ratio", "n_k_ratio", "p_k_ratio", "npk_total", "fert_per_soil_n",
              "ph_deviation", "season_ord", "log_area", "years_since_2000"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Missing-value imputation (per-crop median), IQR winsorising, and feature construction."""

    def __init__(self, clip_k: float = 3.0):
        self.clip_k = clip_k

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).copy()
        self.global_median_ = X[NUMERIC].median()
        self.crop_median_ = X.groupby("crop")[NUMERIC].median()
        filled = self._impute(X)
        q1, q3 = filled[NUMERIC].quantile(.25), filled[NUMERIC].quantile(.75)
        iqr = q3 - q1
        self.lower_ = q1 - self.clip_k * iqr
        self.upper_ = q3 + self.clip_k * iqr
        # year is a time index, not subject to winsorising
        self.lower_["year"], self.upper_["year"] = -np.inf, np.inf
        return self

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in NUMERIC:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            miss = X[col].isna()
            if miss.any():
                per_crop = X.loc[miss, "crop"].map(self.crop_median_[col])
                X.loc[miss, col] = per_crop.fillna(self.global_median_[col])
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._impute(pd.DataFrame(X).copy())
        X[NUMERIC] = X[NUMERIC].clip(self.lower_, self.upper_, axis=1)
        X["rain_temp_interaction"] = X["rainfall_mm"] * X["temperature_c"] / 100.0
        X["total_water_mm"] = X["rainfall_mm"] + X["irrigation"].map(IRRIGATION_MM).fillna(0) * 0.9
        X["aridity_index"] = X["rainfall_mm"] / (X["temperature_c"] + 10.0)
        X["heat_humidity_index"] = X["temperature_c"] * X["humidity_pct"] / 100.0
        eps = 1e-6
        X["n_p_ratio"] = X["nitrogen"] / (X["phosphorus"] + eps)
        X["n_k_ratio"] = X["nitrogen"] / (X["potassium"] + eps)
        X["p_k_ratio"] = X["phosphorus"] / (X["potassium"] + eps)
        X["npk_total"] = X["nitrogen"] + X["phosphorus"] + X["potassium"] + X["fertilizer_kg_ha"]
        X["fert_per_soil_n"] = X["fertilizer_kg_ha"] / (X["nitrogen"] + 1.0)
        X["ph_deviation"] = (X["ph"] - 6.8).abs()
        X["season_ord"] = X["season"].map(SEASON_ORDINAL).fillna(0).astype(int)
        X["log_area"] = np.log1p(X["area_ha"])
        X["years_since_2000"] = X["year"] - 2000
        return X[CATEGORICAL + NUMERIC + ENGINEERED]

    def get_feature_names_out(self, input_features=None):
        return np.array(CATEGORICAL + NUMERIC + ENGINEERED)


# ======================================================================================================
# REAL district-level data: a configurable feature spec + a feature engineer driven by it.
# (FeatureEngineer above is the SYNTHETIC-schema transformer and is intentionally left untouched so that
#  previously pickled models keep loading.)
# ======================================================================================================
from dataclasses import dataclass, field  # noqa: E402


def engineered_for(numeric: list[str]) -> list[str]:
    """Engineered feature names that can be built from the available numeric columns (deterministic)."""
    have, out = set(numeric), []
    if {"rainfall_mm", "temperature_c"} <= have:
        out += ["rain_temp_interaction", "aridity_index"]
    if "temperature_c" in have or "temp_max_c" in have:
        out.append("heat_stress_c")
    if {"soil_sand_pct", "soil_clay_pct"} <= have:
        out.append("soil_texture_ratio")
    if "soil_ph" in have:
        out.append("ph_deviation")
    if "year" in have:
        out.append("years_since_2000")
    return out + ["season_ord"]


@dataclass(frozen=True)
class FeatureSpec:
    """Which columns a model uses. Built from what the dataset genuinely contains; stored in the model bundle."""
    name: str
    categorical: list[str]
    numeric: list[str]
    required: list[str] = field(default_factory=list)  # inputs the API must receive

    @property
    def engineered(self) -> list[str]:
        return engineered_for(self.numeric)

    @property
    def raw(self) -> list[str]:
        return self.categorical + self.numeric

    def to_dict(self) -> dict:
        return {"name": self.name, "categorical": self.categorical, "numeric": self.numeric, "engineered": self.engineered, "required": self.required}


def make_district_spec(df: pd.DataFrame, min_coverage: float | None = None) -> FeatureSpec:
    """Use a candidate feature only if the real data actually has it for at least `min_coverage` of rows."""
    from .real_config import CATEGORICAL_REAL, MIN_FEATURE_COVERAGE, NUMERIC_CANDIDATES, REQUIRED_INPUTS_REAL
    cov = MIN_FEATURE_COVERAGE if min_coverage is None else min_coverage
    numeric = [c for c in NUMERIC_CANDIDATES if c in df.columns and df[c].notna().mean() >= cov]
    for must in ("rainfall_mm", "temperature_c"):
        if must not in numeric:
            raise ValueError(f"'{must}' is missing or has <{cov:.0%} coverage in the real dataset: weather merge failed?")
    return FeatureSpec("real_district", list(CATEGORICAL_REAL), numeric, list(REQUIRED_INPUTS_REAL))


class DistrictFeatureEngineer(BaseEstimator, TransformerMixin):
    """Per-crop median imputation -> winsorising -> engineered features, for whatever columns the spec lists."""

    def __init__(self, categorical: list[str], numeric: list[str], clip_k: float = 5.0):
        self.categorical = categorical
        self.numeric = numeric
        self.clip_k = clip_k

    def _frame(self, X) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for c in self.numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce") if c in X.columns else np.nan
        return X

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        for c in self.numeric:
            miss = X[c].isna()
            if miss.any():
                X.loc[miss, c] = X.loc[miss, "crop"].map(self.crop_median_[c]).fillna(self.global_median_[c])
        return X

    def fit(self, X, y=None):
        X = self._frame(X)
        self.global_median_ = X[self.numeric].median().fillna(0.0)
        self.crop_median_ = X.groupby("crop")[self.numeric].median()
        f = self._impute(X.copy())
        q1, q3 = f[self.numeric].quantile(.25), f[self.numeric].quantile(.75)
        iqr = q3 - q1
        self.lower_, self.upper_ = q1 - self.clip_k * iqr, q3 + self.clip_k * iqr
        for c in ("year", "latitude", "longitude"):
            if c in self.lower_.index:
                self.lower_[c], self.upper_[c] = -np.inf, np.inf
        self.engineered_ = engineered_for(self.numeric)
        return self

    def transform(self, X) -> pd.DataFrame:
        X = self._impute(self._frame(X))
        X[self.numeric] = X[self.numeric].clip(self.lower_, self.upper_, axis=1)
        e = set(self.engineered_)
        if "rain_temp_interaction" in e:
            X["rain_temp_interaction"] = X["rainfall_mm"] * X["temperature_c"] / 100.0
            X["aridity_index"] = X["rainfall_mm"] / (X["temperature_c"] + 10.0)
        if "heat_stress_c" in e:  # degrees of mean daily maximum above 35C (or mean temp above 30C when no Tmax)
            X["heat_stress_c"] = (X["temp_max_c"] - 35.0).clip(lower=0) if "temp_max_c" in X.columns and "temp_max_c" in self.numeric else (X["temperature_c"] - 30.0).clip(lower=0)
        if "soil_texture_ratio" in e:
            X["soil_texture_ratio"] = X["soil_sand_pct"] / (X["soil_clay_pct"] + 1.0)
        if "ph_deviation" in e:
            X["ph_deviation"] = (X["soil_ph"] - 6.8).abs()
        if "years_since_2000" in e:
            X["years_since_2000"] = X["year"] - 2000
        X["season_ord"] = X["season"].map(SEASON_ORDINAL).fillna(0).astype(int)
        return X[self.categorical + self.numeric + self.engineered_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.categorical + self.numeric + engineered_for(self.numeric))