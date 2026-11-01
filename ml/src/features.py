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
