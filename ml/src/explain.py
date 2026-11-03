"""SHAP helpers shared by training (global plots) and the backend (per-prediction explanations)."""
import numpy as np
import pandas as pd

from .config import CATEGORICAL


def group_of(col: str, categorical: list[str] | None = None) -> str:
    """Map a transformed column ('cat__crop_Rice', 'num__ph') back to its original feature name."""
    if col.startswith("cat__"):
        rest = col[5:]
        for c in sorted(categorical or CATEGORICAL, key=len, reverse=True):
            if rest.startswith(c + "_"):
                return c
    return col.replace("num__", "")


def aggregate_shap(values: np.ndarray, names: list[str], categorical: list[str] | None = None) -> pd.DataFrame:
    """Sum SHAP values of one-hot columns back into their parent feature (SHAP is additive)."""
    groups = [group_of(n, categorical) for n in names]
    df = pd.DataFrame(values, columns=names)
    return df.T.groupby(groups).sum().T


def local_explanation(explainer, pipeline, row: pd.DataFrame, categorical: list[str] | None = None):
    """Return (base_value, {feature: shap_value}) for a single raw-input row."""
    Xt = pipeline[:-1].transform(row)
    sv = explainer.shap_values(Xt)
    sv = np.asarray(sv).reshape(len(row), -1)
    agg = aggregate_shap(sv, list(Xt.columns), categorical)
    base = float(np.ravel(explainer.expected_value)[0])
    return base, agg.iloc[0].to_dict()