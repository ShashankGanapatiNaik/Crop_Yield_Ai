import numpy as np
import pandas as pd

from ml.src.config import RAW_FEATURES
from ml.src.data_gen import generate
from ml.src.features import ENGINEERED, FeatureEngineer
from ml.src.train import clean


def test_feature_engineer_imputes_clips_and_adds_features():
    df = generate(600)
    X = df[RAW_FEATURES].copy()
    fe = FeatureEngineer().fit(X)
    X.loc[X.index[:5], "ph"] = np.nan
    X.loc[X.index[5], "rainfall_mm"] = 1e6  # absurd value must be winsorised
    out = fe.transform(X)
    assert not out.isna().any().any()
    assert set(ENGINEERED) <= set(out.columns)
    assert out["rainfall_mm"].max() < 1e5
    assert (out["n_p_ratio"] > 0).all()


def test_cleaning_removes_gross_yield_outliers():
    df = generate(1500)
    cleaned, info = clean(df)
    assert info["outliers_removed"] >= 1 and cleaned["yield_t_ha"].gt(0).all()


def test_unseen_category_does_not_crash_pipeline(client):
    from backend.app.services.model_service import get_model_service
    svc = get_model_service()
    row = {**dict(crop="Rice", region="Karnataka", season="Kharif", soil_type="Red", irrigation="Canal", ph=6, nitrogen=100, phosphorus=30,
                  potassium=120, rainfall_mm=900, temperature_c=27, humidity_pct=70, area_ha=1, fertilizer_kg_ha=80, year=2024), "region": "Atlantis"}
    p, lo, hi = svc.predict_frame(pd.DataFrame([row])[RAW_FEATURES])
    assert lo[0] <= p[0] <= hi[0]
