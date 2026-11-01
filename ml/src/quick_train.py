"""Tiny, fast model with the SAME artifact layout as train.py. Used by CI and the pytest suite
so tests never depend on a multi-minute training run:  python -m ml.src.quick_train --out /tmp/models
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score

from .config import CATEGORICAL, CROP_PROFILE, ECONOMICS, RAW_FEATURES, TARGET
from .data_gen import generate
from .features import ENGINEERED
from .meta import make_benchmarks, make_ranges
from .train import build, clean
from .uncertainty import fit_interval_models


def run(out: Path, n: int = 2500) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    df, info = clean(generate(n))
    X, y = df[RAW_FEATURES], df[TARGET]
    est = lambda: LGBMRegressor(n_estimators=120, learning_rate=0.1, num_leaves=24, random_state=0, n_jobs=-1, verbose=-1)
    pipe = build(est()).fit(X, y)
    q_lo, q_hi, qhat = fit_interval_models(build, est(), X, y)
    Xt = pipe[:-1].transform(X.head(200))
    bundle = {"version": 1, "model_name": "LightGBM (quick)", "pipeline": pipe, "q_low": q_lo, "q_high": q_hi, "qhat": qhat,
              "interval_mode": "relative", "explainer_name": "LightGBM (quick)", "explainer_pipeline": pipe, "feature_names": list(Xt.columns)}
    joblib.dump(bundle, out / "model_v1.joblib")
    r2 = float(r2_score(y, pipe.predict(X)))
    meta = {"version": 1, "trained_at": "quick", "data_source": "synthetic (quick_train)", "cleaning": info, "best_model": "LightGBM (quick)",
            "selection_criterion": "n/a", "explainer_model": "LightGBM (quick)", "n_train": len(X), "n_test": 0, "cv_folds": 0, "optuna_trials": 0,
            "metrics_test": {"r2": r2, "mae": 0, "rmse": 0, "mape": 0}, "metrics_cv": {"r2": r2}, "comparison": [{"model": "LightGBM (quick)", "test_r2": r2}],
            "best_params": {}, "per_crop_metrics": {}, "uncertainty": {"coverage_calibrated": 0.8}, "global_importance": {}, "residual_summary": {},
            "features": {"categorical": CATEGORICAL, "engineered": ENGINEERED}, "benchmarks": make_benchmarks(pd.concat([X, y], axis=1)),
            "feature_ranges": make_ranges(df), "crop_profiles": CROP_PROFILE, "economics": ECONOMICS}
    (out / "model_v1.json").write_text(json.dumps(meta, indent=2, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="ml/artifacts_quick")
    print("wrote", run(Path(ap.parse_args().out)))
