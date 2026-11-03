"""Prediction intervals: quantile regression on the relative error + conformal calibration.

Why relative error?  Yield noise is multiplicative (a 5 t/ha field varies more in absolute terms than a
1 t/ha field), and training data is sparse near high-yield optima, so absolute-scale quantile trees
extrapolate badly there. Instead:
  1. get out-of-fold predictions y_hat_oof from a fast boosted model (no leakage),
  2. fit LightGBM quantile regressors (alpha=0.1, 0.9) for the ratio r = y / y_hat_oof,
  3. conformalised quantile regression (CQR) on a held-out calibration split corrects the band so the
     nominal 80% coverage holds in finite samples,
  4. at inference:  interval = y_hat * [r_lo - q_hat, r_hi + q_hat].
"""
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from .config import RANDOM_STATE

R_MIN, R_MAX = 0.2, 3.0


def _quantile_pipe(build, alpha: float):
    return build(LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=250, learning_rate=0.05, num_leaves=15,
                               min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               reg_lambda=5.0, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))


def fit_interval_models(build, surrogate_estimator, X, y, coverage: float = 0.8):
    """Returns (q_low_pipeline, q_high_pipeline, q_hat). `build` wraps an estimator in the shared preprocessing pipeline."""
    oof = cross_val_predict(build(surrogate_estimator), X, y, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))
    oof = np.clip(oof, 1e-3, None)
    ratio = np.clip(y.values / oof, R_MIN, R_MAX)
    idx = np.arange(len(X))
    fit_i, cal_i = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE, stratify=X["crop"])
    alpha = (1 - coverage) / 2
    q_lo = _quantile_pipe(build, alpha).fit(X.iloc[fit_i], ratio[fit_i])
    q_hi = _quantile_pipe(build, 1 - alpha).fit(X.iloc[fit_i], ratio[fit_i])
    lo, hi = np.sort(np.vstack([q_lo.predict(X.iloc[cal_i]), q_hi.predict(X.iloc[cal_i])]), axis=0)
    scores = np.maximum(lo - ratio[cal_i], ratio[cal_i] - hi)
    n = len(scores)
    q_hat = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * coverage) / n)))
    return q_lo, q_hi, q_hat


def interval(point, X, q_lo, q_hi, q_hat: float, calibrated: bool = True):
    """Vectorised interval around the point prediction."""
    a, b = np.sort(np.vstack([q_lo.predict(X), q_hi.predict(X)]), axis=0)
    k = q_hat if calibrated else 0.0
    lo = point * np.clip(a - k, 0.05, None)
    hi = point * np.maximum(b + k, a)
    return np.minimum(lo, point), np.maximum(hi, point)


def fit_interval_models_temporal(build, surrogate_estimator, X_tr, y_tr, X_cal, y_cal, point_cal, coverage: float = 0.8):
    """Chronological variant for real data.
    Ratio-quantile models are fitted on the TRAINING years (ratios from out-of-fold surrogate predictions); the conformal
    correction q_hat is computed on LATER, held-out calibration years using the real point predictions, so the interval is
    calibrated the way it will be used: on years the model has not seen."""
    oof = np.clip(cross_val_predict(build(surrogate_estimator), X_tr, y_tr, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE)), 1e-3, None)
    ratio = np.clip(np.asarray(y_tr) / oof, R_MIN, R_MAX)
    alpha = (1 - coverage) / 2
    q_lo = _quantile_pipe(build, alpha).fit(X_tr, ratio)
    q_hi = _quantile_pipe(build, 1 - alpha).fit(X_tr, ratio)
    lo, hi = np.sort(np.vstack([q_lo.predict(X_cal), q_hi.predict(X_cal)]), axis=0)
    r_cal = np.clip(np.asarray(y_cal) / np.clip(np.asarray(point_cal), 1e-3, None), R_MIN, R_MAX)
    scores = np.maximum(lo - r_cal, r_cal - hi)
    n = len(scores)
    q_hat = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * coverage) / n)))
    return q_lo, q_hi, q_hat