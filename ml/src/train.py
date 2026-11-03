"""End-to-end training: clean -> split -> tune (Optuna) -> cross-validate -> stack -> evaluate ->
quantile intervals (+ conformal calibration) -> SHAP -> save versioned artifacts. All runs logged to MLflow.

Usage (from repo root):  python -m ml.src.train --trials 15 --folds 5
"""
import argparse
import json
import os
import re
import time
import warnings
from datetime import UTC, datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from . import explain
from .config import (
    ARTIFACT_DIR,
    CATEGORICAL,
    CROP_PROFILE,
    ECONOMICS,
    MLFLOW_EXPERIMENT,
    MLFLOW_URI,
    NUMERIC,
    RANDOM_STATE,
    RAW_FEATURES,
    REPORT_DIR,
    TARGET,
)
from .data_gen import load_or_generate
from .features import ENGINEERED, DistrictFeatureEngineer, FeatureEngineer, FeatureSpec
from .meta import make_benchmarks, make_ranges
from .uncertainty import fit_interval_models, interval

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sns.set_theme(style="whitegrid")
EVAL = REPORT_DIR / "eval"
SEED = RANDOM_STATE


# ----------------------------------------------------------------------------- data
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop rows without target and remove gross per-crop yield outliers (robust MAD z-score > 6)."""
    n0 = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    df = df[df[TARGET] > 0]
    med = df.groupby("crop")[TARGET].transform("median")
    mad = (df[TARGET] - med).abs().groupby(df["crop"]).transform("median") * 1.4826
    z = (df[TARGET] - med) / mad.replace(0, np.nan)
    keep = z.abs() <= 6
    info = {"rows_in": n0, "rows_out": int(keep.sum()), "outliers_removed": int((~keep).sum())}
    return df[keep].reset_index(drop=True), info


def make_preprocessor(spec: FeatureSpec | None = None) -> Pipeline:
    """spec=None -> the original SYNTHETIC schema (unchanged behaviour). spec given -> REAL-data schema."""
    if spec is not None:
        ct = ColumnTransformer([
            ("num", StandardScaler(), spec.numeric + spec.engineered),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), spec.categorical),
        ], verbose_feature_names_out=True).set_output(transform="pandas")
        return Pipeline([("fe", DistrictFeatureEngineer(spec.categorical, spec.numeric)), ("prep", ct)])
    num = NUMERIC + [c for c in ENGINEERED]
    ct = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], verbose_feature_names_out=True).set_output(transform="pandas")
    return Pipeline([("fe", FeatureEngineer()), ("prep", ct)])


def build(model, spec: FeatureSpec | None = None) -> Pipeline:
    if spec is not None:
        pre = make_preprocessor(spec)
        return Pipeline([("fe", pre[0]), ("prep", pre[1]), ("model", model)])
    return Pipeline([("fe", FeatureEngineer()), ("prep", make_preprocessor()[1]), ("model", model)])


# ----------------------------------------------------------------------------- models / search spaces
def make_model(name: str, p: dict):
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Ridge":
        return Ridge(alpha=p.get("alpha", 1.0))
    if name == "Random Forest":
        return RandomForestRegressor(random_state=SEED, n_jobs=-1, **p)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=SEED, **p)
    if name == "XGBoost":
        return XGBRegressor(random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0, **p)
    if name == "LightGBM":
        return LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1, **p)
    if name == "CatBoost":
        return CatBoostRegressor(random_seed=SEED, verbose=0, thread_count=-1, **p)
    raise ValueError(name)


def suggest(name: str, t: optuna.Trial) -> dict:
    if name == "Ridge":
        return {"alpha": t.suggest_float("alpha", 1e-3, 100, log=True)}
    if name == "Random Forest":
        return {"n_estimators": t.suggest_int("n_estimators", 80, 200), "max_depth": t.suggest_int("max_depth", 6, 16),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 2, 8), "max_features": t.suggest_float("max_features", 0.4, 0.9)}
    if name == "Gradient Boosting":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 250), "learning_rate": t.suggest_float("learning_rate", 0.05, 0.25, log=True),
                "max_depth": t.suggest_int("max_depth", 3, 5), "subsample": t.suggest_float("subsample", 0.6, 0.9)}
    if name == "XGBoost":
        return {"n_estimators": t.suggest_int("n_estimators", 200, 700), "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "max_depth": t.suggest_int("max_depth", 3, 8), "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0), "reg_lambda": t.suggest_float("reg_lambda", 0.1, 10, log=True),
                "min_child_weight": t.suggest_int("min_child_weight", 1, 10)}
    if name == "LightGBM":
        return {"n_estimators": t.suggest_int("n_estimators", 200, 700), "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "num_leaves": t.suggest_int("num_leaves", 15, 100), "min_child_samples": t.suggest_int("min_child_samples", 5, 40),
                "subsample": t.suggest_float("subsample", 0.6, 1.0), "subsample_freq": 1,
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0), "reg_lambda": t.suggest_float("reg_lambda", 0.1, 10, log=True)}
    if name == "CatBoost":
        return {"iterations": t.suggest_int("iterations", 300, 800), "depth": t.suggest_int("depth", 4, 8),
                "learning_rate": t.suggest_float("learning_rate", 0.03, 0.2, log=True), "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 10)}
    return {}


SCORING = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error",
           "mape": "neg_mean_absolute_percentage_error"}


def metrics(y, p) -> dict:
    return {"r2": r2_score(y, p), "mae": mean_absolute_error(y, p), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mape": mean_absolute_percentage_error(y, p) * 100}


def cv_metrics(pipe, X, y, folds, cv=None) -> dict:
    """cv: optional precomputed list of (train_idx, val_idx) (e.g. chronological blocks). Default = shuffled KFold (synthetic mode)."""
    res = cross_validate(pipe, X, y, cv=cv if cv is not None else KFold(folds, shuffle=True, random_state=SEED), scoring=SCORING, n_jobs=1)
    out = {}
    for k in SCORING:
        v = res[f"test_{k}"] * (-1 if k != "r2" else 1)
        if k == "mape":
            v = v * 100
        out[f"cv_{k}"], out[f"cv_{k}_std"] = float(v.mean()), float(v.std())
    return out


def tune(name, X, y, trials, spec: FeatureSpec | None = None, cv_fn=None):
    """Optuna TPE search minimising 3-fold CV RMSE; returns best params and the trial history."""
    if name == "Linear Regression" or trials <= 0:
        return {}, []
    history = []
    if len(X) > 4000:  # tune on a subsample for speed; the final model is refit and cross-validated on all data
        X = X.sample(4000, random_state=SEED); y = y.loc[X.index]
    folds = cv_fn(X) if cv_fn is not None else KFold(3, shuffle=True, random_state=SEED)

    def objective(t):
        params = suggest(name, t)
        s = cross_validate(build(make_model(name, params), spec), X, y, cv=folds,
                           scoring="neg_root_mean_squared_error", n_jobs=1)["test_score"].mean()
        history.append(-s)
        return -s

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials if name != "Ridge" else min(trials, 10))
    return study.best_params | ({"subsample_freq": 1} if name == "LightGBM" else {}), history


# ----------------------------------------------------------------------------- plotting
def plot_comparison(tbl: pd.DataFrame, out=None):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    order = tbl.sort_values("test_rmse").index
    t = tbl.loc[order]
    ax[0].barh(t.index, t["test_r2"], color="#2f855a"); ax[0].set_title("Test R²"); ax[0].set_xlim(max(0, t["test_r2"].min() - .1), 1); ax[0].invert_yaxis()
    ax[1].barh(t.index, t["test_rmse"], color="#dd6b20"); ax[1].set_title("Test RMSE (t/ha)"); ax[1].invert_yaxis()
    ax[2].barh(t.index, t["cv_rmse"], xerr=t["cv_rmse_std"], color="#3182ce"); ax[2].set_title("5-fold CV RMSE (± std)"); ax[2].invert_yaxis()
    plt.tight_layout(); plt.savefig((out or EVAL) / "model_comparison.png", dpi=130); plt.close()


def plot_residuals(y, p, crops, name, out=None):
    r = y - p
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax[0, 0].scatter(p, y, s=8, alpha=.4, color="#2f855a"); lim = [0, max(y.max(), p.max())]
    ax[0, 0].plot(lim, lim, "r--"); ax[0, 0].set(xlabel="Predicted", ylabel="Actual", title=f"Actual vs predicted ({name})")
    ax[0, 1].scatter(p, r, s=8, alpha=.4, color="#3182ce"); ax[0, 1].axhline(0, color="r", ls="--"); ax[0, 1].set(xlabel="Predicted", ylabel="Residual", title="Residuals vs predicted")
    sns.histplot(r, kde=True, ax=ax[0, 2], color="#805ad5"); ax[0, 2].set(title="Residual distribution", xlabel="Residual (t/ha)")
    stats.probplot(r, dist="norm", plot=ax[1, 0]); ax[1, 0].set_title("Q-Q plot of residuals")
    d = pd.DataFrame({"crop": crops.values, "residual": r.values})
    sns.boxplot(data=d, x="crop", y="residual", ax=ax[1, 1], color="#9ae6b4"); ax[1, 1].tick_params(axis="x", rotation=45); ax[1, 1].axhline(0, color="r", ls="--"); ax[1, 1].set_title("Residuals by crop")
    ax[1, 2].scatter(p, np.abs(r), s=8, alpha=.4, color="#dd6b20"); ax[1, 2].set(xlabel="Predicted", ylabel="|Residual|", title="Heteroscedasticity check")
    plt.tight_layout(); plt.savefig((out or EVAL) / "residual_analysis.png", dpi=130); plt.close()


# ----------------------------------------------------------------------------- main
def next_version() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    nums = [int(m.group(1)) for f in ARTIFACT_DIR.glob("model_v*.joblib") if (m := re.match(r"model_v(\d+)\.joblib", f.name))]
    return max(nums, default=0) + 1


def main(trials: int, folds: int):
    """SYNTHETIC-data training (unchanged). Real-data training lives in train_real.py and is selected with --data real."""
    t0 = time.time()
    EVAL.mkdir(parents=True, exist_ok=True)
    df, source = load_or_generate()
    df, clean_info = clean(df)
    print(f"[data] {source}; {clean_info}")
    X, y = df[RAW_FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=df["crop"])

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    version = next_version()
    names = ["Linear Regression", "Ridge", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"]
    fitted, best_params, rows = {}, {}, []

    with mlflow.start_run(run_name=f"train_v{version}") as parent:
        mlflow.log_params({"data_source": source, "n_train": len(X_tr), "n_test": len(X_te), "cv_folds": folds, "optuna_trials": trials, **{f"clean_{k}": v for k, v in clean_info.items()}})

        CACHE = ARTIFACT_DIR.parent / ".cache"
        CACHE.mkdir(exist_ok=True)

        def run_model(name, estimator_factory, params, history):
            """Compute (or load from the resume cache) CV + test metrics, then log a nested MLflow run."""
            f = CACHE / f"{name.replace(' ', '_')}_t{trials}_f{folds}_n{len(X_tr)}.joblib"
            if f.exists():
                pipe, cvm, tm = joblib.load(f)
                print(f"[{name}] loaded from resume cache")
            else:
                pipe = build(estimator_factory())
                cvm = cv_metrics(pipe, X_tr, y_tr, folds)
                pipe.fit(X_tr, y_tr)
                tm = {f"test_{k}": v for k, v in metrics(y_te, pipe.predict(X_te)).items()}
                joblib.dump((pipe, cvm, tm), f)
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tag("model", name)
                mlflow.log_params({k: v for k, v in params.items()})
                for i, h in enumerate(history):
                    mlflow.log_metric("optuna_trial_cv_rmse", h, step=i)
                mlflow.log_metrics({**cvm, **tm})
            fitted[name] = pipe
            rows.append({"model": name, **cvm, **tm})
            print(f"[{name:18s}] CV RMSE {cvm['cv_rmse']:.4f}  test R2 {tm['test_r2']:.4f}  RMSE {tm['test_rmse']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

        def tuned(name):
            tf = CACHE / f"params_{name.replace(' ', '_')}_t{trials}_n{len(X_tr)}.joblib"
            if tf.exists():
                return joblib.load(tf)
            res = tune(name, X_tr, y_tr, trials)
            joblib.dump(res, tf)
            return res

        for name in names:
            params, hist = tuned(name)
            best_params[name] = params
            run_model(name, lambda n=name, p=params: make_model(n, p), params, hist)

        # Stacking ensemble: tuned XGB + LGBM + CatBoost, blended by a Ridge meta-learner.
        def stack():
            base = [(n, make_model(n, best_params[n])) for n in ["XGBoost", "LightGBM", "CatBoost"]]
            return StackingRegressor(base, final_estimator=RidgeCV(alphas=np.logspace(-3, 2, 12)), cv=3, n_jobs=1)
        run_model("Stacking Ensemble", stack, {"base": "XGB+LGBM+CatBoost", "meta": "RidgeCV"}, [])

        tbl = pd.DataFrame(rows).set_index("model")
        best_name = tbl["cv_rmse"].idxmin()  # selection on CV, never on the test set
        best = fitted[best_name]
        tbl.round(4).to_csv(EVAL / "model_comparison.csv")
        plot_comparison(tbl)
        pred = best.predict(X_te)
        plot_residuals(y_te.reset_index(drop=True), pd.Series(pred), X_te["crop"].reset_index(drop=True), best_name)
        per_crop = {c: {k: float(v) for k, v in metrics(y_te[X_te["crop"] == c], pred[(X_te["crop"] == c).values]).items()} for c in sorted(X_te["crop"].unique())}
        print(f"[best] {best_name} (selected by CV RMSE)")

        # ---- Uncertainty: relative-error quantile regression + conformal calibration (see ml/src/uncertainty.py)
        q_lo, q_hi, qhat = fit_interval_models(build, make_model("LightGBM", best_params["LightGBM"]), X_tr, y_tr)
        raw_lo, raw_hi = interval(pred, X_te, q_lo, q_hi, qhat, calibrated=False)
        lo_t, hi_t = interval(pred, X_te, q_lo, q_hi, qhat)
        unc = {"nominal_coverage": 0.8, "qhat_conformal_correction": qhat,
               "coverage_raw_quantile": float(np.mean((y_te.values >= raw_lo) & (y_te.values <= raw_hi))),
               "coverage_calibrated": float(np.mean((y_te.values >= lo_t) & (y_te.values <= hi_t))),
               "mean_interval_width_t_ha": float(np.mean(hi_t - lo_t)), "method": "quantile regression on relative error + CQR"}
        print(f"[uncertainty] {unc}")
        order = np.argsort(pred)[:: max(1, len(pred) // 120)]
        plt.figure(figsize=(10, 5.5))
        plt.fill_between(range(len(order)), lo_t[order], hi_t[order], alpha=.3, color="#2f855a", label="80% prediction interval")
        plt.plot(pred[order], color="#276749", label="Predicted"); plt.scatter(range(len(order)), y_te.values[order], s=14, color="#c05621", label="Actual")
        plt.xlabel("Test samples (sorted by prediction)"); plt.ylabel("Yield (t/ha)"); plt.title(f"Prediction intervals (empirical coverage {unc['coverage_calibrated']:.1%})"); plt.legend()
        plt.tight_layout(); plt.savefig(EVAL / "prediction_intervals.png", dpi=130); plt.close()

        # ---- Explainability: SHAP on the best tree model (stacking is explained through its strongest base learner)
        tree_names = ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"]
        exp_name = tbl.loc[tree_names, "cv_rmse"].idxmin()
        exp_pipe = fitted[exp_name]
        explainer = shap.TreeExplainer(exp_pipe[-1])
        sample = X_te.sample(min(1200, len(X_te)), random_state=SEED)
        Xt = exp_pipe[:-1].transform(sample)
        sv = np.asarray(explainer.shap_values(Xt))
        agg = explain.aggregate_shap(sv, list(Xt.columns))
        global_imp = agg.abs().mean().sort_values(ascending=False)
        plt.figure(); shap.summary_plot(sv, Xt, max_display=18, show=False); plt.title(f"SHAP summary ({exp_name})")
        plt.tight_layout(); plt.savefig(EVAL / "shap_summary_beeswarm.png", dpi=130, bbox_inches="tight"); plt.close()
        plt.figure(figsize=(8, 6)); global_imp.head(15)[::-1].plot.barh(color="#2f855a"); plt.xlabel("mean |SHAP| (t/ha)"); plt.title("Global feature importance (one-hot grouped)")
        plt.tight_layout(); plt.savefig(EVAL / "shap_global_importance.png", dpi=130); plt.close()
        base_v, loc = explain.local_explanation(explainer, exp_pipe, sample.iloc[[0]])
        top = pd.Series(loc).reindex(pd.Series(loc).abs().sort_values(ascending=False).index).head(8)[::-1]
        plt.figure(figsize=(8, 5)); plt.barh(top.index, top.values, color=["#2f855a" if v > 0 else "#c53030" for v in top.values])
        plt.title(f"Per-prediction SHAP example (baseline {base_v:.2f} t/ha)"); plt.xlabel("Contribution to yield (t/ha)")
        plt.tight_layout(); plt.savefig(EVAL / "shap_local_example.png", dpi=130); plt.close()

        # ---- Data-driven benchmarks used by the recommendation engine
        full = pd.concat([X, y], axis=1)
        bench, ranges = make_benchmarks(full), make_ranges(df)

        # ---- Save versioned artifacts
        bundle = {"version": version, "model_name": best_name, "pipeline": best, "q_low": q_lo, "q_high": q_hi, "qhat": qhat, "interval_mode": "relative",
                  "explainer_name": exp_name, "explainer_pipeline": exp_pipe, "feature_names": list(Xt.columns)}
        mp = ARTIFACT_DIR / f"model_v{version}.joblib"
        joblib.dump(bundle, mp, compress=3)
        meta = {"version": version, "trained_at": datetime.now(UTC).isoformat(), "data_source": source, "cleaning": clean_info,
                "best_model": best_name, "selection_criterion": "lowest 5-fold CV RMSE on training split", "explainer_model": exp_name,
                "n_train": len(X_tr), "n_test": len(X_te), "cv_folds": folds, "optuna_trials": trials,
                "metrics_test": {k.replace("test_", ""): float(v) for k, v in tbl.loc[best_name].items() if k.startswith("test_")},
                "metrics_cv": {k.replace("cv_", ""): float(v) for k, v in tbl.loc[best_name].items() if k.startswith("cv_")},
                "comparison": tbl.round(5).reset_index().to_dict("records"), "best_params": best_params, "per_crop_metrics": per_crop,
                "uncertainty": unc, "global_importance": {k: float(v) for k, v in global_imp.head(20).items()},
                "residual_summary": {"mean": float((y_te.values - pred).mean()), "std": float((y_te.values - pred).std()),
                                     "skew": float(stats.skew(y_te.values - pred))},
                "features": {"categorical": CATEGORICAL, "numeric": NUMERIC, "engineered": ENGINEERED},
                "benchmarks": bench, "feature_ranges": ranges, "crop_profiles": CROP_PROFILE, "economics": ECONOMICS,
                "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL}}
        (ARTIFACT_DIR / f"model_v{version}.json").write_text(json.dumps(meta, indent=2, default=float))
        (REPORT_DIR / "eval" / "metrics_summary.json").write_text(json.dumps({"best_model": best_name, **meta["metrics_test"], "uncertainty": unc}, indent=2))
        mlflow.log_metrics({f"best_{k}": v for k, v in meta["metrics_test"].items()})
        mlflow.set_tag("best_model", best_name)
        mlflow.log_artifacts(str(EVAL), artifact_path="evaluation")
        mlflow.log_artifact(str(ARTIFACT_DIR / f"model_v{version}.json"))
        mlflow.log_artifact(str(mp))
        print(f"[saved] {mp.name} ({mp.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s. mlflow run {parent.info.run_id}")
    print(tbl[["cv_rmse", "test_r2", "test_mae", "test_rmse", "test_mape"]].round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the crop-yield model on synthetic (default) or real district-level data.")
    ap.add_argument("--data", choices=["synthetic", "real"], default=os.environ.get("CROPAI_DATA_MODE", "synthetic"),
                    help="dataset mode (env CROPAI_DATA_MODE). 'real' needs data/real/processed/ (see docs/real-data.md)")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--models", nargs="*", help="(real mode) subset of models to train, e.g. --models Ridge LightGBM CatBoost")
    ap.add_argument("--final-fit", choices=["train", "train_val", "all"], help="(real mode) data the DEPLOYED model is fitted on")
    a = ap.parse_args()
    if a.data == "real":
        from .train_real import main as real_main
        try:
            real_main(a.trials, a.models, a.final_fit)
        except FileNotFoundError as e:  # no processed real dataset yet: say how to build it instead of silently using synthetic data
            raise SystemExit(f"\nERROR: {e}\nThe synthetic dataset is NOT used as a fallback in real mode.")
    else:
        main(a.trials, a.folds)