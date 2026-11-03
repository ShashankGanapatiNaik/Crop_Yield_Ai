"""Train and evaluate on REAL district-level data (chronological protocol).   python -m ml.src.train --data real

Protocol (no future information reaches the past):
  train  = earliest crop years        -> hyper-parameter tuning (expanding-window year blocks INSIDE train only) and fitting
  val    = the next [split].val_years  -> model SELECTION and interval calibration
  test   = the latest [split].test_years -> reported once, never used for any decision
Naive baselines (crop/state/season climatology, district-crop history) and the synthetic-data model's own test metrics are
reported next to the real-data models so the value added by weather and soil can be judged honestly.
Artifacts are versioned separately (real_district_vN) and NEVER overwrite model_vN.
"""
import json
import logging
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from . import explain
from .config import ARTIFACT_DIR, CROP_PROFILE, ECONOMICS, MLFLOW_URI, RANDOM_STATE, REPORT_DIR
from .data_merge import load_processed
from .features import FeatureSpec, make_district_spec
from .real_config import PROVENANCE, SOIL_COLS, TARGET, load_config
from .train import build, cv_metrics, make_model, metrics, plot_comparison, plot_residuals, tune
from .uncertainty import fit_interval_models_temporal, interval

log = logging.getLogger("train_real")
MLFLOW_EXPERIMENT_REAL = "crop-yield-real-district"
ALL_MODELS = ["Linear Regression", "Ridge", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"]
TREE_MODELS = ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"]


# ------------------------------------------------------------------------------------------ splitting
def chronological_split(df: pd.DataFrame, test_years: int, val_years: int) -> dict:
    """Index arrays for train/val/test where every train year < every val year < every test year."""
    years = sorted(df["year"].unique())
    if len(years) < test_years + val_years + 4:
        raise ValueError(f"{len(years)} distinct years is too few for {test_years} test + {val_years} validation years (need >= 4 training years).")
    test_y, val_y = years[-test_years:], years[-(test_years + val_years):-test_years]
    train_y = [y for y in years if y not in set(test_y) | set(val_y)]
    ix = lambda ys: np.flatnonzero(df["year"].isin(ys).values)  # noqa: E731
    out = {"train": ix(train_y), "val": ix(val_y), "test": ix(test_y), "years": {"train": train_y, "val": val_y, "test": test_y}}
    assert max(train_y) < min(val_y) <= max(val_y) < min(test_y), "temporal leakage in split"
    return out


def blocked_year_splits(years, n_splits: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds by year: fold k trains on blocks 0..k-1 and validates on block k (always strictly later)."""
    years = np.asarray(years)
    blocks = np.array_split(np.array(sorted(set(years))), n_splits + 1)
    folds = []
    for k in range(1, len(blocks)):
        tr = np.flatnonzero(np.isin(years, np.concatenate(blocks[:k])))
        va = np.flatnonzero(np.isin(years, blocks[k]))
        if len(tr) and len(va):
            assert years[tr].max() < years[va].min()
            folds.append((tr, va))
    return folds


# ------------------------------------------------------------------------------------------ evaluation helpers
def _r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()) if len(y) > 1 and ((y - y.mean()) ** 2).sum() > 0 else None


def group_metrics(df: pd.DataFrame, pred: np.ndarray, col: str, min_rows: int) -> tuple[dict, int]:
    out, small = {}, 0
    for v, g in df.assign(_p=pred).groupby(col):
        if len(g) < min_rows:
            small += 1
            continue
        y, p = g[TARGET].values, g["_p"].values
        out[str(v)] = {"n": int(len(g)), "r2": _r2(y, p), "mae": float(np.abs(y - p).mean()), "rmse": float(np.sqrt(((y - p) ** 2).mean())),
                       "mape": float((np.abs(y - p) / np.maximum(y, 1e-6)).mean() * 100)}
    return out, small


def baseline_predictions(train: pd.DataFrame, target: pd.DataFrame) -> dict:
    """Two honest yardsticks that use NO weather or soil."""
    def grp(cols, fallbacks):
        pred = pd.Series(np.nan, index=target.index)
        for c in [cols] + fallbacks:
            m = train.groupby(c)[TARGET].mean().rename("m").reset_index()
            fill = target[c].merge(m, on=c, how="left")["m"].set_axis(target.index)
            pred = pred.fillna(fill)
        return pred.fillna(train[TARGET].mean()).values
    clim = grp(["crop", "region", "season"], [["crop", "season"], ["crop"]])
    dist = grp(["crop", "region", "district", "season"], [["crop", "region", "season"], ["crop", "season"], ["crop"]])
    return {"Baseline: crop/state/season mean": clim, "Baseline: district-crop history mean": dist}


def _clean_json(o):
    """NaN/inf are not valid JSON (the API would crash on them): convert to null."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_json(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return _clean_json(o.item())
    return o


def next_real_version(artifact_dir: Path) -> int:
    nums = [int(m.group(1)) for f in artifact_dir.glob("real_district_v*.joblib") if (m := re.match(r"real_district_v(\d+)\.joblib", f.name))]
    return max(nums, default=0) + 1


def synthetic_reference(artifact_dir: Path) -> dict | None:
    files = sorted(artifact_dir.glob("model_v*.json"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    if not files:
        return None
    m = json.loads(files[-1].read_text())
    return {"model_file": files[-1].name, "best_model": m.get("best_model"), "metrics_test": m.get("metrics_test"), "data_source": m.get("data_source"),
            "note": "Trained and tested on SIMULATED data with farm-level inputs (N, P, K, irrigation, fertiliser). The numbers are NOT comparable "
                    "with the real-data models above: different data, different target noise, different inputs."}


def district_tables(df: pd.DataFrame, spec: FeatureSpec) -> tuple[dict, dict, dict]:
    """Per-district static attributes used to fill soil/lat/lon at serving time (regional estimates, flagged as such)."""
    static = [c for c in SOIL_COLS + ["latitude", "longitude"] if c in spec.numeric]
    d = df.groupby(["region", "district"])[static].median()
    dist = {f"{r}|{n}": {k: (None if pd.isna(v) else float(v)) for k, v in row.items()} for (r, n), row in d.iterrows()}
    reg = df.groupby("region")[static].median()
    region = {r: {k: (None if pd.isna(v) else float(v)) for k, v in row.items()} for r, row in reg.iterrows()}
    by_region = {r: sorted(g["district"].unique().tolist()) for r, g in df.groupby("region")}
    return dist, region, by_region


# ------------------------------------------------------------------------------------------ main
def main(trials: int = 15, models: list[str] | None = None, final_fit: str | None = None, *, df: pd.DataFrame | None = None, data_meta: dict | None = None,
         artifact_dir: Path | None = None, report_dir: Path | None = None, cfg: dict | None = None, mlflow_uri: str | None = None,
         allow_fixture: bool = False) -> dict:
    t0 = time.time()
    cfg = cfg or load_config()
    artifact_dir = Path(artifact_dir or ARTIFACT_DIR)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    final_fit = final_fit or cfg["train"]["final_fit"]
    if df is None:
        df, data_meta = load_processed()
    data_meta = data_meta or {}
    # Lineage guard: a model may only be labelled REAL if every row is an observed APY record. Test fixtures are refused unless a test says so.
    if not allow_fixture and not ("yield_source" in df.columns and (df["yield_source"] == "apy_observed").all()):
        raise ValueError("Refusing to train a 'real_district' model: not every row is marked yield_source='apy_observed'. "
                         "Build the dataset with ml.src.build_real_dataset from real APY records.")
    df = df.sort_values(["year", "region", "district", "crop", "season"]).reset_index(drop=True)
    spec = make_district_spec(df)
    sp = chronological_split(df, cfg["split"]["test_years"], cfg["split"]["val_years"])
    tr, va, te = df.iloc[sp["train"]], df.iloc[sp["val"]], df.iloc[sp["test"]]
    Xtr, ytr, Xva, yva, Xte, yte = tr[spec.raw], tr[TARGET], va[spec.raw], va[TARGET], te[spec.raw], te[TARGET]
    version = next_real_version(artifact_dir)
    name = f"real_district_v{version}"
    rdir = Path(report_dir or REPORT_DIR / "real") / name
    rdir.mkdir(parents=True, exist_ok=True)
    names = [m for m in (models or ALL_MODELS) if m in ALL_MODELS]
    print(f"[data] {len(df)} rows | train {len(tr)} {sp['years']['train'][0]}-{sp['years']['train'][-1]} | val {len(va)} {sp['years']['val']} | test {len(te)} {sp['years']['test']}")
    print(f"[features] {spec.raw} + engineered {spec.engineered}")

    cv_fn = lambda X: blocked_year_splits(X["year"].values, 3)  # noqa: E731
    mlflow.set_tracking_uri(mlflow_uri or MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_REAL)
    fitted, best_params, rows, val_pred = {}, {}, [], {}

    with mlflow.start_run(run_name=name) as parent:
        mlflow.set_tags({"data_mode": "real", "model_name": name, "dataset_version": str(data_meta.get("dataset_version", "unknown")),
                         "yield_provenance": PROVENANCE["yield"], "weather_provenance": PROVENANCE["weather"], "soil_provenance": PROVENANCE["soil"]})
        mlflow.log_params({"n_train": len(tr), "n_val": len(va), "n_test": len(te), "train_years": f"{sp['years']['train'][0]}-{sp['years']['train'][-1]}",
                           "val_years": str(sp["years"]["val"]), "test_years": str(sp["years"]["test"]), "optuna_trials": trials, "final_fit": final_fit,
                           "dataset_version": data_meta.get("dataset_version", "unknown"), "weather_source": str(df["weather_source"].iloc[0]) if "weather_source" in df else "",
                           "soil_source": str(df["soil_source"].iloc[0]) if "soil_source" in df else ""})
        (rdir / "feature_spec.json").write_text(json.dumps(spec.to_dict(), indent=2))
        (rdir / "dataset_meta.json").write_text(json.dumps(_clean_json({k: v for k, v in data_meta.items() if k != "config"}), indent=2, default=str))
        mlflow.log_artifact(str(rdir / "feature_spec.json")); mlflow.log_artifact(str(rdir / "dataset_meta.json"))

        def evaluate(label, estimator_factory, params, history):
            with mlflow.start_run(run_name=label, nested=True):
                pipe = build(estimator_factory(), spec)
                cvm = cv_metrics(pipe, Xtr, ytr, 3, cv=cv_fn(Xtr))                 # chronological blocks inside TRAIN only
                pipe.fit(Xtr, ytr)
                pv, pt = pipe.predict(Xva), pipe.predict(Xte)
                vm = {f"val_{k}": v for k, v in metrics(yva, pv).items()}
                tm = {f"test_{k}": v for k, v in metrics(yte, pt).items()}
                mlflow.set_tag("model", label)
                mlflow.log_params({k: v for k, v in params.items()})
                for i, h in enumerate(history):
                    mlflow.log_metric("optuna_trial_cv_rmse", h, step=i)
                mlflow.log_metrics({**cvm, **vm, **tm})
            fitted[label], val_pred[label] = pipe, pv
            rows.append({"model": label, **cvm, **vm, **tm})
            print(f"[{label:18s}] cv RMSE {cvm['cv_rmse']:.4f} | val RMSE {vm['val_rmse']:.4f} R2 {vm['val_r2']:.3f} | test RMSE {tm['test_rmse']:.4f} R2 {tm['test_r2']:.3f} ({time.time() - t0:.0f}s)", flush=True)

        for label in names:
            params, hist = tune(label, Xtr, ytr, trials, spec=spec, cv_fn=cv_fn)    # tuning sees TRAIN years only
            best_params[label] = params
            evaluate(label, lambda n=label, p=params: make_model(n, p), params, hist)
        base = [n for n in ("XGBoost", "LightGBM", "CatBoost") if n in fitted]
        if len(base) >= 2:
            def stack():
                # unshuffled KFold on year-sorted rows -> contiguous time blocks for the stacker's out-of-fold predictions
                return StackingRegressor([(n, make_model(n, best_params[n])) for n in base], final_estimator=RidgeCV(alphas=np.logspace(-3, 2, 12)),
                                         cv=KFold(3, shuffle=False), n_jobs=1)
            evaluate("Stacking Ensemble", stack, {"base": "+".join(base), "meta": "RidgeCV"}, [])
        tbl = pd.DataFrame(rows).set_index("model")
        best_name = tbl["val_rmse"].idxmin()                                      # selection on VALIDATION years, never on test
        best = fitted[best_name]
        pred_te, pred_va = best.predict(Xte), val_pred[best_name]
        print(f"[best] {best_name} (selected by validation RMSE)")

        # ---- baselines and the synthetic reference
        base_rows = []
        for label, p in baseline_predictions(tr, te).items():
            m = {f"test_{k}": v for k, v in metrics(yte, p).items()}
            pv = baseline_predictions(tr, va)[label]
            base_rows.append({"model": label, **{f"val_{k}": v for k, v in metrics(yva, pv).items()}, **m, "is_baseline": True})
        syn = synthetic_reference(artifact_dir)

        # ---- breakdowns on the untouched test years
        minr = cfg["split"]["min_rows_per_group_report"]
        per_crop, small_c = group_metrics(te, pred_te, "crop", minr)
        per_region, small_r = group_metrics(te, pred_te, "region", minr)
        resid = yte.values - pred_te

        # ---- uncertainty: ratio-quantile models on TRAIN, conformal q_hat on LATER validation years
        surrogate = make_model("LightGBM", best_params.get("LightGBM", {})) if "LightGBM" in fitted else make_model("Ridge", {})
        q_lo, q_hi, qhat = fit_interval_models_temporal(lambda m: build(m, spec), surrogate, Xtr, ytr, Xva, yva, pred_va)
        raw_lo, raw_hi = interval(pred_te, Xte, q_lo, q_hi, qhat, calibrated=False)
        lo_t, hi_t = interval(pred_te, Xte, q_lo, q_hi, qhat)
        unc = {"nominal_coverage": 0.8, "qhat_conformal_correction": qhat, "coverage_raw_quantile": float(np.mean((yte.values >= raw_lo) & (yte.values <= raw_hi))),
               "coverage_calibrated": float(np.mean((yte.values >= lo_t) & (yte.values <= hi_t))), "mean_interval_width_t_ha": float(np.mean(hi_t - lo_t)),
               "method": "ratio quantile regression (train years) + conformal correction (validation years); evaluated on test years"}
        print(f"[uncertainty] {unc}")

        # ---- explainability (best tree model by validation RMSE)
        trees = [t for t in TREE_MODELS if t in tbl.index]
        if trees:
            exp_name = tbl.loc[trees, "val_rmse"].idxmin()
            exp_pipe = fitted[exp_name]
        else:  # e.g. --models Ridge: SHAP's tree explainer needs a tree model, so fit one purely to explain (never used for predictions)
            exp_name = "LightGBM (explanations only)"
            exp_pipe = build(make_model("LightGBM", {}), spec).fit(Xtr, ytr)
        explainer = shap.TreeExplainer(exp_pipe[-1])
        samp = Xte.sample(min(1500, len(Xte)), random_state=RANDOM_STATE)
        Xt = exp_pipe[:-1].transform(samp)
        sv = np.asarray(explainer.shap_values(Xt))
        agg = explain.aggregate_shap(sv, list(Xt.columns), spec.categorical)
        gimp = agg.abs().mean().sort_values(ascending=False)

        # ---- plots
        plot_comparison(tbl.assign(cv_rmse_std=tbl["cv_rmse_std"]), rdir)
        plot_residuals(yte.reset_index(drop=True), pd.Series(pred_te), te["crop"].reset_index(drop=True), best_name, rdir)
        plt.figure(); shap.summary_plot(sv, Xt, max_display=18, show=False); plt.tight_layout(); plt.savefig(rdir / "shap_summary_beeswarm.png", dpi=130, bbox_inches="tight"); plt.close()
        plt.figure(figsize=(8, 6)); gimp.head(15)[::-1].plot.barh(color="#2f855a"); plt.xlabel("mean |SHAP| (t/ha)"); plt.tight_layout(); plt.savefig(rdir / "shap_global_importance.png", dpi=130); plt.close()
        order = np.argsort(pred_te)[:: max(1, len(pred_te) // 150)]
        plt.figure(figsize=(10, 5)); plt.fill_between(range(len(order)), lo_t[order], hi_t[order], alpha=.3, color="#2f855a"); plt.plot(pred_te[order], color="#276749")
        plt.scatter(range(len(order)), yte.values[order], s=12, color="#c05621"); plt.title(f"Test-year prediction intervals (coverage {unc['coverage_calibrated']:.1%})")
        plt.tight_layout(); plt.savefig(rdir / "prediction_intervals.png", dpi=130); plt.close()

        # ---- deployed model: which years it is fitted on (the evaluated model stays the train-only fit)
        def refit(pipe):
            if final_fit == "train":
                return pipe
            idx = np.concatenate([sp["train"], sp["val"]] + ([sp["test"]] if final_fit == "all" else []))
            return clone(pipe).fit(df.iloc[idx][spec.raw], df.iloc[idx][TARGET])
        deploy_pipe = refit(best)
        deploy_exp = deploy_pipe if exp_name == best_name else refit(exp_pipe)
        dist_tab, reg_tab, districts = district_tables(df, spec)

        bench = {}
        for crop, g in df.groupby("crop"):
            top = g[g[TARGET] >= g[TARGET].quantile(.75)]
            bench[crop] = {"median_yield": float(g[TARGET].median()), "p75_yield": float(g[TARGET].quantile(.75)), "p90_yield": float(g[TARGET].quantile(.9)),
                           "top_quartile_median": {c: float(top[c].median()) for c in ("rainfall_mm", "temperature_c") if c in top}, "n": int(len(g))}
        ranges = {c: {"min": float(df[c].quantile(.005)), "max": float(df[c].quantile(.995)), "median": float(df[c].median())} for c in spec.numeric if c != "year"}
        # weather ranges per crop x season: 150 mm is normal for rabi wheat but far outside what monsoon rice has seen
        ranges_by_group = {f"{c}|{se}": {f: {"min": float(g[f].quantile(.01)), "max": float(g[f].quantile(.99)), "p25": float(g[f].quantile(.25)), "p75": float(g[f].quantile(.75))} for f in ("rainfall_mm", "temperature_c", "humidity_pct") if f in g}
                           for (c, se), g in df.groupby(["crop", "season"]) if len(g) >= 30}
        used_optional = [c for c in spec.raw if c not in spec.required]

        comparison = [{**r, "is_baseline": False} for r in tbl.reset_index().to_dict("records")] + base_rows
        meta = {
            "version": version, "name": name, "model_name": name, "data_mode": "real", "trained_at": datetime.now(timezone.utc).isoformat(),
            "data_source": f"REAL district-level crop statistics (APY), {data_meta.get('years', ['?', '?'])[0]}-{data_meta.get('years', ['?', '?'])[-1]}",
            "provenance": {"is_fixture": bool(allow_fixture), "yield": PROVENANCE["yield"], "weather": PROVENANCE["weather"], "soil": PROVENANCE["soil"], "sources": data_meta.get("sources", {}),
                           "dataset_version": data_meta.get("dataset_version"), "statement": "Yields are district averages from government statistics (observed). Weather and soil are gridded "
                           "estimates at the district centroid, not measurements at the user's field. The model predicts a DISTRICT-LEVEL average yield, not a single farm's yield."},
            "best_model": best_name, "selection_criterion": "lowest validation RMSE (chronological hold-out)", "explainer_model": exp_name,
            "split": {"train_years": sp["years"]["train"], "val_years": sp["years"]["val"], "test_years": sp["years"]["test"], "n_train": len(tr), "n_val": len(va), "n_test": len(te)},
            "n_train": len(tr), "n_test": len(te), "cv_folds": 3, "optuna_trials": trials, "final_fit": final_fit,
            "metrics_test": {k.replace("test_", ""): float(v) for k, v in tbl.loc[best_name].items() if k.startswith("test_")},
            "metrics_val": {k.replace("val_", ""): float(v) for k, v in tbl.loc[best_name].items() if k.startswith("val_")},
            "metrics_cv": {k.replace("cv_", ""): float(v) for k, v in tbl.loc[best_name].items() if k.startswith("cv_") and not k.endswith("_std")},
            "comparison": comparison, "synthetic_reference": syn, "best_params": best_params, "per_crop_metrics": per_crop, "per_region_metrics": per_region,
            "groups_below_min_rows": {"crop": small_c, "region": small_r, "min_rows": minr}, "uncertainty": unc,
            "global_importance": {k: float(v) for k, v in gimp.head(20).items()},
            "residual_summary": {"mean": float(resid.mean()), "std": float(resid.std()), "skew": float(stats.skew(resid))},
            "feature_spec": spec.to_dict(), "features": {"categorical": spec.categorical, "numeric": spec.numeric, "engineered": spec.engineered},
            "input_schema": {"required": spec.required, "optional": used_optional + ["district", "area_ha"], "used": spec.raw,
                             "notes": "Soil, latitude and longitude are looked up from the selected district (regional estimates). area_ha only scales total production and profit."},
            "benchmarks": bench, "feature_ranges": ranges, "feature_ranges_by_group": ranges_by_group, "crop_profiles": CROP_PROFILE, "economics": ECONOMICS,
            "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in spec.categorical}, "districts": districts,
            "cleaning": data_meta.get("cleaning_report", {}),
        }
        bundle = {"version": version, "name": name, "model_name": best_name, "data_mode": "real", "pipeline": deploy_pipe, "q_low": q_lo, "q_high": q_hi, "qhat": qhat,
                  "interval_mode": "relative", "explainer_name": exp_name, "explainer_pipeline": deploy_exp, "feature_names": list(Xt.columns),
                  "spec": spec.to_dict(), "categorical": spec.categorical, "raw_features": spec.raw, "district_table": dist_tab, "region_table": reg_tab}
        jl, js = artifact_dir / f"{name}.joblib", artifact_dir / f"{name}.json"
        joblib.dump(bundle, jl, compress=3)
        js.write_text(json.dumps(_clean_json(meta), indent=2))
        tbl.round(5).to_csv(rdir / "model_comparison.csv")
        (rdir / "metrics_summary.json").write_text(json.dumps(_clean_json({"best_model": best_name, **meta["metrics_test"], "uncertainty": unc}), indent=2))
        mlflow.log_metrics({f"best_test_{k}": v for k, v in meta["metrics_test"].items()} | {f"best_val_{k}": v for k, v in meta["metrics_val"].items()})
        mlflow.set_tag("best_model", best_name)
        mlflow.log_artifacts(str(rdir), artifact_path="evaluation")
        mlflow.log_artifact(str(js)); mlflow.log_artifact(str(jl))
        print(f"[saved] {jl.name} ({jl.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s | mlflow run {parent.info.run_id}")
    show = pd.DataFrame(comparison).set_index("model")[["val_rmse", "test_r2", "test_mae", "test_rmse"]]
    print(show.round(4).to_string())
    return {"name": name, "version": version, "bundle": jl, "meta": js, "report_dir": rdir, "best_model": best_name}