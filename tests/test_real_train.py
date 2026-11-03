import json
import shutil

import joblib
import numpy as np
import pytest

from ml.src import train_real as tr
from ml.src.features import DistrictFeatureEngineer, engineered_for, make_district_spec
from ml.src.real_config import TARGET, load_config
from tests.real_fixtures import processed_fixture


@pytest.fixture(scope="module")
def df():
    return processed_fixture()


# ----------------------------------------------------------------------------- chronological protocol
def test_chronological_split_has_no_temporal_leakage(df):
    sp = tr.chronological_split(df.sample(frac=1, random_state=1).reset_index(drop=True), test_years=3, val_years=2)   # shuffled input must not matter
    d = df.sample(frac=1, random_state=1).reset_index(drop=True)
    ytr, yva, yte = d.iloc[sp["train"]].year, d.iloc[sp["val"]].year, d.iloc[sp["test"]].year
    assert ytr.max() < yva.min() and yva.max() < yte.min()
    assert sp["years"]["test"] == [2018, 2019, 2020] and sp["years"]["val"] == [2016, 2017]
    assert len(set(sp["train"]) & set(sp["val"])) == 0 and len(set(sp["val"]) & set(sp["test"])) == 0
    assert len(sp["train"]) + len(sp["val"]) + len(sp["test"]) == len(d)


def test_split_refuses_too_few_years(df):
    with pytest.raises(ValueError, match="too few"):
        tr.chronological_split(df[df.year < 2009], 3, 2)


def test_blocked_cv_folds_only_look_backwards(df):
    folds = tr.blocked_year_splits(df.year.values, 3)
    assert len(folds) == 3
    for a, b in folds:
        assert df.year.values[a].max() < df.year.values[b].min()
    assert folds[0][0].size < folds[-1][0].size                    # expanding window


def test_baselines_are_built_from_training_rows_only(df):
    train, test = df[df.year < 2018], df[df.year >= 2018]
    b = tr.baseline_predictions(train, test)
    manual = train.groupby(["crop", "region", "season"])[TARGET].mean()
    r = test.iloc[0]
    assert b["Baseline: crop/state/season mean"][0] == pytest.approx(manual[(r.crop, r.region, r.season)])
    unseen = test.copy(); unseen["district"] = "never_seen"
    assert np.isfinite(tr.baseline_predictions(train, unseen)["Baseline: district-crop history mean"]).all()   # falls back, never NaN


# ----------------------------------------------------------------------------- feature schema
def test_spec_uses_only_features_the_data_really_has(df):
    full = make_district_spec(df)
    assert {"rainfall_mm", "temperature_c", "soil_ph", "latitude"} <= set(full.numeric) and "ph_deviation" in full.engineered
    no_hum = df.assign(humidity_pct=np.nan, soil_ph=np.nan)
    s = make_district_spec(no_hum)
    assert "humidity_pct" not in s.numeric and "soil_ph" not in s.numeric and "ph_deviation" not in s.engineered
    with pytest.raises(ValueError, match="rainfall_mm"):
        make_district_spec(df.drop(columns="rainfall_mm"))
    for fake_farm_col in ("nitrogen", "phosphorus", "potassium", "fertilizer_kg_ha", "irrigation"):
        assert fake_farm_col not in full.raw                      # farm-level inputs are never invented for the real model


def test_engineer_imputes_clips_and_survives_missing_columns(df):
    spec = make_district_spec(df)
    fe = DistrictFeatureEngineer(spec.categorical, spec.numeric).fit(df[spec.raw])
    x = df[spec.raw].head(5).copy()
    x.loc[x.index[0], "soil_ph"] = np.nan
    x.loc[x.index[1], "rainfall_mm"] = 1e7
    out = fe.transform(x.drop(columns=["humidity_pct"]))          # a column the caller did not supply
    assert not out.isna().any().any() and out["rainfall_mm"].max() < 1e5
    assert list(out.columns) == spec.categorical + spec.numeric + spec.engineered and engineered_for(spec.numeric) == spec.engineered


# ----------------------------------------------------------------------------- end-to-end training on the fixture
def test_end_to_end_real_training_versions_and_never_overwrites(df, tmp_path):
    art = tmp_path / "artifacts"; art.mkdir()
    (art / "model_v1.json").write_text(json.dumps({"best_model": "Stacking Ensemble", "metrics_test": {"r2": 0.938}, "data_source": "synthetic"}))
    (art / "model_v1.joblib").write_bytes(b"synthetic-model-placeholder")
    before = (art / "model_v1.joblib").read_bytes()
    cfg = load_config(); cfg["split"].update(test_years=3, val_years=2, min_rows_per_group_report=20)

    r1 = tr.main(trials=0, models=["Ridge", "LightGBM", "XGBoost"], final_fit="train", df=df, data_meta={"dataset_version": "fixture123", "years": [2005, 2020]},
                 artifact_dir=art, report_dir=tmp_path / "rep", cfg=cfg, mlflow_uri=f"sqlite:///{tmp_path / 'mlf.db'}", allow_fixture=True)
    assert r1["name"] == "real_district_v1" and r1["bundle"].exists() and r1["meta"].exists()
    r2 = tr.main(trials=0, models=["Ridge", "LightGBM"], final_fit="train_val", df=df, artifact_dir=art, report_dir=tmp_path / "rep", cfg=cfg, mlflow_uri=f"sqlite:///{tmp_path / 'mlf.db'}", allow_fixture=True)
    assert r2["name"] == "real_district_v2"                                                         # new version, not an overwrite
    assert (art / "model_v1.joblib").read_bytes() == before                                         # synthetic artifacts untouched

    def strict(path):  # NaN is not valid JSON and would crash the API
        return json.loads(path.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    m = strict(r1["meta"])
    assert m["data_mode"] == "real" and m["provenance"]["yield"] == "observed" and m["provenance"]["weather"] == "regional_estimate"
    assert m["split"]["train_years"][-1] < m["split"]["val_years"][0] < m["split"]["test_years"][0]
    assert m["selection_criterion"].startswith("lowest validation")
    labels = [c["model"] for c in m["comparison"]]
    assert "Stacking Ensemble" in labels and any(x.startswith("Baseline") for x in labels)
    assert m["synthetic_reference"]["metrics_test"]["r2"] == 0.938 and "NOT comparable" in m["synthetic_reference"]["note"]
    assert set(m["per_crop_metrics"]) == {"Rice", "Wheat", "Maize"} and m["per_region_metrics"]
    for c in m["per_crop_metrics"].values():
        assert {"r2", "mae", "rmse"} <= set(c)
    assert m["input_schema"]["required"] == ["crop", "region", "season", "rainfall_mm", "temperature_c"] and "nitrogen" not in m["input_schema"]["used"]
    # weather/soil carry real signal in the fixture, so a sane pipeline must beat the no-weather baseline on the held-out years
    comp = {c["model"]: c for c in m["comparison"]}
    assert m["metrics_test"]["rmse"] < comp["Baseline: crop/state/season mean"]["test_rmse"]
    assert 0.6 <= m["uncertainty"]["coverage_calibrated"] <= 0.97
    # a bundle holds what serving needs
    b = joblib.load(r1["bundle"])
    assert b["data_mode"] == "real" and b["categorical"] == ["crop", "region", "season"] and "Karnataka|district0" in b["district_table"]
    pred = b["pipeline"].predict(df[b["raw_features"]].head(3))
    assert np.isfinite(pred).all() and (pred > 0).all()
    shutil.rmtree(art)


def test_fixture_data_can_never_become_a_real_model(df, tmp_path):
    """Lineage guard: rows not marked yield_source == 'apy_observed' (e.g. test fixtures, synthetic data) are refused."""
    with pytest.raises(ValueError, match="Refusing to train a 'real_district' model"):
        tr.main(trials=0, models=["Ridge"], df=df, artifact_dir=tmp_path, report_dir=tmp_path / "r", mlflow_uri=f"sqlite:///{tmp_path / 'm.db'}")
    assert not list(tmp_path.glob("real_district_v*"))
    ok = df.assign(yield_source="apy_observed")
    cfg = load_config(); cfg["split"].update(test_years=2, val_years=2, min_rows_per_group_report=20)
    r = tr.main(trials=0, models=["Ridge"], df=ok, artifact_dir=tmp_path, report_dir=tmp_path / "r", cfg=cfg, mlflow_uri=f"sqlite:///{tmp_path / 'm.db'}")
    assert json.loads(r["meta"].read_text())["provenance"]["is_fixture"] is False