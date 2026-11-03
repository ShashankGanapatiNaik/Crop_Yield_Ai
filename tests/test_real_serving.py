"""Serving a REAL-schema model next to the synthetic one. The model here is trained on a labelled FIXTURE (see real_fixtures.py)."""
import pytest

from backend.app.main import app
from backend.app.services.model_service import InputError, ModelService, get_model_service
from ml.src import train_real as tr
from ml.src.real_config import load_config
from tests.real_fixtures import processed_fixture


@pytest.fixture(scope="module")
def real_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("real_models")
    cfg = load_config(); cfg["split"].update(test_years=2, val_years=2, min_rows_per_group_report=20)
    tr.main(trials=0, models=["Ridge", "LightGBM"], final_fit="train", df=processed_fixture(), data_meta={"dataset_version": "fx", "years": [2005, 2020]},
            artifact_dir=d, report_dir=d / "rep", cfg=cfg, mlflow_uri=f"sqlite:///{d / 'm.db'}", allow_fixture=True)
    return d


@pytest.fixture(scope="module")
def svc(real_dir):
    return ModelService(model_dir=str(real_dir), version="real_district_v1")


@pytest.fixture
def use_real(svc):
    from backend.app.api.predict import get_optional_service
    app.dependency_overrides[get_model_service] = lambda: svc
    app.dependency_overrides[get_optional_service] = lambda: svc
    yield
    app.dependency_overrides.clear()


BODY = dict(crop="Rice", region="Karnataka", season="Kharif", rainfall_mm=950, temperature_c=27, district="district0", area_ha=2.0, year=2024)


# ----------------------------------------------------------------------------- model selection
def test_version_selector_keeps_synthetic_default_and_reaches_real(real_dir, tmp_path):
    import shutil
    for f in real_dir.glob("real_district_v1.*"):
        shutil.copy(f, tmp_path / f.name.replace("v1", "v2"))
        shutil.copy(f, tmp_path / f.name)
    s = ModelService.__new__(ModelService); s.dir = tmp_path
    assert s._resolve_stem("real_latest") == "real_district_v2" and s._resolve_stem("real_district_v1") == "real_district_v1"
    with pytest.raises(FileNotFoundError):
        s._resolve_stem("latest")                      # only real models here: 'latest' means the synthetic model_vN series
    with pytest.raises(ValueError):
        s._resolve_stem("banana")


def test_active_model_describes_its_input_schema(svc):
    d = svc.describe()
    assert d["data_mode"] == "real" and d["model_name"] == "real_district_v1"
    assert d["input_schema"]["required"] == ["crop", "region", "season", "rainfall_mm", "temperature_c"]
    assert "nitrogen" not in d["input_schema"]["used"] and "irrigation" not in d["input_schema"]["used"] and "district0" in d["districts"]["Karnataka"]
    assert d["provenance"]["yield"] == "observed" and d["provenance"]["soil"] == "regional_estimate"


# ----------------------------------------------------------------------------- input rules
def test_missing_required_and_unknown_values_are_rejected(svc):
    with pytest.raises(InputError) as e:
        svc.prepare({"crop": "Rice", "region": "Karnataka", "season": "Kharif"})
    assert {x["field"] for x in e.value.errors} == {"rainfall_mm", "temperature_c"}
    with pytest.raises(InputError, match="unknown region"):
        svc.prepare({**BODY, "region": "Atlantis"})
    with pytest.raises(InputError, match="not found for Karnataka"):
        svc.prepare({**BODY, "district": "nowhere"})
    with pytest.raises(InputError):
        svc.canonical_crops(["Rice", "Banana"])


def test_district_values_are_filled_and_flagged_as_estimates(svc):
    row, notes = svc.prepare({**BODY, "crop": "rice", "season": "KHARIF", "region": "karnataka"})
    assert (row["crop"], row["season"], row["region"]) == ("Rice", "Kharif", "Karnataka")          # case-insensitive, canonical output
    assert {"soil_ph", "latitude", "longitude"} <= set(notes["estimated_inputs"]) and row["soil_ph"] > 0
    assert any("regional estimate" in n for n in notes["data_notes"]) and any("district-level average" in n for n in notes["data_notes"])
    user_ph, n2 = svc.prepare({**BODY, "soil_ph": 5.1})
    assert user_ph["soil_ph"] == 5.1 and "soil_ph" not in n2["estimated_inputs"]                    # a value the user typed is never overwritten
    nodist, n3 = svc.prepare({k: v for k, v in BODY.items() if k != "district"})
    assert "soil_ph" in nodist and any("state median" in n for n in n3["data_notes"])               # falls back to the state median, and says so
    near, n4 = svc.prepare({**{k: v for k, v in BODY.items() if k != "district"}, "latitude": 25.0, "longitude": 75.0})
    assert n4["district"] is not None and any("nearest district" in n for n in n4["data_notes"])
    again, n5 = svc.prepare(row)                                                                     # idempotent
    assert again["soil_ph"] == row["soil_ph"]


def test_area_defaults_to_one_hectare_with_a_note(svc):
    row, notes = svc.prepare({k: v for k, v in BODY.items() if k != "area_ha"})
    assert row["area_ha"] == 1.0 and any("per 1 ha" in n for n in notes["data_notes"])


# ----------------------------------------------------------------------------- predictions
def test_full_prediction_payload_for_district_model(svc):
    out = svc.full_prediction({k: v for k, v in BODY.items() if k != "crop"}, "Rice")
    assert out["data_mode"] == "real" and out["model_name"] == "real_district_v1" and out["estimated_inputs"]
    assert out["lower_t_ha"] <= out["predicted_yield_t_ha"] <= out["upper_t_ha"] and out["total_production_t"] == pytest.approx(out["predicted_yield_t_ha"] * 2.0, rel=0.01)
    assert len(out["top_factors"]) == 5 and all(f["feature"] not in ("irrigation", "fertilizer_kg_ha", "nitrogen") for f in out["top_factors"])
    cats = {r["category"] for r in out["recommendations"]}
    assert not cats & {"nitrogen", "phosphorus", "potassium"}                                          # no NPK advice without farm-level soil tests
    assert all(r["expected_gain_t_ha"] is None for r in out["recommendations"])                         # no model-simulated levers for this schema
    assert svc.levers({}, "Rice", 3.0) == {}


def test_weather_matters_to_the_prediction_in_the_expected_direction(svc):
    base = {k: v for k, v in BODY.items() if k != "crop"}
    good = svc.full_prediction({**base, "rainfall_mm": 900}, "Rice")["predicted_yield_t_ha"]
    dry = svc.full_prediction({**base, "rainfall_mm": 450}, "Rice")["predicted_yield_t_ha"]           # inside the training range (well supported by data)
    assert dry < good                                                                                  # the fixture encodes this relation; the pipeline must learn it


def test_out_of_range_inputs_get_an_honest_warning(svc):
    base = {k: v for k, v in BODY.items() if k != "crop"}
    out = svc.full_prediction({**base, "rainfall_mm": 150}, "Rice")
    assert any("Rainfall 150 is outside the range the model was trained on" in n for n in out["data_notes"])
    assert not any("outside the range" in n for n in svc.full_prediction({**base, "rainfall_mm": 900}, "Rice")["data_notes"])
    # district-estimated values are never flagged as the user's out-of-range input
    assert not any("Soil pH" in n and "outside" in n for n in out["data_notes"])


def test_compare_and_what_if_work_with_district_model(svc):
    base = {k: v for k, v in BODY.items() if k != "crop"}
    c = svc.compare(base, None)
    assert {r["crop"] for r in c["results"]} == {"Rice", "Wheat", "Maize"}                             # only crops this model knows
    w = svc.what_if({**base, "rainfall_mm": 900}, "Rice", {"rainfall_mm": 450}, {"param": "temperature_c", "min": 22, "max": 30, "steps": 5})
    assert w["delta_yield_t_ha"] < 0 and len(w["sweep"]["points"]) == 5 and w["scenario"]["inputs"]["rainfall_mm"] == 450


# ----------------------------------------------------------------------------- HTTP contract
def test_api_accepts_the_reduced_input_set(client, farmer, use_real):
    r = client.post("/predict", json=BODY, headers=farmer)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["data_mode"] == "real" and d["model_name"] == "real_district_v1" and "soil_ph" in d["estimated_inputs"] and d["data_notes"]
    h = client.get(f"/history/{d['id']}", headers=farmer).json()
    assert h["data_mode"] == "real"                                                                    # provenance is stored with the prediction


def test_api_errors_keep_the_422_contract(client, farmer, use_real):
    r = client.post("/predict", json={k: v for k, v in BODY.items() if k != "rainfall_mm"}, headers=farmer)
    assert r.status_code == 422 and r.json()["detail"] == "Validation failed" and r.json()["errors"][0]["field"] == "rainfall_mm"
    assert client.post("/predict", json={**BODY, "region": "Atlantis"}, headers=farmer).status_code == 422
    assert client.post("/predict", json={**BODY, "rainfall_mm": -5}, headers=farmer).status_code == 422
    assert client.post("/compare-crops", json={**{k: v for k, v in BODY.items() if k != "crop"}, "crops": ["Banana"]}, headers=farmer).status_code == 422


def test_reference_endpoints_follow_the_active_model(client, use_real):
    c = client.get("/crops").json()
    assert c["data_mode"] == "real" and "district0" in c["districts"]["Karnataka"] and c["soil_types"] == [] and c["irrigation"] == []
    assert set(c["input_schema"]["used"]) >= {"crop", "region", "rainfall_mm"} and {x["name"] for x in c["crops"]} == {"Rice", "Wheat", "Maize"}
    m = client.get("/model-info").json()
    assert m["data_mode"] == "real" and m["provenance"]["weather"] == "regional_estimate" and m["split"]["test_years"] and m["per_region_metrics"] is not None


def test_synthetic_contract_is_unchanged_without_override(client):
    c = client.get("/crops").json()
    assert c["data_mode"] == "synthetic" and len(c["crops"]) == 10 and c["soil_types"] and c["input_schema"]["required"]
    assert client.get("/model-info").json()["version"] == 1


def test_real_model_judges_rainfall_against_real_training_data_not_simulator_constants(svc):
    base = {k: v for k, v in BODY.items() if k != "crop"}
    ref = svc.meta["feature_ranges_by_group"]["Rice|Kharif"]["rainfall_mm"]
    water = lambda rain: next(r for r in svc.full_prediction({**base, "rainfall_mm": rain}, "Rice")["recommendations"] if r["category"] == "irrigation")  # noqa: E731
    assert water(ref["p25"] * 0.4)["severity"] == "info" and "typical range" in water(ref["p25"] * 0.4)["title"]
    assert water((ref["p25"] + ref["p75"]) / 2)["severity"] == "success"
    assert water(ref["p75"] * 2)["severity"] == "warning"
    assert "1100" not in water(ref["p25"] * 0.4)["detail"]                 # the simulator's Rice optimum must not leak into real-data advice
    assert not any(r["category"] == "climate" for r in svc.full_prediction({**base, "temperature_c": 44}, "Rice")["recommendations"])