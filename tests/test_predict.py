import pytest

import backend.app.services.weather as weather_mod


def test_predict_returns_full_payload(client, farmer, payload):
    r = client.post("/predict", json=payload, headers=farmer)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["predicted_yield_t_ha"] > 0
    assert d["lower_t_ha"] <= d["predicted_yield_t_ha"] <= d["upper_t_ha"]
    assert d["total_production_t"] == pytest.approx(d["predicted_yield_t_ha"] * payload["area_ha"], rel=0.01)
    assert len(d["top_factors"]) == 5
    assert {"feature", "label", "value", "shap", "direction"} <= set(d["top_factors"][0])
    assert abs(d["top_factors"][0]["shap"]) >= abs(d["top_factors"][-1]["shap"])
    assert d["recommendations"] and d["economics"]["revenue"] > 0 and d["id"]


def test_prediction_reflects_agronomy(client, farmer, payload):
    """Sanity: a drought-stressed, rainfed field must yield less than a well-watered one."""
    wet = client.post("/predict", json=payload, headers=farmer).json()["predicted_yield_t_ha"]
    dry = client.post("/predict", json={**payload, "rainfall_mm": 150, "irrigation": "Rainfed"}, headers=farmer).json()["predicted_yield_t_ha"]
    assert dry < wet * 0.8


@pytest.mark.parametrize("patch", [{"ph": 99}, {"ph": -1}, {"area_ha": 0}, {"humidity_pct": 150}, {"crop": "Banana"}, {"season": "Winter"}, {"rainfall_mm": "abc"}])
def test_validation_errors(client, farmer, payload, patch):
    r = client.post("/predict", json={**payload, **patch}, headers=farmer)
    assert r.status_code == 422
    assert r.json()["detail"] == "Validation failed" and r.json()["errors"]


def test_missing_field_is_422(client, farmer, payload):
    bad = {k: v for k, v in payload.items() if k != "nitrogen"}
    assert client.post("/predict", json=bad, headers=farmer).status_code == 422


def test_category_is_case_insensitive(client, farmer, payload):
    assert client.post("/predict", json={**payload, "crop": "rice", "season": "KHARIF"}, headers=farmer).status_code == 200


def test_history_report_and_ownership(client, farmer, farmer2, payload):
    pid = client.post("/predict", json=payload, headers=farmer).json()["id"]
    hist = client.get("/history", headers=farmer).json()
    assert hist["total"] >= 1 and any(i["id"] == pid for i in hist["items"])
    assert client.get(f"/history/{pid}", headers=farmer).status_code == 200
    pdf = client.get(f"/report/{pid}", headers=farmer)
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf" and pdf.content.startswith(b"%PDF")
    # another user can neither read, report on, nor delete it
    assert client.get(f"/history/{pid}", headers=farmer2).status_code == 404
    assert client.get(f"/report/{pid}", headers=farmer2).status_code == 404
    assert client.delete(f"/history/{pid}", headers=farmer2).status_code == 404
    assert client.delete(f"/history/{pid}", headers=farmer).status_code == 204
    assert client.get(f"/history/{pid}", headers=farmer).status_code == 404


def test_compare_crops_sorted_by_profit(client, farmer, payload):
    body = {k: v for k, v in payload.items() if k != "crop"}
    d = client.post("/compare-crops", json={**body, "crops": ["Rice", "Maize", "Cotton"]}, headers=farmer).json()
    assert {r["crop"] for r in d["results"]} == {"Rice", "Maize", "Cotton"}
    suitable = [r["profit"] for r in d["results"] if r["season_suitable"]]
    assert suitable == sorted(suitable, reverse=True)
    allc = client.post("/compare-crops", json=body, headers=farmer).json()["results"]
    assert len(allc) == 10


def test_what_if_delta_and_sweep(client, farmer, payload):
    body = {"base": payload, "changes": {"rainfall_mm": 200}, "sweep": {"param": "temperature_c", "min": 15, "max": 40, "steps": 6}}
    d = client.post("/what-if", json=body, headers=farmer).json()
    assert d["delta_yield_t_ha"] < 0 and d["scenario"]["inputs"]["rainfall_mm"] == 200
    assert len(d["sweep"]["points"]) == 6
    assert client.post("/what-if", json={"base": payload, "changes": {"colour": 1}}, headers=farmer).status_code == 422
    assert client.post("/what-if", json={"base": payload, "changes": {"ph": 50}}, headers=farmer).status_code == 422


def test_meta_endpoints(client):
    assert len(client.get("/crops").json()["crops"]) == 10
    info = client.get("/model-info").json()
    assert info["version"] == 1 and "metrics_test" in info
    assert client.get("/health").json()["status"] == "ok"


def test_weather_endpoint_mocked(client, monkeypatch):
    monkeypatch.setattr(weather_mod, "fetch_weather", lambda lat, lon: {"temperature_c": 26.1, "humidity_pct": 70, "rainfall_mm": 410.0})
    r = client.get("/weather", params={"lat": 14.28, "lon": 74.44})
    assert r.status_code == 200 and r.json()["rainfall_mm"] == 410.0
    assert client.get("/weather", params={"lat": 95, "lon": 10}).status_code == 422

    def boom(lat, lon):
        raise weather_mod.WeatherError("down")
    monkeypatch.setattr(weather_mod, "fetch_weather", boom)
    assert client.get("/weather", params={"lat": 10, "lon": 10}).status_code == 502


def test_weather_summarise_parses_open_meteo():
    days = 97
    data = {"daily": {"time": list(range(days)), "precipitation_sum": [2.0] * days, "temperature_2m_mean": [25.0] * days, "relative_humidity_2m_mean": [60.0] * days},
            "current": {"temperature_2m": 30, "relative_humidity_2m": 50}}
    out = weather_mod.summarise(data, 14.0, 74.0)
    assert out["period_days"] == 90 and out["rainfall_mm"] == 180.0 and out["next_7d_rainfall_mm"] == 14.0
    assert out["temperature_c"] == 25.0 and out["current_temperature_c"] == 30
