"""Crop-health API with a SCRIPTED fake satellite provider: no network, and nothing is invented. Values come only from the test script."""
import math
from datetime import UTC, datetime, timedelta

import pytest

from backend.app.api.crop_health import get_satellite_provider
from backend.app.core.config import get_settings
from backend.app.db import SessionLocal
from backend.app.main import app
from backend.app.models.orm import NdviObservation
from backend.app.services.satellite import SatelliteError, SatelliteNotConfigured, SatelliteTimeout

SQUARE = {"type": "Polygon", "coordinates": [[[74.44, 14.28], [74.441, 14.28], [74.441, 14.281], [74.44, 14.281], [74.44, 14.28]]]}
BANNED = ("disease", "blight", "pest", "infect", "fungus", "virus", "diagnos")


def day(n_ago):
    return (datetime.now(UTC).date() - timedelta(days=n_ago)).isoformat()


def scene(n_ago, ndvi, vf=0.95, cloud=5.0, tile="43PCR"):
    return {"date": day(n_ago), "ndvi_mean": ndvi, "valid_fraction": vf, "cloud_pct": cloud, "tile": tile}


class FakeProvider:
    def __init__(self):
        self.rows, self.scene_calls, self.layer_calls = [], [], []
        self.scene_error = self.layer_error = None

    def scene_stats(self, geometry, start, end):
        self.scene_calls.append((geometry, start, end))
        if self.scene_error:
            raise self.scene_error
        return list(self.rows)

    def map_layer(self, geometry, obs_date, th, style):
        self.layer_calls.append((obs_date, th, style))
        if self.layer_error:
            raise self.layer_error
        return {"tile_url": "https://tiles.example/{z}/{x}/{y}", "style": style, "date": obs_date.isoformat(), "attribution": "Contains modified Copernicus Sentinel data",
                "expires_note": "temporary"}


@pytest.fixture
def fake():
    p = FakeProvider()
    app.dependency_overrides[get_satellite_provider] = lambda: p
    yield p
    app.dependency_overrides.pop(get_satellite_provider, None)


_n = 0
_created: list[tuple[dict, int]] = []


def make_field(client, headers, **body):
    global _n
    _n += 1
    r = client.post("/crop-health/fields", json={"name": f"Field {_n}", "geometry": SQUARE, **body}, headers=headers)
    assert r.status_code == 201, r.text
    _created.append((headers, r.json()["id"]))
    return r.json()


@pytest.fixture(autouse=True)
def _delete_fields_created_by_the_test(client):
    """The `farmer` fixture is shared by the whole session, so each test removes its own fields (and the per-user limit stays out of the way)."""
    yield
    while _created:
        headers, fid = _created.pop()
        client.delete(f"/crop-health/fields/{fid}", headers=headers)


# ----------------------------------------------------------------------------- status and authentication
def test_status_is_public_and_reports_configuration_without_secrets(client):
    r = client.get("/crop-health/status")
    d = r.json()
    assert r.status_code == 200 and d["configured"] is False and d["dataset"] == "COPERNICUS/S2_SR_HARMONIZED"
    assert d["thresholds"] == {"low": 0.2, "high": 0.5} and [x["color"] for x in d["legend"]] == ["#dc2626", "#eab308", "#16a34a"]
    assert "key" not in str(d).lower() and d["limits"]["max_field_ha"] == 1000


def test_all_field_endpoints_need_a_login(client):
    assert client.post("/crop-health/fields", json={"name": "x", "geometry": SQUARE}).status_code == 401
    for path in ("/crop-health/fields", "/crop-health/fields/1", "/crop-health/fields/1/latest", "/crop-health/fields/1/history"):
        assert client.get(path).status_code == 401
    assert client.delete("/crop-health/fields/1").status_code == 401


# ----------------------------------------------------------------------------- creating and retrieving fields
def test_create_field_from_polygon(client, farmer):
    f = make_field(client, farmer)
    assert f["source"] == "polygon" and f["area_ha"] == pytest.approx(1.2, rel=0.02) and f["radius_m"] is None
    assert f["centroid_lat"] == pytest.approx(14.2805, abs=1e-3) and f["geometry"]["coordinates"][0][0] == f["geometry"]["coordinates"][0][-1]
    assert client.get(f"/crop-health/fields/{f['id']}", headers=farmer).json()["name"] == f["name"]


def test_create_field_from_point_and_radius(client, farmer):
    r = client.post("/crop-health/fields", json={"name": "Round", "latitude": 14.28, "longitude": 74.44, "radius_m": 150}, headers=farmer)
    f = r.json()
    _created.append((farmer, f["id"]))
    assert r.status_code == 201 and f["source"] == "point_radius" and f["radius_m"] == 150 and f["area_ha"] == pytest.approx(math.pi * 150 ** 2 / 10_000, rel=0.01)


@pytest.mark.parametrize("body,field", [
    ({"latitude": 91, "longitude": 74}, "latitude"),
    ({"latitude": 14, "longitude": 181}, "longitude"),
    ({"latitude": 14, "longitude": 74, "radius_m": 3}, "radius_m"),
    ({"latitude": 14, "longitude": 74, "radius_m": 99999}, "radius_m"),
    ({"geometry": {"type": "Polygon", "coordinates": [[[74.44, 14.28], [74.44001, 14.28], [74.44001, 14.28001], [74.44, 14.28]]]}}, "geometry"),   # tiny
    ({"geometry": {"type": "Polygon", "coordinates": [[[74.44, 14.28], [74.441, 14.281], [74.441, 14.28], [74.44, 14.281], [74.44, 14.28]]]}}, "geometry"),   # bowtie
    ({"geometry": {"type": "Point", "coordinates": [74.44, 14.28]}}, "geometry"),
    ({"geometry": {**SQUARE, "coordinates": SQUARE["coordinates"] + SQUARE["coordinates"]}}, "geometry"),                                              # hole
])
def test_invalid_fields_are_rejected_with_a_clear_422(client, farmer, body, field):
    r = client.post("/crop-health/fields", json={"name": "bad", **body}, headers=farmer)
    assert r.status_code == 422 and r.json()["detail"] == "Validation failed" and r.json()["errors"][0]["field"] == field, r.text


@pytest.mark.parametrize("body", [{"name": "x"}, {"name": "x", "latitude": 14}, {"name": "x", "geometry": SQUARE, "latitude": 14, "longitude": 74}, {"name": "", "geometry": SQUARE}])
def test_exactly_one_way_of_defining_a_field_is_required(client, farmer, body):
    assert client.post("/crop-health/fields", json=body, headers=farmer).status_code == 422


def test_field_limit_per_user(client, farmer2, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_fields_per_user", 2)
    for _ in range(2):
        make_field(client, farmer2)
    r = client.post("/crop-health/fields", json={"name": "third", "geometry": SQUARE}, headers=farmer2)
    assert r.status_code == 422 and "limit" in r.json()["errors"][0]["message"]


def test_fields_are_private_to_their_owner(client, farmer, farmer2, fake):
    f = make_field(client, farmer)
    fid = f["id"]
    assert fid not in [x["id"] for x in client.get("/crop-health/fields", headers=farmer2).json()]
    for path in (f"/crop-health/fields/{fid}", f"/crop-health/fields/{fid}/latest", f"/crop-health/fields/{fid}/history"):
        assert client.get(path, headers=farmer2).status_code == 404
    assert client.delete(f"/crop-health/fields/{fid}", headers=farmer2).status_code == 404
    assert fake.scene_calls == []                                                         # a stranger never triggers satellite work


# ----------------------------------------------------------------------------- latest result
def test_latest_returns_ndvi_date_category_layer_and_cautious_explanation(client, farmer, fake):
    fake.rows = [scene(20, 0.45), scene(10, 0.72), scene(7, 0.9, vf=0.1), scene(3, None, vf=0.0, cloud=95)]     # 0.9 is mostly cloud; the last scene is fully clouded
    f = make_field(client, farmer)
    r = client.get(f"/crop-health/fields/{f['id']}/latest", headers=farmer)
    d = r.json()
    assert r.status_code == 200 and d["status"] == "ok"
    assert d["observation"]["date"] == day(10) and d["observation"]["ndvi_mean"] == 0.72 and d["observation"]["category"] == "high" and d["observation"]["color"] == "#16a34a"
    assert d["previous"]["date"] == day(20) and d["previous"]["category"] == "moderate"
    assert d["layer"]["tile_url"].endswith("{z}/{x}/{y}") and d["layer"]["date"] == day(10) and d["layer"]["bounds"] == [[14.28, 74.44], [14.281, 74.441]]
    assert "rose by 0.27" in d["interpretation"]["summary"] and not any(w in d["interpretation"]["summary"].lower() for w in BANNED)
    assert any("not a diagnosis" in c for c in d["interpretation"]["caveats"]) and "Copernicus" in d["source"]["attribution"]
    assert d["thresholds"]["low"] == 0.2 and d["window"]["days"] == 90 and d["from_cache"] is False and d["stale"] is False
    geom, start, end = fake.scene_calls[0]
    assert geom == f["geometry"] and (end - start).days == 90                             # the saved boundary and the default window were sent to the provider
    assert fake.layer_calls[0][0].isoformat() == day(10)


@pytest.mark.parametrize("ndvi,cat,color", [(0.05, "low", "#dc2626"), (0.19, "low", "#dc2626"), (0.2, "moderate", "#eab308"), (0.49, "moderate", "#eab308"), (0.5, "high", "#16a34a")])
def test_colour_classes_follow_the_thresholds(client, farmer, fake, ndvi, cat, color):
    fake.rows = [scene(5, ndvi)]
    f = make_field(client, farmer)
    o = client.get(f"/crop-health/fields/{f['id']}/latest", headers=farmer).json()["observation"]
    assert (o["category"], o["color"]) == (cat, color)


def test_no_usable_imagery_returns_no_data_and_invents_nothing(client, farmer, fake):
    fake.rows = [scene(4, 0.6, vf=0.1), scene(9, None, vf=0.0, cloud=99), scene(12, 0.7, vf=0.3)]       # cloud-covered / mostly masked scenes
    f = make_field(client, farmer)
    r = client.get(f"/crop-health/fields/{f['id']}/latest", headers=farmer)
    d = r.json()
    assert r.status_code == 200 and d["status"] == "no_data" and d["observation"] is None and d["layer"] is None and d["interpretation"] is None
    assert "No cloud-free Sentinel-2 observation" in d["reason"] and "No values are estimated" in d["reason"]
    assert fake.layer_calls == []                                                          # no map is requested for a date that does not exist
    assert client.get(f"/crop-health/fields/{f['id']}/history", headers=farmer).json()["observations"] == []


def test_empty_scene_list_is_also_no_data(client, farmer, fake):
    fake.rows = []
    f = make_field(client, farmer)
    assert client.get(f"/crop-health/fields/{f['id']}/latest", headers=farmer).json()["status"] == "no_data"


@pytest.mark.parametrize("err,code", [(SatelliteError("Could not reach Earth Engine."), 502), (SatelliteNotConfigured("not configured"), 503), (SatelliteTimeout("too long"), 504)])
def test_provider_failures_become_clean_http_errors(client, farmer, fake, err, code):
    fake.scene_error = err
    f = make_field(client, farmer)
    r = client.get(f"/crop-health/fields/{f['id']}/latest", headers=farmer)
    assert r.status_code == code and r.json()["detail"] == str(err)
    assert client.get(f"/crop-health/fields/{f['id']}/history", headers=farmer).status_code == code


def test_real_provider_reports_not_configured_when_gee_project_is_missing(client, farmer):
    """No dependency override: the real provider must refuse politely (503) instead of crashing or fabricating data."""
    f = make_field(client, farmer)
    r = client.get(f"/crop-health/fields/{f['id']}/latest", headers=farmer)
    assert r.status_code == 503 and "GEE_PROJECT" in r.json()["detail"]


def test_a_failed_map_layer_does_not_hide_the_ndvi_value(client, farmer, fake):
    fake.rows = [scene(5, 0.6)]
    fake.layer_error = SatelliteError("tiles unavailable")
    f = make_field(client, farmer)
    d = client.get(f"/crop-health/fields/{f['id']}/latest", headers=farmer).json()
    assert d["status"] == "ok" and d["observation"]["ndvi_mean"] == 0.6 and d["layer"] is None and d["layer_error"] == "tiles unavailable"


# ----------------------------------------------------------------------------- caching
def test_results_are_cached_and_refresh_forces_a_new_request(client, farmer, fake):
    fake.rows = [scene(5, 0.6)]
    f = make_field(client, farmer)
    url = f"/crop-health/fields/{f['id']}"
    assert client.get(f"{url}/latest", headers=farmer).json()["from_cache"] is False
    again = client.get(f"{url}/latest", headers=farmer).json()
    client.get(f"{url}/history", headers=farmer)
    assert again["from_cache"] is True and len(fake.scene_calls) == 1                      # latest + history share one satellite request
    assert client.get(f"{url}/latest?refresh=true", headers=farmer).json()["from_cache"] is False and len(fake.scene_calls) == 2
    client.get(f"{url}/history?days=365", headers=farmer)                                   # a longer window than cached needs a new request
    assert len(fake.scene_calls) == 3
    client.get(f"{url}/history?days=200", headers=farmer)                                   # ... but a shorter one is served from the 365-day cache
    assert len(fake.scene_calls) == 3


def test_no_data_results_are_cached_too(client, farmer, fake):
    fake.rows = []
    f = make_field(client, farmer)
    for _ in range(3):
        client.get(f"/crop-health/fields/{f['id']}/latest", headers=farmer)
    assert len(fake.scene_calls) == 1


def test_stale_cache_is_served_with_a_warning_when_the_satellite_service_fails(client, farmer, fake):
    fake.rows = [scene(5, 0.6)]
    f = make_field(client, farmer)
    url = f"/crop-health/fields/{f['id']}/latest"
    client.get(url, headers=farmer)
    fake.scene_error = SatelliteError("Earth Engine usage limit reached. Please try again later.")
    r = client.get(f"{url}?refresh=true", headers=farmer)
    d = r.json()
    assert r.status_code == 200 and d["stale"] is True and "usage limit" in d["warning"] and d["observation"]["ndvi_mean"] == 0.6


def test_refresh_replaces_old_observations_of_the_window(client, farmer, fake):
    f = make_field(client, farmer)
    fake.rows = [scene(10, 0.3), scene(5, 0.4)]
    client.get(f"/crop-health/fields/{f['id']}/latest", headers=farmer)
    fake.rows = [scene(5, 0.8)]
    client.get(f"/crop-health/fields/{f['id']}/latest?refresh=true", headers=farmer)
    h = client.get(f"/crop-health/fields/{f['id']}/history", headers=farmer).json()
    assert [(o["date"], o["ndvi_mean"]) for o in h["observations"]] == [(day(5), 0.8)]      # the withdrawn 10-day-old scene is gone


# ----------------------------------------------------------------------------- history
def test_history_is_chronological_with_categories(client, farmer, fake):
    fake.rows = [scene(30, 0.65), scene(60, 0.15), scene(45, 0.35)]
    f = make_field(client, farmer)
    h = client.get(f"/crop-health/fields/{f['id']}/history?days=120", headers=farmer).json()
    assert h["status"] == "ok" and h["count"] == 3 and [o["date"] for o in h["observations"]] == [day(60), day(45), day(30)]
    assert [o["category"] for o in h["observations"]] == ["low", "moderate", "high"] and h["window"]["days"] == 120


@pytest.mark.parametrize("q", ["days=3", "days=5000", "days=abc"])
def test_window_length_is_validated(client, farmer, fake, q):
    f = make_field(client, farmer)
    assert client.get(f"/crop-health/fields/{f['id']}/history?{q}", headers=farmer).status_code == 422
    assert fake.scene_calls == []


def test_layer_style_is_validated_and_passed_through(client, farmer, fake):
    fake.rows = [scene(5, 0.6)]
    f = make_field(client, farmer)
    assert client.get(f"/crop-health/fields/{f['id']}/latest?style=rainbow", headers=farmer).status_code == 422
    assert client.get(f"/crop-health/fields/{f['id']}/latest?style=gradient", headers=farmer).json()["layer"]["style"] == "gradient"


# ----------------------------------------------------------------------------- deletion
def test_deleting_a_field_removes_its_cached_observations(client, farmer, fake):
    fake.rows = [scene(5, 0.6), scene(15, 0.5)]
    f = make_field(client, farmer)
    client.get(f"/crop-health/fields/{f['id']}/latest", headers=farmer)
    with SessionLocal() as db:
        assert db.query(NdviObservation).filter_by(field_id=f["id"]).count() == 2
    assert client.delete(f"/crop-health/fields/{f['id']}", headers=farmer).status_code == 204
    with SessionLocal() as db:
        assert db.query(NdviObservation).filter_by(field_id=f["id"]).count() == 0
    assert client.get(f"/crop-health/fields/{f['id']}", headers=farmer).status_code == 404