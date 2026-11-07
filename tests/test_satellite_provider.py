"""EarthEngineProvider against a FAKE `ee` module (Earth Engine cannot run offline or without live credentials).
These tests verify OUR logic: credential selection, error mapping, band/mask choices, secret hygiene. They cannot verify that Google's
servers accept the expression: that needs a live smoke test (see docs/crop-health.md)."""
import json
from datetime import date
from unittest.mock import MagicMock, call

import pytest

from backend.app.core.config import Settings
from backend.app.services import earth_engine as ge
from backend.app.services.satellite import SatelliteError, SatelliteNotConfigured, SatelliteTimeout, Thresholds

SECRET_KEY_JSON = json.dumps({"type": "service_account", "client_email": "svc@proj.iam.gserviceaccount.com", "private_key": "-----BEGIN PRIVATE KEY-----TOPSECRET-----END PRIVATE KEY-----"})
GEOM = {"type": "Polygon", "coordinates": [[[74.44, 14.28], [74.441, 14.28], [74.441, 14.281], [74.44, 14.28]]]}


def fake_ee():
    ee = MagicMock(name="ee")
    ee.EEException = type("EEException", (Exception,), {})
    return ee


def settings(**kw):
    return Settings(_env_file=None, **kw)


def provider(ee=None, **kw):
    return ge.EarthEngineProvider(settings=settings(**kw), ee_module=ee or fake_ee())


def scenes_chain(ee):
    """The mock at the end of ImageCollection().filterBounds().filterDate().filter()."""
    return ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.filter.return_value


# ----------------------------------------------------------------------------- configuration and credentials
def test_not_configured_without_project_and_never_touches_earth_engine():
    ee = fake_ee()
    with pytest.raises(SatelliteNotConfigured, match="GEE_PROJECT"):
        provider(ee).scene_stats(GEOM, date(2025, 8, 1), date(2025, 8, 31))
    ee.Initialize.assert_not_called()


def test_service_account_json_is_used_as_key_data():
    ee = fake_ee()
    provider(ee, gee_project="my-proj", gee_service_account_json=SECRET_KEY_JSON).ensure_ready()
    ee.ServiceAccountCredentials.assert_called_once_with(None, key_data=SECRET_KEY_JSON)
    ee.Initialize.assert_called_once_with(credentials=ee.ServiceAccountCredentials.return_value, project="my-proj")


def test_service_account_key_file_with_optional_email(tmp_path):
    key = tmp_path / "key.json"; key.write_text(SECRET_KEY_JSON)
    ee = fake_ee()
    provider(ee, gee_project="p", gee_service_account_key_file=str(key), gee_service_account_email="svc@proj.iam.gserviceaccount.com").ensure_ready()
    ee.ServiceAccountCredentials.assert_called_once_with("svc@proj.iam.gserviceaccount.com", key_file=str(key))


def test_json_wins_over_key_file(tmp_path):
    key = tmp_path / "key.json"; key.write_text("{}")
    ee = fake_ee()
    provider(ee, gee_project="p", gee_service_account_json=SECRET_KEY_JSON, gee_service_account_key_file=str(key)).ensure_ready()
    assert ee.ServiceAccountCredentials.call_args.kwargs.keys() == {"key_data"}


def test_falls_back_to_saved_user_credentials_for_local_development():
    ee = fake_ee()
    provider(ee, gee_project="my-proj").ensure_ready()
    ee.ServiceAccountCredentials.assert_not_called()
    ee.Initialize.assert_called_once_with(project="my-proj")


@pytest.mark.parametrize("kw,match", [({"gee_service_account_json": "{not json"}, "not valid JSON"), ({"gee_service_account_key_file": "/no/such/key.json"}, "existing file")])
def test_bad_credential_configuration_is_reported_clearly(kw, match):
    with pytest.raises(SatelliteNotConfigured, match=match):
        provider(gee_project="p", **kw).ensure_ready()


def test_initialisation_failure_never_leaks_the_secret_key():
    ee = fake_ee()
    ee.Initialize.side_effect = ee.EEException(f"auth failed for key {SECRET_KEY_JSON}")
    with pytest.raises(SatelliteNotConfigured) as e:
        provider(ee, gee_project="p", gee_service_account_json=SECRET_KEY_JSON).ensure_ready()
    assert "TOPSECRET" not in str(e.value) and "private_key" not in str(e.value) and "GEE_PROJECT" in str(e.value)


def test_missing_user_credentials_gives_actionable_message():
    ee = fake_ee()
    ee.Initialize.side_effect = OSError("no credentials")
    with pytest.raises(SatelliteNotConfigured, match="earthengine authenticate"):
        provider(ee, gee_project="p").ensure_ready()


def test_initialises_only_once_even_with_repeated_calls():
    ee = fake_ee()
    p = provider(ee, gee_project="p")
    for _ in range(3):
        p.ensure_ready()
    assert ee.Initialize.call_count == 1


def test_secret_is_hidden_in_settings_repr():
    s = settings(gee_project="p", gee_service_account_json=SECRET_KEY_JSON)
    assert "TOPSECRET" not in repr(s) and "TOPSECRET" not in str(s.model_dump())


# ----------------------------------------------------------------------------- NDVI bands and cloud mask
class FakeBand:
    """Records which SCL class values are tested, so we can assert the mask without Earth Engine."""
    def __init__(self, log): self.log = log
    def eq(self, v): self.log.append(v); return self
    def Or(self, other): return self
    def Not(self): return self


def test_cloud_mask_removes_the_expected_scl_classes():
    seen = []
    img = MagicMock()
    img.select.return_value = FakeBand(seen)
    ge.EarthEngineProvider._mask_clouds(img)
    img.select.assert_called_once_with("SCL")
    assert sorted(seen) == [1, 3, 8, 9, 10, 11]            # saturated, cloud shadow, medium+high cloud, thin cirrus, snow/ice
    img.updateMask.assert_called_once()


def test_ndvi_uses_nir_band_8_and_red_band_4():
    img = MagicMock()
    ge.EarthEngineProvider._ndvi(img)
    img.normalizedDifference.assert_called_once_with(["B8", "B4"])        # (B8 - B4) / (B8 + B4)
    img.normalizedDifference.return_value.rename.assert_called_once_with("NDVI")


# ----------------------------------------------------------------------------- scene statistics
def test_scene_stats_queries_harmonized_sentinel2_for_the_date_range_and_returns_properties():
    ee = fake_ee()
    chain = scenes_chain(ee)
    chain.map.return_value = "mapped"
    ee.FeatureCollection.return_value.getInfo.return_value = {"features": [
        {"properties": {"date": "2025-08-05", "ndvi_mean": 0.61, "valid_fraction": 0.95, "cloud_pct": 3.2, "tile": "43PCR"}},
        {"properties": {"date": "2025-08-10", "ndvi_mean": None, "valid_fraction": 0, "cloud_pct": 70, "tile": "43PCR"}}]}
    rows = provider(ee, gee_project="p").scene_stats(GEOM, date(2025, 8, 1), date(2025, 8, 31))
    ee.ImageCollection.assert_called_once_with("COPERNICUS/S2_SR_HARMONIZED")
    ee.ImageCollection.return_value.filterBounds.return_value.filterDate.assert_called_once_with("2025-08-01", "2025-09-01")   # end date is inclusive
    ee.Filter.lt.assert_called_once_with("CLOUDY_PIXEL_PERCENTAGE", 80.0)
    ee.FeatureCollection.assert_called_once_with("mapped")
    assert rows[0]["ndvi_mean"] == 0.61 and rows[1]["ndvi_mean"] is None            # a fully clouded scene stays "no value", never 0


def test_scene_stats_with_no_scenes_returns_empty_list():
    ee = fake_ee()
    ee.FeatureCollection.return_value.getInfo.return_value = {"features": []}
    assert provider(ee, gee_project="p").scene_stats(GEOM, date(2025, 8, 1), date(2025, 8, 31)) == []
    ee.FeatureCollection.return_value.getInfo.return_value = None
    assert provider(ee, gee_project="p").scene_stats(GEOM, date(2025, 8, 1), date(2025, 8, 31)) == []


def test_per_scene_function_emits_the_expected_properties():
    ee = fake_ee()
    ee.FeatureCollection.return_value.getInfo.return_value = {"features": []}
    provider(ee, gee_project="p").scene_stats(GEOM, date(2025, 8, 1), date(2025, 8, 31))
    per_scene = scenes_chain(ee).map.call_args.args[0]
    per_scene(MagicMock())
    props = ee.Feature.call_args.args[1]
    assert set(props) == {"date", "ndvi_mean", "valid_fraction", "cloud_pct", "tile"}


@pytest.mark.parametrize("message,exc,fragment", [
    ("Computation timed out.", SatelliteTimeout, "too long"),
    ("User memory limit exceeded", SatelliteError, "could not process"),
    ("Quota exceeded for requests per minute", SatelliteError, "usage limit"),
    ("Caller does not have required permission to use project", SatelliteNotConfigured, "permission"),
])
def test_earth_engine_errors_are_translated_without_leaking_internals(message, exc, fragment):
    ee = fake_ee()
    ee.FeatureCollection.return_value.getInfo.side_effect = ee.EEException(message + " internal-id-12345")
    with pytest.raises(exc) as e:
        provider(ee, gee_project="p").scene_stats(GEOM, date(2025, 8, 1), date(2025, 8, 31))
    assert fragment in str(e.value).lower() and "internal-id-12345" not in str(e.value)
    if exc is SatelliteError:
        assert not isinstance(e.value, (SatelliteTimeout, SatelliteNotConfigured))


def test_network_failure_becomes_satellite_error():
    ee = fake_ee()
    ee.FeatureCollection.return_value.getInfo.side_effect = ConnectionError("down")
    with pytest.raises(SatelliteError, match="internet"):
        provider(ee, gee_project="p").scene_stats(GEOM, date(2025, 8, 1), date(2025, 8, 31))


# ----------------------------------------------------------------------------- map layer
def mosaic(ee):
    return ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.map.return_value.mosaic.return_value.clip.return_value


def test_map_layer_returns_tile_url_and_attribution():
    ee = fake_ee()
    ee.Image.return_value.where.return_value.where.return_value.updateMask.return_value.getMapId.return_value = {"tile_fetcher": MagicMock(url_format="https://tiles/{z}/{x}/{y}")}
    out = provider(ee, gee_project="p").map_layer(GEOM, date(2025, 8, 10), Thresholds(), "classes")
    assert out["tile_url"] == "https://tiles/{z}/{x}/{y}" and out["date"] == "2025-08-10" and "Copernicus" in out["attribution"] and "temporary" in out["expires_note"]


def test_class_layer_uses_the_configured_thresholds_and_legend_colours():
    ee = fake_ee()
    final = ee.Image.return_value.where.return_value.where.return_value.updateMask.return_value
    final.getMapId.return_value = {"tile_fetcher": MagicMock(url_format="u")}
    provider(ee, gee_project="p").map_layer(GEOM, date(2025, 8, 10), Thresholds(low=0.3, high=0.6), "classes")
    assert mosaic(ee).gte.call_args_list == [call(0.3), call(0.6)]
    vis = final.getMapId.call_args.args[0]
    assert vis["palette"] == ["#dc2626", "#eab308", "#16a34a"] and (vis["min"], vis["max"]) == (0, 2)            # red, yellow, green


def test_gradient_layer_and_invalid_style():
    ee = fake_ee()
    mosaic(ee).getMapId.return_value = {"tile_fetcher": MagicMock(url_format="u")}
    out = provider(ee, gee_project="p").map_layer(GEOM, date(2025, 8, 10), Thresholds(), "gradient")
    assert out["style"] == "gradient" and mosaic(ee).getMapId.call_args.args[0]["max"] == 0.9
    with pytest.raises(ValueError):
        provider(ee, gee_project="p").map_layer(GEOM, date(2025, 8, 10), Thresholds(), "rainbow")


def test_map_layer_errors_are_translated():
    ee = fake_ee()
    ee.Image.return_value.where.return_value.where.return_value.updateMask.return_value.getMapId.side_effect = ee.EEException("Image.getMapId: Computation timed out.")
    with pytest.raises(SatelliteTimeout):
        provider(ee, gee_project="p").map_layer(GEOM, date(2025, 8, 10), Thresholds(), "classes")


# ----------------------------------------------------------------------------- smoke-test helper (pure orchestration)
def test_smoke_helper_reports_what_the_provider_returned_and_never_invents():
    from backend.app.gee_smoke import run_smoke

    class P:
        def __init__(self, rows): self.rows, self.layers = rows, 0
        def scene_stats(self, geometry, start, end): return self.rows
        def map_layer(self, geometry, d, th, style): self.layers += 1; return {"tile_url": "https://t/{z}/{x}/{y}"}

    today = __import__("datetime").datetime.now(__import__("datetime").UTC).date().isoformat()
    good = P([{"date": today, "ndvi_mean": 0.6, "valid_fraction": 0.9}, {"date": today, "ndvi_mean": None, "valid_fraction": 0.0}])
    out = run_smoke(good, 14.28, 74.44)
    assert out["scenes_returned"] == 2 and len(out["usable_observations"]) == 1 and out["layer"]["tile_url"].endswith("{z}/{x}/{y}") and good.layers == 1
    empty = P([])
    out = run_smoke(empty, 14.28, 74.44)
    assert out["usable_observations"] == [] and out["layer"] is None and empty.layers == 0       # no map requested when there is no observation