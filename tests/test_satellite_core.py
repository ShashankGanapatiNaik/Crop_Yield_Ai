"""Provider-independent satellite logic: NDVI maths, thresholds, geometry validation, aggregation, interpretation. No network."""
import math
from datetime import date

import numpy as np
import pytest

from backend.app.services import satellite as sat
from backend.app.services.satellite import GeometryError, Observation, Thresholds

TH = Thresholds()


def square(lat=14.28, lon=74.44, d=0.001):
    return {"type": "Polygon", "coordinates": [[[lon, lat], [lon + d, lat], [lon + d, lat + d], [lon, lat + d], [lon, lat]]]}


# ----------------------------------------------------------------------------- NDVI = (NIR - Red) / (NIR + Red)
@pytest.mark.parametrize("nir,red,expected", [(0.5, 0.1, 2 / 3), (0.4, 0.4, 0.0), (0.0, 0.3, -1.0), (0.3, 0.0, 1.0), (0.6, 0.05, 0.55 / 0.65)])
def test_ndvi_formula(nir, red, expected):
    assert sat.ndvi(nir, red) == pytest.approx(expected)


def test_ndvi_is_independent_of_reflectance_scale():
    """Sentinel-2 surface reflectance is stored as DN = reflectance x 10000; the ratio must not change."""
    assert sat.ndvi(5000, 1000) == pytest.approx(sat.ndvi(0.5, 0.1))


@pytest.mark.parametrize("nir,red", [(0, 0), (float("nan"), 0.1), (0.1, float("inf"))])
def test_ndvi_undefined_inputs_return_none_not_zero(nir, red):
    assert sat.ndvi(nir, red) is None


def test_ndvi_stays_within_minus_one_and_one():
    rng = np.random.default_rng(0)
    nir, red = rng.uniform(0, 1, 1000), rng.uniform(0, 1, 1000)
    out = sat.ndvi_array(nir, red)
    assert np.nanmin(out) >= -1 and np.nanmax(out) <= 1


def test_ndvi_array_marks_zero_sum_pixels_as_nan():
    out = sat.ndvi_array([0.5, 0.0], [0.1, 0.0])
    assert out[0] == pytest.approx(2 / 3) and math.isnan(out[1])


# ----------------------------------------------------------------------------- thresholds
@pytest.mark.parametrize("v,cat", [(-0.3, "low"), (0.0, "low"), (0.199, "low"), (0.2, "moderate"), (0.499, "moderate"), (0.5, "high"), (0.9, "high")])
def test_default_classification(v, cat):
    assert sat.classify(v, TH) == cat


def test_thresholds_are_configurable_and_validated():
    custom = Thresholds(low=0.3, high=0.6)
    assert sat.classify(0.25, custom) == "low" and sat.classify(0.55, custom) == "moderate" and sat.classify(0.6, custom) == "high"
    assert "0.3" in sat.legend(custom)[0]["range"]
    for bad in [(0.5, 0.2), (0.3, 0.3), (-1.0, 0.5), (0.2, 1.0)]:
        with pytest.raises(ValueError):
            Thresholds(*bad)


def test_legend_colours_are_red_yellow_green():
    assert [x["category"] for x in sat.legend(TH)] == ["low", "moderate", "high"]
    assert [x["color"] for x in sat.legend(TH)] == [sat.COLORS["low"], sat.COLORS["moderate"], sat.COLORS["high"]]


# ----------------------------------------------------------------------------- coordinates
@pytest.mark.parametrize("lat,lon,field", [(91, 74, "latitude"), (-91, 74, "latitude"), (14, 181, "longitude"), (14, -181, "longitude"), ("x", 74, "latitude"),
                                           (float("nan"), 74, "latitude"), (14, float("inf"), "latitude"), (-70, 0, "latitude"), (85, 0, "latitude")])
def test_invalid_coordinates_are_rejected(lat, lon, field):
    with pytest.raises(GeometryError) as e:
        sat.validate_coordinate(lat, lon)
    assert e.value.field == field


def test_valid_coordinate_passes():
    assert sat.validate_coordinate(14.28, 74.44) == (14.28, 74.44)
    assert sat.validate_coordinate("14.28", "74.44") == (14.28, 74.44)          # numeric strings from a form


def test_swapped_latitude_longitude_is_caught_only_when_out_of_range():
    """A swap is detectable only if the swapped value cannot be a latitude (e.g. longitude 140 typed as latitude).
    A numerically valid swap (Honnavar 14.28N 74.44E typed as 74.44, 14.28) is indistinguishable from a real point in the Arctic Ocean,
    so the UI also shows the chosen point on a map before saving."""
    with pytest.raises(GeometryError, match="latitude 140"):
        sat.validate_coordinate(140.0, 14.28)
    assert sat.validate_coordinate(74.44, 14.28) == (74.44, 14.28)


# ----------------------------------------------------------------------------- polygons
def test_valid_polygon_area_centroid_and_closure():
    r = sat.validate_polygon(square())
    assert r["area_ha"] == pytest.approx(1.2, rel=0.01)                       # 111.3 m x 107.9 m
    assert r["geometry"]["coordinates"][0][0] == r["geometry"]["coordinates"][0][-1]
    assert r["centroid"][0] == pytest.approx(14.2805, abs=1e-4) and r["centroid"][1] == pytest.approx(74.4405, abs=1e-4)
    assert r["bbox"] == pytest.approx([74.44, 14.28, 74.441, 14.281])


def test_open_ring_feature_wrapper_and_duplicate_points_are_normalised():
    g = square()
    g["coordinates"][0] = g["coordinates"][0][:-1]                             # not closed
    g["coordinates"][0].insert(1, g["coordinates"][0][1])                      # consecutive duplicate
    r = sat.validate_polygon({"type": "Feature", "geometry": g, "properties": {}})
    assert len(r["geometry"]["coordinates"][0]) == 5 and r["area_ha"] == pytest.approx(1.2, rel=0.01)


def test_one_hectare_square_has_one_hectare():
    d_lat = 100 / 111_320
    d_lon = 100 / (111_320 * math.cos(math.radians(14.28)))
    g = {"type": "Polygon", "coordinates": [[[74.44, 14.28], [74.44 + d_lon, 14.28], [74.44 + d_lon, 14.28 + d_lat], [74.44, 14.28 + d_lat], [74.44, 14.28]]]}
    assert sat.validate_polygon(g)["area_ha"] == pytest.approx(1.0, rel=0.01)


@pytest.mark.parametrize("bad,match", [
    ({"type": "Point", "coordinates": [74, 14]}, "Polygon"),
    ("not geojson", "Polygon"),
    ({"type": "Polygon", "coordinates": []}, "no coordinates"),
    ({"type": "Polygon", "coordinates": [[[74.44, 14.28], [74.441, 14.28]]]}, "at least 3"),
    ({"type": "Polygon", "coordinates": [[[74.44, 14.28], [74.441, 14.28], ["a", 14.281]]]}, "number"),
    ({"type": "Polygon", "coordinates": [[[74.44, 14.28], [74.441, 14.28], [74.441]]]}, "longitude, latitude"),
])
def test_malformed_polygons_are_rejected(bad, match):
    with pytest.raises(GeometryError, match=match):
        sat.validate_polygon(bad)


def test_polygon_with_hole_is_rejected():
    g = square()
    g["coordinates"].append(g["coordinates"][0])
    with pytest.raises(GeometryError, match="holes"):
        sat.validate_polygon(g)


def test_self_intersecting_bowtie_is_rejected():
    bow = {"type": "Polygon", "coordinates": [[[74.44, 14.28], [74.441, 14.281], [74.441, 14.28], [74.44, 14.281], [74.44, 14.28]]]}
    with pytest.raises(GeometryError, match="crosses itself"):
        sat.validate_polygon(bow)


def test_area_limits_are_enforced_and_configurable():
    with pytest.raises(GeometryError, match="below the minimum"):
        sat.validate_polygon(square(d=0.00002))                              # ~2 m x 2 m
    with pytest.raises(GeometryError, match="exceeds the maximum"):
        sat.validate_polygon(square(d=0.05))                                 # ~5 km x 5 km = ~2400 ha
    assert sat.validate_polygon(square(d=0.05), max_ha=5000)["area_ha"] > 2000
    with pytest.raises(GeometryError, match="too large"):
        sat.validate_polygon(square(d=0.5), max_ha=1e9)
    with pytest.raises(GeometryError, match="too many vertices"):
        ring = [[74.44 + 0.00001 * i, 14.28 + (0.0001 if i % 2 else 0)] for i in range(600)]
        sat.validate_polygon({"type": "Polygon", "coordinates": [ring]})


def test_polygon_outside_sentinel2_coverage_or_world_is_rejected():
    with pytest.raises(GeometryError):
        sat.validate_polygon(square(lat=89.0, lon=10.0))
    with pytest.raises(GeometryError, match="longitude"):
        sat.validate_polygon({"type": "Polygon", "coordinates": [[[200, 14], [201, 14], [201, 15], [200, 14]]]})


def test_circle_polygon_has_the_requested_area_and_validates_radius():
    c = sat.circle_polygon(14.28, 74.44, 100)
    assert sat.validate_polygon(c)["area_ha"] == pytest.approx(math.pi * 100 ** 2 / 10_000, rel=0.01)
    for r in (5, 5000):
        with pytest.raises(GeometryError, match="radius"):
            sat.circle_polygon(14.28, 74.44, r)
    with pytest.raises(GeometryError):
        sat.circle_polygon(95, 74.44, 100)


# ----------------------------------------------------------------------------- aggregation of scene rows
def row(d, m, vf, **k):
    return {"date": d, "ndvi_mean": m, "valid_fraction": vf, **k}


def test_aggregate_orders_by_date_and_rounds():
    obs = sat.aggregate_scene_rows([row("2025-08-10", 0.61234, 0.9), row("2025-08-05", 0.5, 1.0)])
    assert [o.date for o in obs] == [date(2025, 8, 5), date(2025, 8, 10)] and obs[1].ndvi_mean == 0.6123


def test_scenes_with_too_little_clear_sky_or_invalid_values_are_discarded_not_guessed():
    rows = [row("2025-08-01", 0.6, 0.2),                 # mostly cloud
            row("2025-08-02", None, 0.9),                # fully masked: no mean
            row("2025-08-03", 1.7, 0.9),                 # impossible NDVI
            row("2025-08-04", float("nan"), 0.9),
            row("2025-08-05", 0.6, None),
            row("garbage", 0.6, 0.9), {"ndvi_mean": 0.6, "valid_fraction": 0.9},
            row("2025-08-06", 0.55, 0.5)]                # exactly at the limit is kept
    obs = sat.aggregate_scene_rows(rows, min_valid_fraction=0.5)
    assert [o.date for o in obs] == [date(2025, 8, 6)]
    assert sat.aggregate_scene_rows([row("2025-08-01", 0.6, 0.4)], min_valid_fraction=0.3)[0].valid_fraction == 0.4   # threshold is configurable


def test_same_day_tiles_are_combined_with_valid_share_weights_and_capped():
    obs = sat.aggregate_scene_rows([row("2025-08-05", 0.4, 0.6), row("2025-08-05", 0.8, 0.6)])
    assert len(obs) == 1 and obs[0].scenes == 2 and obs[0].ndvi_mean == pytest.approx(0.6) and obs[0].valid_fraction == 1.0
    heavy = sat.aggregate_scene_rows([row("2025-08-05", 0.4, 0.9), row("2025-08-05", 0.8, 0.3 + 0.2)])
    assert heavy[0].ndvi_mean == pytest.approx((0.4 * 0.9 + 0.8 * 0.5) / 1.4, abs=1e-3)


def test_no_rows_means_no_observations():
    assert sat.aggregate_scene_rows([]) == [] and sat.aggregate_scene_rows([row("2025-08-01", 0.5, 0.1)]) == []


def test_trend_needs_two_observations():
    a, b = Observation(date(2025, 8, 1), 0.7, 1.0), Observation(date(2025, 8, 11), 0.5, 1.0)
    assert sat.trend_change([a]) == (None, None) and sat.trend_change([]) == (None, None)
    assert sat.trend_change([a, b]) == (a, -0.2)


# ----------------------------------------------------------------------------- interpretation: greenness indicator, never a diagnosis
BANNED = ("disease", "blight", "rust", "infect", "pest", "fungus", "virus", "deficien", "will fail", "diagnos")


@pytest.mark.parametrize("v", [0.05, 0.35, 0.75])
def test_interpretation_never_claims_a_diagnosis(v):
    i = sat.interpret(Observation(date(2025, 8, 10), v, 1.0), Observation(date(2025, 7, 31), v + 0.2, 1.0), TH)
    text = i["summary"].lower()
    assert not any(w in text for w in BANNED), text
    assert i["category"] == sat.classify(v, TH) and i["color"] == sat.COLORS[i["category"]]


def test_low_ndvi_lists_non_stress_explanations():
    s = sat.interpret(Observation(date(2025, 8, 10), 0.1, 1.0), None, TH)["summary"].lower()
    assert "bare soil" in s and "stress" in s and "harvested" in s


def test_caveats_state_the_limits_of_ndvi():
    c = " ".join(sat.interpret(Observation(date(2025, 8, 10), 0.5, 1.0), None, TH)["caveats"]).lower()
    assert "not a diagnosis" in c and "cannot identify a specific disease" in c and "growth stage" in c and "10 m" in c


def test_trend_wording_and_thresholds():
    prev = Observation(date(2025, 7, 31), 0.70, 1.0)
    assert "fell by 0.25" in sat.interpret(Observation(date(2025, 8, 10), 0.45, 1.0), prev, TH)["summary"]
    assert "rose by 0.20" in sat.interpret(Observation(date(2025, 8, 10), 0.90, 1.0), prev, TH)["summary"]
    assert "broadly unchanged" in sat.interpret(Observation(date(2025, 8, 10), 0.72, 1.0), prev, TH)["summary"]
    assert sat.interpret(Observation(date(2025, 8, 10), 0.5, 1.0), None, TH)["change"] is None