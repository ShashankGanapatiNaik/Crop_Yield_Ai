from datetime import date

import numpy as np
import pandas as pd
import pytest

from ml.src import data_download as dd
from ml.src import data_merge as dm
from ml.src.data_cleaning import clean_apy
from ml.src.data_validation import validate_processed
from ml.src.real_config import load_config
from tests.real_fixtures import kaggle_style_raw


def daily(start="2008-01-01", end="2012-12-31", rain=5.0, tmean=26.0):
    idx = pd.date_range(start, end, name="date")
    return pd.DataFrame({"rain_mm": rain, "tmean_c": tmean, "tmax_c": tmean + 6, "rh_pct": 70.0}, index=idx)


# ----------------------------------------------------------------------------- season windows & weather aggregation
def test_season_windows_follow_crop_year():
    assert dm.season_window("Kharif", 2010) == (date(2010, 6, 1), date(2010, 10, 31))
    assert dm.season_window("Rabi", 2010) == (date(2010, 11, 1), date(2011, 2, 28))      # Rabi of crop-year 2010-11 straddles New Year
    assert dm.season_window("Summer", 2010) == (date(2011, 3, 1), date(2011, 5, 31))


def test_season_weather_totals_and_means():
    f = dm.season_weather_features(daily(), "Kharif", 2010)
    assert f["rainfall_mm"] == pytest.approx(153 * 5.0) and f["temperature_c"] == 26.0 and f["temp_max_c"] == 32.0 and f["humidity_pct"] == 70.0


def test_thin_coverage_is_rejected_not_filled():
    d = daily(); d.loc["2010-06-01":"2010-08-31", "rain_mm"] = np.nan          # 92 of 153 days missing
    assert dm.season_weather_features(d, "Kharif", 2010) is None
    assert dm.season_weather_features(daily("2010-08-01", "2010-12-31"), "Kharif", 2010) is None   # season window not covered at all


def test_few_missing_days_are_scaled_to_full_window():
    d = daily(); d.loc["2010-06-01":"2010-06-10", "rain_mm"] = np.nan          # 10 of 153 days missing (93.5% coverage)
    f = dm.season_weather_features(d, "Kharif", 2010)
    assert f["rainfall_mm"] == pytest.approx(143 * 5.0 * 153 / 143)


def test_humidity_absent_stays_nan_never_invented():
    d = daily(); d["rh_pct"] = np.nan
    assert np.isnan(dm.season_weather_features(d, "Kharif", 2010)["humidity_pct"])


# ----------------------------------------------------------------------------- full merge with accounting
def _inputs():
    clean, _ = clean_apy(kaggle_style_raw(), load_config())
    cen = pd.DataFrame([{"region": "Karnataka", "district": "uttara kannada", "latitude": 14.5, "longitude": 74.5},
                        {"region": "Karnataka", "district": "dharwad", "latitude": 15.4, "longitude": 75.0},
                        {"region": "Odisha", "district": "khordha", "latitude": np.nan, "longitude": np.nan}])   # unresolved centroid
    soil = pd.DataFrame([{"region": "Karnataka", "district": "uttara kannada", "soil_ph": 5.9, "soil_organic_carbon": 18.0, "soil_sand_pct": 40.0, "soil_clay_pct": 25.0}])
    return clean, cen, soil


def test_merge_joins_on_keys_and_accounts_for_every_dropped_row():
    clean, cen, soil = _inputs()
    loader = lambda st, d: daily("2005-01-01", "2013-12-31") if d == "uttara kannada" else None   # dharwad has no weather file
    cfg = load_config()
    out, rep = dm.merge_dataset(clean, cen, loader, soil, "nasa_power", "soilgrids", cfg)
    assert set(out.district) == {"uttara kannada"}
    assert rep["dropped"]["no_district_centroid"] >= 1 and rep["dropped"]["no_weather_file_for_district"] >= 1
    assert (out.soil_ph == 5.9).all() and (out.weather_source.str.startswith("nasa_power")).all() and (out.yield_source == "apy_observed").all()
    assert len(out) + sum(rep["dropped"].values()) == rep["rows_clean"]           # nothing disappears silently
    assert not out.duplicated(["region", "district", "crop", "season", "year"]).any()
    row = out[(out.year == 2010) & (out.crop == "Rice")].iloc[0]
    assert row.rainfall_mm == pytest.approx(765.0) and row.yield_t_ha == pytest.approx(3.0)


def test_missing_soil_row_dropped_when_required_and_kept_when_not():
    clean, cen, _ = _inputs()
    soil_other = pd.DataFrame([{"region": "Karnataka", "district": "dharwad", "soil_ph": 6.0, "soil_organic_carbon": 10, "soil_sand_pct": 30, "soil_clay_pct": 30}])
    loader = lambda st, d: daily("2005-01-01", "2013-12-31")
    cfg = load_config()
    out, rep = dm.merge_dataset(clean, cen, loader, soil_other, "nasa_power", "soilgrids", cfg)
    assert set(out.district) == {"dharwad"} and rep["dropped"]["no_soil_record"] > 0
    cfg["soil"]["require_soil"] = False
    out2, _ = dm.merge_dataset(clean, cen, loader, soil_other, "nasa_power", "soilgrids", cfg)
    assert "uttara kannada" in set(out2.district) and out2.loc[out2.district == "uttara kannada", "soil_ph"].isna().all()


def test_processed_validation_flags_problems():
    clean, cen, soil = _inputs()
    out, _ = dm.merge_dataset(clean, cen, lambda s, d: daily("2005-01-01", "2013-12-31"), soil, "nasa_power", "soilgrids", load_config())
    assert not validate_processed(out, min_rows=5).ok                       # < 8 distinct years in this tiny fixture: must be rejected
    big = pd.concat([out.assign(year=out.year + k * 100 // 100 + k) for k in range(9)]).drop_duplicates(["region", "district", "crop", "season", "year"])
    assert validate_processed(big, min_rows=5).ok
    bad = big.copy(); bad.loc[bad.index[0], "rainfall_mm"] = 99999
    assert not validate_processed(bad, min_rows=5).ok
    dup = pd.concat([big, big.iloc[[0]]]); assert not validate_processed(dup, min_rows=5).ok


def test_save_and_load_roundtrip_has_dataset_version(tmp_path):
    clean, cen, soil = _inputs()
    out, rep = dm.merge_dataset(clean, cen, lambda s, d: daily("2005-01-01", "2013-12-31"), soil, "nasa_power", "soilgrids", load_config())
    meta = dm.save_processed(out, rep, csv=tmp_path / "d.csv", meta=tmp_path / "d.json")
    back, m2 = dm.load_processed(tmp_path / "d.csv", tmp_path / "d.json")
    assert len(back) == len(out) and m2["dataset_version"] == meta["dataset_version"] and m2["provenance_classes"]["yield"] == "observed"
    with pytest.raises(FileNotFoundError):
        dm.load_processed(tmp_path / "nope.csv")


# ----------------------------------------------------------------------------- download parsers with MOCKED payloads
def test_parse_nasa_power_handles_fill_values():
    payload = {"properties": {"parameter": {"PRECTOTCORR": {"20100601": 12.5, "20100602": -999.0}, "T2M": {"20100601": 25.1, "20100602": 25.3},
                                              "T2M_MAX": {"20100601": 29.0, "20100602": 30.0}, "RH2M": {"20100601": 88.0, "20100602": 87.0}}}}
    d = dd.parse_power(payload)
    assert d.loc["2010-06-01", "rain_mm"] == 12.5 and np.isnan(d.loc["2010-06-02", "rain_mm"]) and d.loc["2010-06-02", "rh_pct"] == 87.0


def test_parse_open_meteo_has_no_invented_humidity():
    d = dd.parse_open_meteo({"daily": {"time": ["2010-06-01", "2010-06-02"], "precipitation_sum": [1.0, None], "temperature_2m_mean": [25.0, 26.0], "temperature_2m_max": [30.0, 31.0]}})
    assert d["rain_mm"].iloc[0] == 1.0 and d["rh_pct"].isna().all() and len(d) == 2


def test_fetch_weather_routes_to_provider_and_params():
    seen = {}
    def fake(url, params=None, **kw):
        seen["url"], seen["p"] = url, params
        return {"properties": {"parameter": {"PRECTOTCORR": {"20100601": 1.0}, "T2M": {"20100601": 2.0}, "T2M_MAX": {"20100601": 3.0}, "RH2M": {"20100601": 4.0}}}}
    dd.fetch_weather_daily(14.5, 74.5, date(2010, 6, 1), date(2010, 6, 1), "nasa_power", get=fake)
    assert "nasa" in seen["url"] or "power" in seen["url"]
    assert seen["p"]["latitude"] == 14.5 and seen["p"]["start"] == "20100601" and seen["p"]["community"] == "AG"
    with pytest.raises(ValueError):
        dd.fetch_weather_daily(1, 1, date(2010, 1, 1), date(2010, 1, 2), "unknown")


def test_parse_soilgrids_applies_d_factor_and_averages_depths():
    payload = {"properties": {"layers": [
        {"name": "phh2o", "unit_measure": {"d_factor": 10}, "depths": [{"label": "0-5cm", "values": {"mean": 60}}, {"label": "5-15cm", "values": {"mean": 64}}]},
        {"name": "soc", "unit_measure": {"d_factor": 10}, "depths": [{"label": "0-5cm", "values": {"mean": 200}}]},
        {"name": "sand", "unit_measure": {"d_factor": 10}, "depths": [{"label": "0-5cm", "values": {"mean": 450}}]},
        {"name": "clay", "unit_measure": {"d_factor": 10}, "depths": [{"label": "0-5cm", "values": {"mean": None}}]},
        {"name": "other", "unit_measure": {"d_factor": 1}, "depths": []}]}}
    s = dd.parse_soilgrids(payload)
    assert s["soil_ph"] == pytest.approx(6.2) and s["soil_organic_carbon"] == pytest.approx(20.0) and s["soil_sand_pct"] == pytest.approx(45.0)
    assert np.isnan(s["soil_clay_pct"]) and "other" not in s


def test_geocode_validates_india_and_state(monkeypatch):
    good = lambda url, params=None, **k: [{"lat": "14.8", "lon": "74.1", "address": {"state": "Karnataka"}}]
    assert dd.geocode_district("Karnataka", "uttara kannada", good) == {"latitude": 14.8, "longitude": 74.1}
    wrong_state = lambda url, params=None, **k: [{"lat": "14.8", "lon": "74.1", "address": {"state": "Goa"}}]
    assert dd.geocode_district("Karnataka", "x", wrong_state) is None                        # same name, different state: refused
    abroad = lambda url, params=None, **k: [{"lat": "51.5", "lon": "-0.1", "address": {"state": "Karnataka"}}]
    assert dd.geocode_district("Karnataka", "x", abroad) is None
    assert dd.geocode_district("Karnataka", "x", lambda *a, **k: []) is None


def test_resumable_downloads_skip_cached_and_record_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "_sleep", lambda s: None)
    cen = pd.DataFrame([{"region": "Karnataka", "district": "a", "latitude": 14.0, "longitude": 75.0}, {"region": "Karnataka", "district": "b", "latitude": 15.0, "longitude": 75.0}])
    calls = []
    def fake(url, params=None, **k):
        calls.append(params["latitude"])
        if params["latitude"] == 15.0:
            raise RuntimeError("boom")
        return {"properties": {"parameter": {"PRECTOTCORR": {"20100601": 1.0}, "T2M": {"20100601": 2.0}, "T2M_MAX": {"20100601": 3.0}, "RH2M": {"20100601": 4.0}}}}
    s1 = dd.download_weather(cen, 2010, 2010, "nasa_power", sleep=0, raw_dir=tmp_path, get=fake)
    assert s1["fetched"] == 1 and len(s1["failed"]) == 1 and dd.load_weather("nasa_power", "Karnataka", "a", tmp_path) is not None
    s2 = dd.download_weather(cen.iloc[:1], 2010, 2010, "nasa_power", sleep=0, raw_dir=tmp_path, get=fake)
    assert s2["cached"] == 1 and s2["fetched"] == 0 and calls.count(14.0) == 1             # cached district is not requested again


def test_import_apy_copies_and_records_checksum(tmp_path):
    src = tmp_path / "in.csv"; kaggle = __import__("tests.real_fixtures", fromlist=["x"]).kaggle_style_raw(); kaggle.to_csv(src, index=False)
    dest = dd.import_apy(src, raw_dir=tmp_path / "raw")
    prov = __import__("json").loads((tmp_path / "raw" / "apy_provenance.json").read_text())
    assert dest.exists() and prov["sha256"] == dd.sha256_file(dest) and prov["source"] == str(src)
    with pytest.raises(FileNotFoundError):
        dd.import_apy(tmp_path / "missing.csv", raw_dir=tmp_path / "raw")


def test_import_centroids_csv_normalises_and_validates(tmp_path):
    p = tmp_path / "c.csv"; pd.DataFrame({"state": ["Orissa"], "district": ["Khordha District"], "lat": [20.2], "lon": [85.8]}).to_csv(p, index=False)
    out = dd.import_centroids_csv(p, cache=tmp_path / "cache.csv")
    assert out.iloc[0].region == "Odisha" and out.iloc[0].district == "khordha"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        dd.import_centroids_csv(p, cache=tmp_path / "cache.csv")