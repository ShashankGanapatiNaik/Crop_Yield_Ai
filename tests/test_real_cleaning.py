import numpy as np
import pandas as pd
import pytest

from ml.src import data_cleaning as dc
from ml.src.data_validation import ValidationError, validate_clean, validate_raw_apy
from ml.src.real_config import load_config
from tests.real_fixtures import india_portal_raw, kaggle_style_raw


@pytest.mark.parametrize("raw,expected", [("Orissa", "Odisha"), ("  uttar   pradesh ", "Uttar Pradesh"), ("Jammu & Kashmir", "Jammu and Kashmir"), ("TAMIL NADU", "Tamil Nadu"), (None, None), ("", None)])
def test_normalize_state(raw, expected):
    assert dc.normalize_state(raw) == expected


@pytest.mark.parametrize("raw,expected", [("Khordha District", "khordha"), ("MYSORE", "mysuru"), ("Bengaluru (Urban)", "bengaluru"), ("D.K. Kannada", "d k kannada"), ("  ", None)])
def test_normalize_district(raw, expected):
    assert dc.normalize_district(raw) == expected


def test_normalize_crop_season_and_year():
    assert dc.normalize_crop("Paddy") == "Rice" and dc.normalize_crop("Arhar/Tur") == "Pigeonpea" and dc.normalize_crop("Coconut") is None
    assert dc.normalize_season("Kharif     ") == "Kharif" and dc.normalize_season("Whole Year ") == "Whole Year" and dc.normalize_season("Monsoon") is None
    assert dc.parse_crop_year(1997) == 1997 and dc.parse_crop_year("2015-16") == 2015 and dc.parse_crop_year("2015-2016") == 2015
    assert np.isnan(dc.parse_crop_year("abc")) and np.isnan(dc.parse_crop_year(None)) and np.isnan(dc.parse_crop_year(99999))


def test_standardise_columns_rejects_wrong_file():
    with pytest.raises(ValueError, match="missing required column"):
        dc.standardise_columns(pd.DataFrame({"foo": [1], "bar": [2]}))


def test_clean_kaggle_layout_and_yield_formula():
    clean, rep = dc.clean_apy(kaggle_style_raw(), load_config())
    # yield is production_t / area_ha, never taken from the source
    assert np.allclose(clean["yield_t_ha"], clean["production_t"] / clean["area_ha"])
    rice = clean[(clean.crop == "Rice") & (clean.district == "uttara kannada")].set_index("year")["yield_t_ha"]
    assert rice[2010] == pytest.approx(3.0) and rice[2011] == pytest.approx(2.5)
    # dropped: zero area, missing/zero production, impossible 100 t/ha, unsupported crop, other season, tiny area
    assert 2012 not in rice.index and 2013 not in rice.index and 2014 not in rice.index and 2015 not in rice.index
    assert "Coconut" not in set(clean.crop) and not ((clean.crop == "Groundnut")).any()
    assert not ((clean.region == "Karnataka") & (clean.district == "mysuru") & (clean.crop == "Rice")).any()  # Whole Year season dropped
    # name normalisation
    assert "Odisha" in set(clean.region) and "khordha" in set(clean.district) and "mysuru" in set(clean.district)
    # duplicate key summed, not double counted
    m = clean[(clean.district == "dharwad") & (clean.crop == "Maize")].iloc[0]
    assert m.area_ha == 1000.0 and m.production_t == 2500.0 and m.yield_t_ha == pytest.approx(2.5)
    # '2009-10' parses to the crop-year start
    assert 2009 in set(clean[clean.crop == "Sorghum"].year)
    # cotton has no unit info -> excluded, not guessed
    assert "Cotton" not in set(clean.crop) and rep["units"]["dropped_no_unit_info"] == 1
    assert rep["duplicate_key_rows_merged"] == 2
    assert validate_clean(clean).ok


def test_explicit_cotton_assumption_is_honoured_and_reported():
    cfg = load_config(); cfg["cleaning"]["unit_assumptions"] = {"Cotton": 0.17}
    clean, rep = dc.clean_apy(kaggle_style_raw(), cfg)
    c = clean[clean.crop == "Cotton"].iloc[0]
    assert c.production_t == pytest.approx(900 * 0.17) and rep["units"]["assumed_units_used"] == {"Cotton": 0.17}


def test_clean_india_portal_layout_uses_explicit_units():
    clean, rep = dc.clean_apy(india_portal_raw(), load_config())
    cot = clean[clean.crop == "Cotton"].iloc[0]
    assert cot.area_ha == 100_000 and cot.production_t == pytest.approx(250_000 * 0.17)       # bales -> tonnes via the crop-specific factor
    assert cot.yield_t_ha == pytest.approx(0.425)                                              # 425 kg/ha, matching the source's own yield column
    assert rep["yield_crosscheck_agree_pct"] == 100.0
    rice = clean[clean.crop == "Rice"].iloc[0]
    assert rice.area_ha == 50_000 and rice.production_t == 100_000 and rice.yield_t_ha == pytest.approx(2.0)
    assert rep["units"]["dropped_unknown_production_unit"] == 1            # 'Furlongs' cannot be converted: dropped, never guessed
    assert rice.year == 2015


def test_implausible_yield_is_dropped_not_kept():
    """250 bales over 100k ha is 0.0004 t/ha: below the physical floor, so it must be removed rather than trained on."""
    raw = india_portal_raw(); raw.loc[0, "production"] = 250.0
    clean, rep = dc.clean_apy(raw, load_config())
    assert "Cotton" not in set(clean.crop)
    assert any(s["step"].startswith("yield outside physical bounds") and s["dropped"] >= 1 for s in rep["steps"])


def test_validation_catches_bad_raw_and_bad_clean():
    assert not validate_raw_apy(pd.DataFrame({"x": [1]})).ok
    assert not validate_raw_apy(pd.DataFrame()).ok
    r = validate_raw_apy(kaggle_style_raw())
    assert r.ok and any("no unit columns" in w for w in r.warnings)
    clean, _ = dc.clean_apy(kaggle_style_raw(), load_config())
    broken = clean.copy(); broken.loc[0, "area_ha"] = -5
    assert not validate_clean(broken).ok
    broken = pd.concat([clean, clean.iloc[[0]]]); assert not validate_clean(broken).ok           # duplicate key
    broken = clean.copy(); broken.loc[0, "yield_t_ha"] = 99; assert not validate_clean(broken).ok  # formula + bounds
    with pytest.raises(ValidationError):
        validate_clean(broken).raise_if_errors()


def test_state_filter_for_trial_runs():
    cfg = load_config(); cfg["apy"]["states"] = ["orissa"]          # aliases are normalised before matching
    clean, rep = dc.clean_apy(kaggle_style_raw(), cfg)
    assert set(clean.region) == {"Odisha"} and any(s["step"].startswith("state not in") and s["dropped"] > 0 for s in rep["steps"])