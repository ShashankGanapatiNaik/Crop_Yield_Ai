import pytest

from backend.app.services import recommendations as rec

BASE = dict(region="Karnataka", season="Kharif", soil_type="Red", irrigation="Canal", ph=6.2, nitrogen=105, phosphorus=30, potassium=125,
            rainfall_mm=900, temperature_c=27, humidity_pct=70, area_ha=2, fertilizer_kg_ha=95, year=2024)
PRED = {"yield": 4.0, "lower": 3.5, "upper": 4.5, "profit": 50000, "roi_pct": 40, "revenue": 200000}


def by_cat(recs, cat):
    return [r for r in recs if r["category"] == cat]


def test_healthy_field_has_no_critical_agronomy_alerts():
    recs = rec.generate(BASE, "Rice", PRED)
    assert not [r for r in recs if r["severity"] == "critical"]
    assert by_cat(recs, "soil_ph")[0]["severity"] == "success"


def test_water_deficit_flagged_with_gain():
    recs = rec.generate({**BASE, "rainfall_mm": 150, "irrigation": "Rainfed"}, "Rice", PRED, levers={"irrigation": 1.2})
    w = by_cat(recs, "irrigation")[0]
    assert w["severity"] == "critical" and "deficit" in w["title"].lower() and w["expected_gain_t_ha"] == 1.2


def test_waterlogging_warning():
    w = by_cat(rec.generate({**BASE, "rainfall_mm": 3500}, "Wheat", PRED), "irrigation")[0]
    assert "waterlogging" in w["title"].lower()


def test_acidic_soil_gets_lime_dose():
    r = by_cat(rec.generate({**BASE, "ph": 4.8}, "Rice", PRED), "soil_ph")[0]
    assert "acidic" in r["title"].lower() and "lime" in r["detail"].lower()
    # (6.2 - 4.8) * 2.5 = 3.5 t/ha
    assert "3.5 t/ha" in r["detail"]


def test_alkaline_soil_gets_sulphur_gypsum_advice():
    r = by_cat(rec.generate({**BASE, "ph": 8.9}, "Rice", PRED), "soil_ph")[0]
    assert r["severity"] == "critical" and ("gypsum" in r["detail"].lower() or "sulphur" in r["detail"].lower())


def test_low_nitrogen_gives_urea_quantity():
    r = by_cat(rec.generate({**BASE, "nitrogen": 20, "fertilizer_kg_ha": 0}, "Maize", PRED), "nitrogen")[0]
    assert r["severity"] in ("warning", "critical") and "urea" in r["detail"]


def test_excess_potassium_warning():
    r = by_cat(rec.generate({**BASE, "potassium": 500, "fertilizer_kg_ha": 200}, "Rice", PRED), "potassium")[0]
    assert "excess" in r["title"].lower()


def test_wrong_season_flagged():
    recs = rec.generate({**BASE, "season": "Rabi"}, "Cotton", PRED)
    assert any(r["category"] == "crop" and r["severity"] == "critical" for r in recs)


def test_heat_stress():
    assert any("heat" in r["title"].lower() for r in rec.generate({**BASE, "temperature_c": 38}, "Wheat", PRED))


def test_loss_is_critical_economics():
    r = by_cat(rec.generate(BASE, "Rice", {**PRED, "profit": -20000}), "economics")[0]
    assert r["severity"] == "critical"


def test_better_alternative_suggested():
    alts = [{"crop": "Rice", "profit": 40000, "season_suitable": True}, {"crop": "Maize", "profit": 90000, "season_suitable": True}]
    recs = rec.generate(BASE, "Rice", PRED, alternatives=alts)
    assert any("more profitable" in r["title"].lower() and "Maize" in r["detail"] for r in recs)


def test_sorted_by_severity():
    recs = rec.generate({**BASE, "ph": 4.5, "rainfall_mm": 100, "irrigation": "Rainfed"}, "Rice", PRED)
    order = {"critical": 0, "warning": 1, "info": 2, "success": 3}
    sev = [order[r["severity"]] for r in recs]
    assert sev == sorted(sev)


@pytest.mark.parametrize("crop", list(rec.CROP_PROFILE))
def test_every_crop_produces_recommendations(crop):
    assert rec.generate(BASE, crop, PRED)


def test_deficit_downgraded_when_model_sees_small_gain():
    recs = rec.generate({**BASE, "rainfall_mm": 250, "irrigation": "Rainfed"}, "Rice", PRED, levers={"irrigation": 0.09})
    w = by_cat(recs, "irrigation")[0]
    assert w["severity"] == "info" and "small" in w["title"].lower() and "+0.09" in w["detail"]
    assert "250 mm" in w["detail"] and "whole season" in w["detail"]


def test_deficit_stays_serious_when_gain_is_large():
    w = by_cat(rec.generate({**BASE, "rainfall_mm": 250, "irrigation": "Rainfed"}, "Rice", PRED, levers={"irrigation": 0.9}), "irrigation")[0]
    assert w["severity"] in ("warning", "critical") and "deficit" in w["title"].lower()


def test_heavy_monsoon_rice_is_not_flagged_as_waterlogging():
    # coastal-style 3000 mm paddy season: fine for rice, but still flagged for a dryland crop
    assert by_cat(rec.generate({**BASE, "rainfall_mm": 3000}, "Rice", PRED), "irrigation")[0]["severity"] == "success"
    assert "waterlogging" in by_cat(rec.generate({**BASE, "rainfall_mm": 3000}, "Chickpea", PRED), "irrigation")[0]["title"].lower()