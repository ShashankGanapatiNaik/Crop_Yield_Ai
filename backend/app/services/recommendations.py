"""Hybrid recommendation engine.

Rule-based layer: agronomic reference ranges per crop (irrigation need, pH window, N/P/K supply).
Data-driven layer: (a) top-quartile benchmarks mined from the training data and (b) *model-simulated
levers* - the ML model re-predicts yield after each candidate change, so every suggestion carries
an expected gain that the model itself estimated.

All functions are pure (no I/O) so they are trivially unit-testable.
"""
from ml.src.config import CROP_PROFILE, IRRIGATION_MM

LIME_T_PER_PH_UNIT = 2.5      # t/ha of agricultural lime to raise pH by 1 unit (loam; generic guide)
UREA_N, SSP_P2O5, MOP_K2O = 0.46, 0.16, 0.60
NUTRIENT_OPT_FACTOR = 1.0


def _nutrient_supply(inputs: dict) -> dict:
    f = inputs["fertilizer_kg_ha"]
    # Effective crop-available supply = soil test + fertiliser contribution (same agronomic weights the simulator uses).
    return {"nitrogen": (inputs["nitrogen"] + 0.5 * f) * 1.2, "phosphorus": (inputs["phosphorus"] * 1.5 + 0.28 * f) * 2.0,
            "potassium": (inputs["potassium"] + 0.22 * f) * 1.2}


def water_status(inputs: dict, crop: str) -> tuple[float, float, float]:
    opt, tol = CROP_PROFILE[crop]["rain"]
    water = inputs["rainfall_mm"] + IRRIGATION_MM[inputs["irrigation"]] * 0.9
    return water, opt, tol


def generate(inputs: dict, crop: str, pred: dict, levers: dict | None = None, alternatives: list[dict] | None = None,
             benchmark: dict | None = None) -> list[dict]:
    """pred = {'yield','lower','upper'}; levers = {name: gain_t_ha}; alternatives = compare-crops rows sorted by profit."""
    prof, levers, recs = CROP_PROFILE[crop], levers or {}, []
    gain = lambda k: round(levers[k], 3) if levers.get(k, 0) > 0.01 else None

    # ---- irrigation / water balance
    water, w_opt, w_tol = water_status(inputs, crop)
    deficit = w_opt - water
    if deficit > 0.6 * w_tol:
        sev = "critical" if deficit > 1.5 * w_tol else "warning"
        better = "Drip" if inputs["irrigation"] in ("Rainfed", "Canal") else "Sprinkler"
        recs.append(dict(category="irrigation", severity=sev, title="Water deficit expected",
                         detail=f"Effective water supply is ~{water:.0f} mm against ~{w_opt:.0f} mm needed by {crop}. "
                                f"Add about {deficit:.0f} mm via supplementary irrigation (e.g. {better.lower()} irrigation for best water-use efficiency).",
                         expected_gain_t_ha=gain("irrigation")))
    elif water > w_opt + 2.2 * w_tol:
        recs.append(dict(category="irrigation", severity="warning", title="Risk of waterlogging",
                         detail=f"Expected water (~{water:.0f} mm) is far above the {w_opt:.0f} mm optimum. Ensure drainage channels, avoid irrigating, and watch for root diseases.",
                         expected_gain_t_ha=None))
    else:
        recs.append(dict(category="irrigation", severity="success", title="Water supply is adequate",
                         detail=f"Effective water (~{water:.0f} mm) is within the comfortable range for {crop}. Schedule irrigation only at critical growth stages.", expected_gain_t_ha=None))

    # ---- soil pH
    ph, ph_opt = inputs["ph"], prof["ph"]
    if ph < ph_opt - 0.7:
        lime = (ph_opt - ph) * LIME_T_PER_PH_UNIT
        recs.append(dict(category="soil_ph", severity="warning" if ph > 5.2 else "critical", title="Soil is too acidic",
                         detail=f"pH {ph:.1f} is below the {ph_opt:.1f} target. Apply roughly {lime:.1f} t/ha agricultural lime (confirm with a lime-requirement test) and add organic matter.",
                         expected_gain_t_ha=gain("ph")))
    elif ph > ph_opt + 0.8:
        recs.append(dict(category="soil_ph", severity="warning" if ph < 8.5 else "critical", title="Soil is too alkaline",
                         detail=f"pH {ph:.1f} is above the {ph_opt:.1f} target. Use gypsum or elemental sulphur (200-400 kg/ha), apply compost/green manure, and prefer acidifying fertilisers such as ammonium sulphate.",
                         expected_gain_t_ha=gain("ph")))
    else:
        recs.append(dict(category="soil_ph", severity="success", title="Soil pH is in range", detail=f"pH {ph:.1f} suits {crop} (target ~{ph_opt:.1f}). No correction needed.", expected_gain_t_ha=None))

    # ---- N / P / K
    supply, (n_opt, p_opt, k_opt) = _nutrient_supply(inputs), prof["npk"]
    spec = {"nitrogen": ("N", n_opt, UREA_N, "urea"), "phosphorus": ("P", p_opt, SSP_P2O5, "single super phosphate"), "potassium": ("K", k_opt, MOP_K2O, "muriate of potash")}
    excess_at = {"nitrogen": 1.9, "phosphorus": 3.5, "potassium": 2.4}  # P and K accumulate in soil, so tolerate more before warning
    bench = (benchmark or {}).get("top_quartile_median", {})
    for key, (sym, opt, conv, product) in spec.items():
        have = supply[key]
        if have < 0.8 * opt:
            need = opt - have
            recs.append(dict(category=key, severity="critical" if have < 0.5 * opt else "warning", title=f"Low {key} supply",
                             detail=f"Estimated {sym} supply ~{have:.0f} kg/ha vs ~{opt:.0f} kg/ha for {crop}. Add ~{need:.0f} kg/ha {sym} (~{need / conv:.0f} kg/ha of {product}), split into 2-3 doses." +
                                    (f" Top-quartile {crop} farms in the data applied ~{bench['fertilizer_kg_ha']:.0f} kg/ha total fertiliser." if key == "nitrogen" and "fertilizer_kg_ha" in bench else ""),
                             expected_gain_t_ha=gain("fertilizer")))
        elif have > excess_at[key] * opt:
            recs.append(dict(category=key, severity="warning", title=f"Excess {key} supply",
                             detail=f"{sym} supply (~{have:.0f} kg/ha) is far above the {opt:.0f} kg/ha need. Reduce {product} to save cost and avoid leaching and lodging.", expected_gain_t_ha=None))
        else:
            recs.append(dict(category=key, severity="success", title=f"{key.capitalize()} is sufficient", detail=f"{sym} supply (~{have:.0f} kg/ha) is close to the {opt:.0f} kg/ha target.", expected_gain_t_ha=None))

    # ---- climate / suitability
    t_opt, t_tol = prof["temp"]
    if inputs["temperature_c"] > t_opt + 7:
        recs.append(dict(category="climate", severity="warning", title="Heat stress risk",
                         detail=f"Mean temperature {inputs['temperature_c']:.1f} C is well above the {t_opt} C optimum. Consider earlier sowing, mulching, or a heat-tolerant variety; irrigate during flowering.", expected_gain_t_ha=None))
    elif inputs["temperature_c"] < t_opt - 7:
        recs.append(dict(category="climate", severity="warning", title="Cold stress risk",
                         detail=f"Mean temperature {inputs['temperature_c']:.1f} C is well below the {t_opt} C optimum. Delay sowing or choose a cold-tolerant variety.", expected_gain_t_ha=None))
    if inputs["season"] not in prof["seasons"]:
        recs.append(dict(category="crop", severity="critical", title=f"{crop} is not usually grown in {inputs['season']}",
                         detail=f"{crop} is typically a {'/'.join(prof['seasons'])} crop. Predictions outside its normal season are less reliable.", expected_gain_t_ha=None))
    if alternatives:
        cur = next((a for a in alternatives if a["crop"] == crop), None)
        top = [a for a in alternatives if a["crop"] != crop and a.get("season_suitable", True)][:2]
        if cur and top and top[0]["profit"] > cur["profit"] * 1.15 and top[0]["profit"] > 0:
            names = ", ".join(f"{a['crop']} (~INR {a['profit']:,.0f})" for a in top)
            recs.append(dict(category="crop", severity="info", title="A more profitable crop may exist",
                             detail=f"Under the same conditions, {names} are projected to out-earn {crop} (~INR {cur['profit']:,.0f}). Compare crops for details.", expected_gain_t_ha=None))
        elif cur and top:
            recs.append(dict(category="crop", severity="success", title=f"{crop} is a strong choice here", detail="It ranks among the best options for these conditions by projected profit.", expected_gain_t_ha=None))

    # ---- economics
    profit = pred.get("profit")
    if profit is not None:
        if profit < 0:
            recs.append(dict(category="economics", severity="critical", title="Projected loss",
                             detail=f"Estimated profit is INR {profit:,.0f}. Even the optimistic estimate ({pred['upper']:.2f} t/ha) is tight - reduce input costs or choose a different crop.", expected_gain_t_ha=None))
        else:
            recs.append(dict(category="economics", severity="info", title="Profit outlook",
                             detail=f"Estimated profit INR {profit:,.0f} ({pred.get('roi_pct', 0):.0f}% return on cost). A 10% fall in crop price would cut profit by about INR {0.1 * pred.get('revenue', 0):,.0f}.", expected_gain_t_ha=None))
    order = {"critical": 0, "warning": 1, "info": 2, "success": 3}
    return sorted(recs, key=lambda r: order[r["severity"]])
