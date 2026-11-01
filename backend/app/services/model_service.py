"""Loads the versioned model bundle and serves point predictions, intervals, SHAP and what-if levers."""
import json
import logging
import re
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from ml.src import explain
from ml.src.config import CROP_PROFILE, ECONOMICS, IRRIGATION, RAW_FEATURES
from ml.src.uncertainty import interval

from ..core.config import get_settings
from . import recommendations as rec

log = logging.getLogger("model")

LABELS = {"crop": "Crop type", "region": "Region", "season": "Season", "soil_type": "Soil type", "irrigation": "Irrigation method",
          "ph": "Soil pH", "nitrogen": "Soil nitrogen", "phosphorus": "Soil phosphorus", "potassium": "Soil potassium",
          "rainfall_mm": "Rainfall", "temperature_c": "Temperature", "humidity_pct": "Humidity", "area_ha": "Field area",
          "fertilizer_kg_ha": "Fertiliser applied", "year": "Year / technology trend",
          "rain_temp_interaction": "Rainfall x temperature", "total_water_mm": "Total water (rain + irrigation)",
          "aridity_index": "Aridity index", "heat_humidity_index": "Heat-humidity index", "n_p_ratio": "N:P ratio",
          "n_k_ratio": "N:K ratio", "p_k_ratio": "P:K ratio", "npk_total": "Total NPK + fertiliser",
          "fert_per_soil_n": "Fertiliser vs soil N", "ph_deviation": "pH deviation from 6.8",
          "season_ord": "Season code", "log_area": "Field area (log)", "years_since_2000": "Years since 2000"}


class ModelService:
    def __init__(self, model_dir: str | None = None, version: str | None = None):
        s = get_settings()
        self.dir = Path(model_dir or s.model_dir)
        want = version or s.model_version
        files = {int(m.group(1)): f for f in self.dir.glob("model_v*.joblib") if (m := re.match(r"model_v(\d+)\.joblib", f.name))}
        if not files:
            raise FileNotFoundError(f"No model_v*.joblib in {self.dir}. Run `python -m ml.src.train` first.")
        v = max(files) if want == "latest" else int(want)
        self.version = v
        self.bundle = joblib.load(files[v])
        self.meta = json.loads((self.dir / f"model_v{v}.json").read_text())
        self.explainer = shap.TreeExplainer(self.bundle["explainer_pipeline"][-1])
        self._lock = threading.Lock()
        log.info("Loaded model v%s (%s), explainer=%s", v, self.bundle["model_name"], self.bundle["explainer_name"])

    # ---- core prediction
    @staticmethod
    def frame(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)[RAW_FEATURES]

    def predict_frame(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b = self.bundle
        with self._lock:
            point = np.clip(b["pipeline"].predict(df), 0, None)
        lo, hi = interval(point, df, b["q_low"], b["q_high"], b["qhat"])
        return point, lo, hi

    def economics(self, crop: str, y: float, area: float) -> dict:
        e = ECONOMICS[crop]
        revenue, cost = y * area * e["price_per_tonne"], e["cost_per_ha"] * area
        return {"price_per_tonne": e["price_per_tonne"], "revenue": round(revenue, 2), "cost": round(cost, 2),
                "profit": round(revenue - cost, 2), "roi_pct": round((revenue - cost) / cost * 100, 1) if cost else 0.0}

    def explain_row(self, row: pd.DataFrame, top: int = 5) -> tuple[float, list[dict]]:
        with self._lock:
            base, contrib = explain.local_explanation(self.explainer, self.bundle["explainer_pipeline"], row)
            engineered = self.bundle["explainer_pipeline"][0].transform(row).iloc[0]
        items = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:top]
        out = []
        for feat, val in items:
            v = engineered[feat]
            out.append({"feature": feat, "label": LABELS.get(feat, feat), "value": v if isinstance(v, str) else round(float(v), 2),
                        "shap": round(float(val), 4), "direction": "increases" if val >= 0 else "decreases"})
        return round(base, 4), out

    # ---- levers: ask the model what each candidate improvement is worth
    def levers(self, inputs: dict, crop: str, base_yield: float) -> dict:
        variants, keys = [], []
        row = {**inputs, "crop": crop}

        def add(key, **chg):
            variants.append({**row, **chg}); keys.append(key)

        best_irr = max(IRRIGATION, key=lambda i: 0 if i == "Rainfed" else rec.IRRIGATION_MM[i])
        add("irrigation", irrigation=best_irr)
        add("fertilizer", fertilizer_kg_ha=min(row["fertilizer_kg_ha"] * 1.3 + 25, 300))
        add("ph", ph=CROP_PROFILE[crop]["ph"])
        preds, _, _ = self.predict_frame(self.frame(variants))
        best = {}
        for k, p in zip(keys, preds, strict=False):
            best[k] = max(best.get(k, -9), float(p) - base_yield)
        return best

    # ---- public high-level operations
    def full_prediction(self, inputs: dict, crop: str, with_alternatives: bool = True) -> dict:
        row = {**inputs, "crop": crop}
        df = self.frame([row])
        p, lo, hi = (float(x[0]) for x in self.predict_frame(df))
        area = inputs["area_ha"]
        eco = self.economics(crop, p, area)
        base, factors = self.explain_row(df)
        alternatives = self.compare(inputs, None)["results"] if with_alternatives else None
        levers = self.levers(inputs, crop, p)
        recs = rec.generate(inputs, crop, {"yield": p, "lower": lo, "upper": hi, **eco}, levers, alternatives,
                            self.meta["benchmarks"].get(crop))
        return {"crop": crop, "inputs": row, "predicted_yield_t_ha": round(p, 3), "lower_t_ha": round(lo, 3), "upper_t_ha": round(hi, 3),
                "confidence_level": 0.8, "total_production_t": round(p * area, 2), "economics": eco, "base_value_t_ha": base,
                "top_factors": factors, "recommendations": recs, "model_version": self.version}

    def compare(self, inputs: dict, crops: list[str] | None) -> dict:
        crops = crops or list(CROP_PROFILE)
        df = self.frame([{**inputs, "crop": c} for c in crops])
        p, lo, hi = self.predict_frame(df)
        out = []
        for c, a, b, d in zip(crops, p, lo, hi, strict=False):
            e = self.economics(c, float(a), inputs["area_ha"])
            out.append({"crop": c, "yield_t_ha": round(float(a), 3), "lower_t_ha": round(float(b), 3), "upper_t_ha": round(float(d), 3),
                        "total_production_t": round(float(a) * inputs["area_ha"], 2), "season_suitable": inputs["season"] in CROP_PROFILE[c]["seasons"],
                        **{k: e[k] for k in ("revenue", "cost", "profit", "roi_pct")}})
        # Crops outside their normal season are listed but ranked after suitable ones.
        out.sort(key=lambda r: (not r["season_suitable"], -r["profit"]))
        return {"results": out, "best_crop": next((r["crop"] for r in out if r["season_suitable"]), out[0]["crop"])}

    def what_if(self, base_in: dict, crop: str, changes: dict, sweep: dict | None) -> dict:
        base = {k: v for k, v in base_in.items() if k != "crop"}
        scen = {**base, **{k: v for k, v in changes.items() if k in base}}
        df = self.frame([{**base, "crop": crop}, {**scen, "crop": crop}])
        p, lo, hi = self.predict_frame(df)
        area = base["area_ha"]
        sc_area = scen["area_ha"]
        res = {"baseline": {"yield_t_ha": round(float(p[0]), 3), "lower_t_ha": round(float(lo[0]), 3), "upper_t_ha": round(float(hi[0]), 3),
                            **self.economics(crop, float(p[0]), area)},
               "scenario": {"yield_t_ha": round(float(p[1]), 3), "lower_t_ha": round(float(lo[1]), 3), "upper_t_ha": round(float(hi[1]), 3),
                            **self.economics(crop, float(p[1]), sc_area), "inputs": scen}}
        res["delta_yield_t_ha"] = round(float(p[1] - p[0]), 3)
        res["delta_pct"] = round(float((p[1] - p[0]) / max(p[0], 1e-6) * 100), 2)
        res["delta_profit"] = round(res["scenario"]["profit"] - res["baseline"]["profit"], 2)
        _, res["scenario"]["top_factors"] = self.explain_row(self.frame([{**scen, "crop": crop}]), top=5)
        if sweep:
            xs = np.linspace(sweep["min"], sweep["max"], sweep["steps"])
            rows = [{**scen, sweep["param"]: float(x), "crop": crop} for x in xs]
            sp, slo, shi = self.predict_frame(self.frame(rows))
            res["sweep"] = {"param": sweep["param"], "points": [{"x": round(float(x), 3), "yield_t_ha": round(float(a), 3), "lower_t_ha": round(float(b), 3), "upper_t_ha": round(float(c), 3)}
                                                            for x, a, b, c in zip(xs, sp, slo, shi, strict=False)]}
        return res


_service: ModelService | None = None


def get_model_service() -> ModelService:
    global _service
    if _service is None:
        _service = ModelService()
    return _service


def reset_model_service() -> None:
    global _service
    _service = None
