"""Loads the versioned model bundle and serves point predictions, intervals, SHAP and what-if levers."""
import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from ml.src import explain
from ml.src.config import CROP_PROFILE, CROPS, ECONOMICS, IRRIGATION, RAW_FEATURES, REGIONS, SEASONS, SOILS
from ml.src.data_cleaning import normalize_district
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
          "season_ord": "Season code", "log_area": "Field area (log)", "years_since_2000": "Years since 2000",
          "soil_ph": "Soil pH (district estimate)", "soil_organic_carbon": "Soil organic carbon (district estimate)", "soil_sand_pct": "Soil sand % (district estimate)",
          "soil_clay_pct": "Soil clay % (district estimate)", "temp_max_c": "Mean daily maximum temperature", "latitude": "Latitude (district)", "longitude": "Longitude (district)",
          "heat_stress_c": "Heat stress (above 35 C)", "soil_texture_ratio": "Sand : clay ratio"}
SYNTH_CATEGORIES = {"crop": CROPS, "region": REGIONS, "season": SEASONS, "soil_type": SOILS, "irrigation": IRRIGATION}
CAT_FIELDS = ["crop", "region", "season", "soil_type", "irrigation"]


class InputError(Exception):
    """Invalid request content that Pydantic cannot judge because it depends on the ACTIVE model (categories, required inputs)."""

    def __init__(self, errors: list[dict]):
        super().__init__("; ".join(e["message"] for e in errors))
        self.errors = errors


class ModelService:
    def __init__(self, model_dir: str | None = None, version: str | None = None):
        """version: 'latest' (highest model_vN, synthetic), N, 'real_latest', or 'real_district_vN'."""
        s = get_settings()
        self.dir = Path(model_dir or s.model_dir)
        stem = self._resolve_stem(str(version or s.model_version))
        self.bundle = joblib.load(self.dir / f"{stem}.joblib")
        self.meta = json.loads((self.dir / f"{stem}.json").read_text())
        b = self.bundle
        self.name = b.get("name", stem)
        self.version = int(b["version"])
        self.data_mode = b.get("data_mode", "synthetic")
        self.raw_features = list(b.get("raw_features", RAW_FEATURES))
        self.categorical = b.get("categorical")  # None -> synthetic default inside ml.src.explain
        schema = self.meta.get("input_schema")
        if schema:
            self.required, self.used = list(schema["required"]), list(schema["used"])
        else:  # synthetic models: every field except `year` (which defaults to the current year) is required
            self.required, self.used = [f for f in RAW_FEATURES if f != "year"], list(RAW_FEATURES)
        self.explainer = shap.TreeExplainer(b["explainer_pipeline"][-1])
        self._lock = threading.Lock()
        log.info("Loaded %s [%s] (%s), explainer=%s", self.name, self.data_mode, b["model_name"], b["explainer_name"])

    def _resolve_stem(self, want: str) -> str:
        def newest(prefix: str) -> str:
            nums = {int(m.group(1)) for f in self.dir.glob(f"{prefix}*.joblib") if (m := re.match(rf"{prefix}(\d+)\.joblib", f.name))}
            if not nums:
                raise FileNotFoundError(f"No {prefix}*.joblib in {self.dir}. Train a model first (python -m ml.src.train [--data real]).")
            return f"{prefix}{max(nums)}"
        if want == "latest":
            return newest("model_v")
        if want == "real_latest":
            return newest("real_district_v")
        if re.fullmatch(r"real_district_v\d+", want):
            return want
        if want.isdigit():
            return f"model_v{int(want)}"
        raise ValueError(f"MODEL_VERSION={want!r}: use 'latest', a number, 'real_latest' or 'real_district_vN'")

    # ---- input handling (depends on the active model)
    def categories(self, field: str) -> list[str]:
        cats = self.meta.get("categories")
        if cats is not None:  # the model's own categories; a field it does not use (e.g. irrigation in a district model) is simply empty
            return list(cats.get(field, []))
        return list(SYNTH_CATEGORIES.get(field, []))  # minimal/test metadata without a category table

    def canonical(self, field: str, value):
        for c in self.categories(field):
            if c.lower() == str(value).lower():
                return c
        return None

    def prepare(self, inputs: dict, need_crop: bool = True) -> tuple[dict, dict]:
        """Validate against THIS model, fill defaults, resolve district-level attributes. Returns (row, notes). Idempotent."""
        row = {k: v for k, v in inputs.items() if v is not None}
        errs: list[dict] = []
        for f in CAT_FIELDS:
            if f in row and (f in self.used or f == "crop"):
                c = self.canonical(f, row[f])
                if c is None:
                    errs.append({"field": f, "message": f"unknown {f} '{row[f]}'. Allowed: {', '.join(self.categories(f))}"})
                else:
                    row[f] = c
        for f in self.required:
            if f not in row and (f != "crop" or need_crop):
                errs.append({"field": f, "message": f"'{f}' is required by the active model ({self.name})"})
        if errs:
            raise InputError(errs)
        notes = {"estimated_inputs": [], "data_notes": [], "district": None}
        if "year" in self.used:
            row.setdefault("year", datetime.now().year)
        if "area_ha" not in row:
            row["area_ha"] = 1.0
            if self.data_mode == "real":
                notes["data_notes"].append("Field area not given: totals and profit are shown per 1 ha.")
        if self.data_mode == "real":
            self._resolve_district(row, notes)
        self._range_warnings(row, notes)
        return row, notes

    def _range_warnings(self, row: dict, notes: dict) -> None:
        """Tree models cannot extrapolate: flag values the user typed that lie outside what the model saw for THIS crop and season
        (falls back to the pooled range when the group is small or unknown)."""
        label = {"rainfall_mm": "Rainfall", "temperature_c": "Temperature", "humidity_pct": "Humidity", "ph": "Soil pH", "soil_ph": "Soil pH",
                 "nitrogen": "Nitrogen", "phosphorus": "Phosphorus", "potassium": "Potassium", "fertilizer_kg_ha": "Fertiliser"}
        est = set(notes.get("estimated_inputs", []))
        grp = (self.meta.get("feature_ranges_by_group") or {}).get(f"{row.get('crop')}|{row.get('season')}", {})
        for f, pooled in (self.meta.get("feature_ranges") or {}).items():
            v = row.get(f)
            if f not in label or v is None or f in est:
                continue
            r, where = (grp[f], f" for {row['crop']} in {row['season']}") if f in grp else (pooled, "")
            if v < r["min"] or v > r["max"]:
                notes["data_notes"].append(f"{label[f]} {v:g} is outside the range the model was trained on{where} ({r['min']:.0f} to {r['max']:.0f}): treat this estimate with extra caution.")

    def _resolve_district(self, row: dict, notes: dict) -> None:
        static = [c for c in ("soil_ph", "soil_organic_carbon", "soil_sand_pct", "soil_clay_pct", "latitude", "longitude") if c in self.raw_features]
        dtab, rtab = self.bundle.get("district_table", {}), self.bundle.get("region_table", {})
        region, dname = row["region"], normalize_district(row.get("district")) if row.get("district") else None
        src, label = None, None
        if dname:
            src = dtab.get(f"{region}|{dname}")
            if src is None:
                ex = ", ".join(list(self.meta.get("districts", {}).get(region, []))[:5])
                raise InputError([{"field": "district", "message": f"district '{row['district']}' not found for {region} in the training data. Examples: {ex}"}])
            label = f"{dname.title()} district"
        elif row.get("latitude") is not None and row.get("longitude") is not None:
            best = None
            for key, v in dtab.items():
                if key.startswith(f"{region}|") and v.get("latitude") is not None:
                    d = (v["latitude"] - row["latitude"]) ** 2 + ((v["longitude"] - row["longitude"]) * 0.95) ** 2
                    if best is None or d < best[0]:
                        best = (d, key, v)
            if best:
                src, label, dname = best[2], f"nearest district ({best[1].split('|')[1].title()})", best[1].split("|")[1]
        if src is None:
            src, label = rtab.get(region, {}), f"{region} state median"
        used_est = []
        for c in static:
            if row.get(c) is None and src.get(c) is not None:
                row[c] = src[c]
                used_est.append(c)
        notes["district"] = dname
        notes["estimated_inputs"] = used_est
        if used_est:
            notes["data_notes"].append(f"Soil and location values come from {label}: a regional estimate, not a measurement of your field.")
        notes["data_notes"].append("This model predicts a district-level average yield from government statistics; an individual farm can differ.")

    def canonical_crops(self, crops: list[str] | None) -> list[str] | None:
        if crops is None:
            return None
        out, errs = [], []
        for c in crops:
            k = self.canonical("crop", c)
            (out if k else errs).append(k or {"field": "crops", "message": f"unknown crop '{c}'. Allowed: {', '.join(self.categories('crop'))}"})
        if errs:
            raise InputError(errs)
        return out

    def describe(self) -> dict:
        """What the UI needs to build the right form for this model."""
        return {"data_mode": self.data_mode, "model_name": self.name, "input_schema": {"required": self.required, "used": self.used,
                "optional": [f for f in self.used if f not in self.required]}, "provenance": self.meta.get("provenance"),
                "districts": self.meta.get("districts") if self.data_mode == "real" else None}

    # ---- core prediction
    def frame(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        for c in self.raw_features:
            if c not in df.columns:
                df[c] = np.nan  # optional numeric input not supplied: the pipeline imputes it (per-crop median)
        return df[self.raw_features]

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
            base, contrib = explain.local_explanation(self.explainer, self.bundle["explainer_pipeline"], row, self.categorical)
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
        if "irrigation" not in self.raw_features:  # district-level models cannot simulate farm-level irrigation / fertiliser / pH changes
            return {}
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
        row, notes = self.prepare({**inputs, "crop": crop})
        crop = row["crop"]
        inputs = {k: v for k, v in row.items() if k != "crop"}
        df = self.frame([row])
        p, lo, hi = (float(x[0]) for x in self.predict_frame(df))
        area = inputs["area_ha"]
        eco = self.economics(crop, p, area)
        base, factors = self.explain_row(df)
        alternatives = self.compare(inputs, None)["results"] if with_alternatives else None
        levers = self.levers(inputs, crop, p)
        recs = rec.generate(inputs, crop, {"yield": p, "lower": lo, "upper": hi, **eco}, levers, alternatives,
                            self.meta["benchmarks"].get(crop), estimated=notes["estimated_inputs"],
                            real_ref=(self.meta.get("feature_ranges_by_group") or {}).get(f"{crop}|{row['season']}") if self.data_mode == "real" else None)
        return {"model_name": self.name, "data_mode": self.data_mode, "estimated_inputs": notes["estimated_inputs"], "data_notes": notes["data_notes"],
                "crop": crop, "inputs": row, "predicted_yield_t_ha": round(p, 3), "lower_t_ha": round(lo, 3), "upper_t_ha": round(hi, 3),
                "confidence_level": 0.8, "total_production_t": round(p * area, 2), "economics": eco, "base_value_t_ha": base,
                "top_factors": factors, "recommendations": recs, "model_version": self.version}

    def compare(self, inputs: dict, crops: list[str] | None) -> dict:
        inputs, _ = self.prepare(inputs, need_crop=False)
        crops = self.canonical_crops(crops) or [c for c in CROP_PROFILE if c in self.categories("crop")]
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
        base, _ = self.prepare({**base_in, "crop": crop})
        crop = base.pop("crop")
        scen, _ = self.prepare({**base, **{k: v for k, v in changes.items() if k in base or k in self.raw_features}, "crop": crop})
        scen.pop("crop")
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