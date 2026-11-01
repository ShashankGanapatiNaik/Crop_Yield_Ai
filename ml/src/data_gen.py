"""Synthetic crop-yield dataset generator.

No public dataset was present in /data, so we generate a realistic one. Yield is produced by an
agronomic response model (Gaussian optimum curves for temperature, water and pH, Mitscherlich
nutrient response with a Liebig-style limiting term, soil suitability, irrigation, yearly
technology trend and heat stress) plus multiplicative noise. We then inject realistic data
defects: ~3% missing values and ~0.6% gross outliers, so the cleaning steps are meaningful.

Drop a real CSV with the same columns into /data/crop_yield.csv and train.py will use it instead.
"""
import numpy as np
import pandas as pd

from .config import (
    CROP_PROFILE,
    CROPS,
    DATA_DIR,
    IRRIGATION,
    IRRIGATION_MM,
    RANDOM_STATE,
    REGION_CLIMATE,
    REGIONS,
    SOIL_FACTOR,
    SOILS,
    TARGET,
)

SEASON_RAIN = {"Kharif": 1.0, "Rabi": 0.32, "Summer": 0.28}
SEASON_TEMP = {"Kharif": 27.0, "Rabi": 20.0, "Summer": 30.5}


def _gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


def _mitscherlich(supply, opt):
    term = 1 - np.exp(-2.5 * supply / opt)
    term = term / 0.918
    excess = np.clip(supply / opt - 1.8, 0, None)
    return np.clip(term - 0.12 * excess, 0.05, 1.04)


def generate(n: int = 9000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    crops = rng.choice(CROPS, n)
    for crop in crops:
        p = CROP_PROFILE[crop]
        region = rng.choice(REGIONS)
        season = rng.choice(p["seasons"])
        soil = rng.choice(SOILS, p=[.22, .2, .2, .1, .08, .2])
        irr = rng.choice(IRRIGATION, p=[.3, .22, .22, .12, .14])
        rain_mult, temp_off = REGION_CLIMATE[region]
        year = int(rng.integers(2000, 2024))

        rain = rng.lognormal(np.log(SEASON_RAIN[season] * 900 * rain_mult * (p["rain"][0] / 700) ** 0.5), 0.35)
        temp = rng.normal(SEASON_TEMP[season] + temp_off + 0.04 * (year - 2000), 2.0)
        hum = np.clip(35 + 0.035 * rain + rng.normal(0, 8) + (8 if season == "Kharif" else 0), 20, 98)
        ph = np.clip(rng.normal(6.7, 0.8), 4.3, 9.0)
        n_soil, p_soil, k_soil = np.clip(rng.normal([105, 28, 125], [35, 12, 40]), 15, None)
        fert = np.clip(rng.normal(95, 55), 0, 300)
        area = np.clip(rng.lognormal(0.7, 0.8), 0.2, 60)

        # --- response model ---
        water = rain + IRRIGATION_MM[irr] * 0.9
        w_opt, w_tol = p["rain"]
        f_water = 0.3 + 0.7 * _gauss(water, w_opt, w_tol * (1.0 if water < w_opt else 2.4))
        f_temp = 0.4 + 0.6 * _gauss(temp, p["temp"][0], p["temp"][1] * 1.4)
        heat = max(0.0, temp - (p["temp"][0] + 7))
        f_heat = max(0.5, 1 - 0.05 * heat)
        f_ph = 0.7 + 0.3 * _gauss(ph, p["ph"], 0.9)
        on, op, ok = p["npk"]
        sn, sp, sk = n_soil + 0.5 * fert, p_soil * 1.5 + 0.28 * fert, k_soil + 0.22 * fert
        terms = np.array([_mitscherlich(sn * 1.2, on), _mitscherlich(sp * 2.0, op), _mitscherlich(sk * 1.2, ok)])
        f_nut = 0.5 * terms.min() + 0.5 * terms.mean()
        f_soil = SOIL_FACTOR[crop][soil]
        f_trend = 1 + 0.011 * (year - 2000)
        f_area = 1 - 0.015 * np.log(area)
        f_hum = 1 - 0.0025 * max(0, hum - 85) ** 1.2 if crop in ("Wheat", "Chickpea", "Groundnut") else 1.0
        noise = rng.lognormal(0, 0.08)
        y = p["base"] * f_water * f_temp * f_heat * f_ph * f_nut * f_soil * f_trend * f_area * f_hum * noise
        rows.append([crop, region, season, soil, irr, ph, n_soil, p_soil, k_soil, rain, temp, hum, area, fert, year, y])

    df = pd.DataFrame(rows, columns=["crop", "region", "season", "soil_type", "irrigation", "ph", "nitrogen",
                                     "phosphorus", "potassium", "rainfall_mm", "temperature_c", "humidity_pct",
                                     "area_ha", "fertilizer_kg_ha", "year", TARGET])
    # Defects: gross outliers in target and missing values.
    out_idx = rng.choice(n, int(0.006 * n), replace=False)
    df.loc[out_idx, TARGET] *= rng.choice([0.1, 3.5], len(out_idx))
    for col in ["ph", "nitrogen", "phosphorus", "potassium", "humidity_pct", "fertilizer_kg_ha", "temperature_c"]:
        df.loc[rng.choice(n, int(0.03 * n), replace=False), col] = np.nan
    return df.round(3)


def load_or_generate() -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source_description)."""
    real = DATA_DIR / "crop_yield.csv"
    if real.exists():
        return pd.read_csv(real), f"real dataset: {real.name}"
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "crop_yield_synthetic.csv"
    if out.exists():
        return pd.read_csv(out), "synthetic dataset (agronomic response simulator)"
    df = generate()
    df.to_csv(out, index=False)
    return df, "synthetic dataset (agronomic response simulator)"


if __name__ == "__main__":
    d, src = load_or_generate()
    print(src, d.shape)
    print(d.describe().T)
