"""TEST FIXTURES ONLY. These numbers are fabricated to exercise code paths. They are NOT agricultural observations
and must never be written to data/real/."""
import numpy as np
import pandas as pd


def kaggle_style_raw() -> pd.DataFrame:
    """Mimics the Kaggle/data.gov.in crop_production.csv layout, including its quirks (padded season, UPPERCASE districts, NaN)."""
    rows = [
        # State_Name,           District_Name,  Crop_Year, Season,        Crop,         Area,   Production
        ("Karnataka",           "UTTARA KANNADA", 2010, "Kharif     ", "Rice",         1000.0, 3000.0),   # 3.0 t/ha
        ("Karnataka",           "UTTARA KANNADA", 2011, "Kharif     ", "Paddy",        1200.0, 3000.0),   # alias of Rice, 2.5 t/ha
        ("Orissa",              "Khordha district", 2010, "Rabi       ", "Wheat",        500.0,  900.0),   # state alias, 'district' suffix
        ("Karnataka",           "MYSORE",         2010, "Kharif     ", "Ragi",         800.0,  1600.0),   # district alias -> mysuru
        ("Karnataka",           "MYSORE",         2010, "Whole Year ", "Rice",         800.0,  1600.0),   # season dropped
        ("Karnataka",           "UTTARA KANNADA", 2012, "Kharif     ", "Rice",         0.0,    10.0),     # zero area
        ("Karnataka",           "UTTARA KANNADA", 2013, "Kharif     ", "Rice",         900.0,  np.nan),   # missing production
        ("Karnataka",           "UTTARA KANNADA", 2014, "Kharif     ", "Rice",         900.0,  0.0),      # zero production
        ("Karnataka",           "UTTARA KANNADA", 2015, "Kharif     ", "Rice",         900.0,  90000.0),  # 100 t/ha: impossible
        ("Karnataka",           "UTTARA KANNADA", 2010, "Kharif     ", "Coconut",      900.0,  900.0),    # unsupported crop
        ("Karnataka",           "UTTARA KANNADA", 2010, "Kharif     ", "Cotton(lint)", 900.0,  900.0),    # needs bale unit info
        ("Karnataka",           "UTTARA KANNADA", 2010, "Kharif     ", "Groundnut",    20.0,   30.0),     # below min area
        ("Karnataka",           "DHARWAD",        2010, "Kharif     ", "Maize",        400.0,  1000.0),
        ("Karnataka",           "DHARWAD",        2010, "Kharif     ", "Maize",        600.0,  1500.0),   # duplicate key -> summed: 1000 ha, 2500 t
        ("Karnataka",           "DHARWAD",        "2009-10", "Rabi     ", "Sorghum",     300.0,  300.0),    # crop-year string -> 2009
    ]
    return pd.DataFrame(rows, columns=["State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area", "Production"])


def india_portal_raw() -> pd.DataFrame:
    """Mimics the India Data Portal APY export that carries explicit unit columns."""
    return pd.DataFrame([
        {"year": "2015-16", "state_name": "Maharashtra", "district_name": "Nagpur", "crop_name": "Cotton(lint)", "season": "Kharif",
         "area": 100.0, "area_unit": "Thousand Hectares", "production": 250000.0, "production_unit": "Bales", "yield": 425.0, "yield_unit": "Kg/Hectare"},
        {"year": "2015-16", "state_name": "Maharashtra", "district_name": "Nagpur", "crop_name": "Rice", "season": "Kharif",
         "area": 50.0, "area_unit": "Thousand Hectares", "production": 100.0, "production_unit": "Thousand Tonnes", "yield": 2000.0, "yield_unit": "Kg/Hectare"},
        {"year": "2015-16", "state_name": "Maharashtra", "district_name": "Nagpur", "crop_name": "Maize", "season": "Kharif",
         "area": 5000.0, "area_unit": "Hectare", "production": 9000.0, "production_unit": "Furlongs", "yield": 1800.0, "yield_unit": "Kg/Hectare"},  # unknown unit
    ])


def processed_fixture(n_districts: int = 24, years=range(2005, 2021), seed: int = 0) -> pd.DataFrame:
    """A merged, processed-format frame with a known weather/soil -> yield relationship, to test training mechanics only."""
    rng = np.random.default_rng(seed)
    states = ["Karnataka", "Maharashtra", "Punjab", "Odisha"]
    rows = []
    base = {"Rice": 2.6, "Wheat": 3.0, "Maize": 2.8}
    dist = [(states[i % 4], f"district{i}", rng.uniform(10, 30), rng.uniform(72, 85), rng.normal(6.6, 0.5), rng.uniform(5, 20), rng.uniform(20, 60), rng.uniform(15, 45)) for i in range(n_districts)]
    for st, d, lat, lon, ph, oc, sand, clay in dist:
        for crop, b in base.items():
            season = "Rabi" if crop == "Wheat" else "Kharif"
            for y in years:
                rain = max(50, rng.normal(900 if season == "Kharif" else 150, 250)); t = rng.normal(27 if season == "Kharif" else 20, 1.5)
                yld = b * (1 + 0.012 * (y - 2005)) * (0.6 + 0.4 * np.exp(-((rain - 900) / 600) ** 2 if season == "Kharif" else -((rain - 150) / 200) ** 2)) \
                    * (0.8 + 0.2 * np.exp(-((ph - 6.5) / 1.2) ** 2)) * rng.lognormal(0, 0.07)
                rows.append(dict(region=st, district=d, crop=crop, season=season, year=y, yield_t_ha=yld, area_ha=rng.uniform(500, 5000), production_t=0.0,
                                 rainfall_mm=rain, temperature_c=t, temp_max_c=t + 6, humidity_pct=rng.uniform(40, 85), soil_ph=ph, soil_organic_carbon=oc,
                                 soil_sand_pct=sand, soil_clay_pct=clay, latitude=lat, longitude=lon, weather_source="FIXTURE", soil_source="FIXTURE",
                                 yield_source="FIXTURE"))
    df = pd.DataFrame(rows)
    df["production_t"] = df["yield_t_ha"] * df["area_ha"]
    return df