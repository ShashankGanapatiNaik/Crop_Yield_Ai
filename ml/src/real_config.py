"""Configuration and domain knowledge for the REAL district-level data pipeline.

Nothing here invents observations: it holds name mappings, unit conversions, plausibility bounds and the
feature schema. Tunables can be overridden in ml/real_data.toml (or the file named by CROPAI_REAL_CONFIG).
"""
import copy
import os
import tomllib
from pathlib import Path

from .config import DATA_DIR, ROOT

REAL_DIR = DATA_DIR / "real"
RAW_DIR = REAL_DIR / "raw"
PROCESSED_DIR = REAL_DIR / "processed"
PROCESSED_CSV = PROCESSED_DIR / "district_yield.csv"
PROCESSED_META = PROCESSED_DIR / "district_yield.meta.json"
CONFIG_FILE = ROOT / "ml" / "real_data.toml"

DEFAULTS = {
    "apy": {"states": [], "seasons": ["Kharif", "Rabi", "Summer"], "min_year": 1997, "max_year": 2022, "min_area_ha": 100.0, "drop_zero_production": True},
    "cleaning": {"unit_assumptions": {}},
    "weather": {"provider": "nasa_power", "min_season_coverage": 0.9, "request_sleep_s": 1.0},
    "soil": {"source": "soilgrids", "request_sleep_s": 13.0, "require_soil": True},
    "split": {"test_years": 3, "val_years": 2, "min_rows_per_group_report": 30},
    "train": {"final_fit": "train_val"},
}


def load_config(path: Path | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path or os.environ.get("CROPAI_REAL_CONFIG", CONFIG_FILE))
    if p.exists():
        user = tomllib.loads(p.read_text())
        for section, vals in user.items():
            cfg.setdefault(section, {}).update(vals)
    return cfg


# ---------------------------------------------------------------- name normalisation (lower-case keys)
CROP_ALIASES = {
    "rice": "Rice", "paddy": "Rice", "wheat": "Wheat", "maize": "Maize", "corn": "Maize",
    "cotton(lint)": "Cotton", "cotton": "Cotton", "cotton lint": "Cotton",
    "groundnut": "Groundnut", "ground nut": "Groundnut",
    "ragi": "Ragi", "finger millet": "Ragi", "ragi (finger millet)": "Ragi",
    "soyabean": "Soybean", "soybean": "Soybean", "soya bean": "Soybean",
    "gram": "Chickpea", "chickpea": "Chickpea", "bengal gram": "Chickpea", "chana": "Chickpea",
    "arhar/tur": "Pigeonpea", "tur": "Pigeonpea", "arhar": "Pigeonpea", "pigeonpea": "Pigeonpea", "pigeon pea": "Pigeonpea",
    "jowar": "Sorghum", "sorghum": "Sorghum",
}
STATE_ALIASES = {
    "orissa": "Odisha", "uttaranchal": "Uttarakhand", "pondicherry": "Puducherry", "chattisgarh": "Chhattisgarh",
    "nct of delhi": "Delhi", "tamilnadu": "Tamil Nadu", "jammu & kashmir": "Jammu and Kashmir", "andaman & nicobar islands": "Andaman and Nicobar Islands",
    "dadra & nagar haveli": "Dadra and Nagar Haveli", "a & n islands": "Andaman and Nicobar Islands", "west bengal.": "West Bengal",
}
DISTRICT_ALIASES = {  # normalised-key -> normalised-key. Extend via data/real/raw/district_aliases.csv (columns: state,from,to)
    "gurgaon": "gurugram", "bangalore rural": "bengaluru rural", "bangalore urban": "bengaluru urban", "belgaum": "belagavi", "bijapur": "vijayapura",
    "gulbarga": "kalaburagi", "mysore": "mysuru", "shimoga": "shivamogga", "tumkur": "tumakuru", "bellary": "ballari", "chikmagalur": "chikkamagaluru",
    "allahabad": "prayagraj", "faizabad": "ayodhya",
}
SEASON_ALIASES = {"kharif": "Kharif", "rabi": "Rabi", "summer": "Summer", "whole year": "Whole Year", "autumn": "Autumn", "winter": "Winter", "total": "Whole Year"}

# ---------------------------------------------------------------- units -> hectares / tonnes
AREA_UNIT_TO_HA = {"hectare": 1.0, "hectares": 1.0, "ha": 1.0, "000 hectares": 1000.0, "thousand hectares": 1000.0, "'000 hectare": 1000.0,
                   "lakh hectares": 100000.0, "acre": 0.404686, "acres": 0.404686}
PROD_UNIT_TO_T = {"tonne": 1.0, "tonnes": 1.0, "ton": 1.0, "tons": 1.0, "t": 1.0, "mt": 1.0, "000 tonnes": 1000.0, "thousand tonnes": 1000.0,
                  "'000 tonnes": 1000.0, "lakh tonnes": 100000.0, "quintal": 0.1, "quintals": 0.1, "kg": 0.001, "kilogram": 0.001}
# Crop-specific units that need a per-crop factor (tonnes per unit). Only applied when the unit is EXPLICIT in the file.
CROP_PROD_UNIT_TO_T = {("Cotton", "bales"): 0.17, ("Cotton", "bale"): 0.17}  # APY cotton bale = 170 kg

# Hard physical plausibility bounds on district-average yield (t/ha). Outside = data error, not "bad year".
YIELD_BOUNDS = {"Rice": (0.1, 10.0), "Wheat": (0.1, 9.0), "Maize": (0.1, 12.0), "Cotton": (0.02, 3.5), "Groundnut": (0.05, 6.0),
                "Ragi": (0.05, 6.0), "Soybean": (0.05, 4.5), "Chickpea": (0.05, 4.0), "Pigeonpea": (0.05, 3.5), "Sorghum": (0.05, 8.0)}

# ---------------------------------------------------------------- season windows (crop-year start y; offsets in years)
# (start_month, start_day, start_year_offset, end_month, end_day, end_year_offset). Rabi ends Feb to match the app's weather auto-fill windows.
SEASON_WINDOWS = {"Kharif": (6, 1, 0, 10, 31, 0), "Rabi": (11, 1, 0, 2, 28, 1), "Summer": (3, 1, 1, 5, 31, 1),
                  "Whole Year": (6, 1, 0, 5, 31, 1), "Autumn": (7, 1, 0, 11, 30, 0), "Winter": (11, 1, 0, 2, 28, 1)}

# ---------------------------------------------------------------- features
KEY_COLS = ["region", "district", "crop", "season", "year"]
TARGET = "yield_t_ha"
CATEGORICAL_REAL = ["crop", "region", "season"]
NUMERIC_CANDIDATES = ["rainfall_mm", "temperature_c", "temp_max_c", "humidity_pct", "soil_ph", "soil_organic_carbon", "soil_sand_pct",
                      "soil_clay_pct", "latitude", "longitude", "year"]
SOIL_COLS = ["soil_ph", "soil_organic_carbon", "soil_sand_pct", "soil_clay_pct"]
REQUIRED_INPUTS_REAL = ["crop", "region", "season", "rainfall_mm", "temperature_c"]
MIN_FEATURE_COVERAGE = 0.5  # a candidate feature is used only if at least half the real rows have it

# plausible ranges used by validation (units as named)
RANGES = {"rainfall_mm": (0, 6000), "temperature_c": (-5, 40), "temp_max_c": (0, 50), "humidity_pct": (0, 100), "soil_ph": (3.5, 10),
          "soil_organic_carbon": (0, 400), "soil_sand_pct": (0, 100), "soil_clay_pct": (0, 100), "latitude": (6, 38), "longitude": (68, 98),
          "area_ha": (0, 5e6), TARGET: (0, 15)}

# Data provenance classes, shown in the app so users can tell what is observed vs estimated vs simulated.
PROVENANCE = {
    "yield": "observed",           # district yield records (area, production) from government statistics
    "weather": "regional_estimate",  # reanalysis grid cell at the district centroid, not a station at the field
    "soil": "regional_estimate",     # gridded soil property at the district centroid, not a soil test of the field
    "synthetic": "synthetic",
}