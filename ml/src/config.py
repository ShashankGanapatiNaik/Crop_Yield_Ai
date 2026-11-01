"""Shared configuration for the ML pipeline and the backend."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "ml" / "artifacts"
REPORT_DIR = ROOT / "ml" / "reports"
MLFLOW_URI = f"sqlite:///{ROOT / 'ml' / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "crop-yield-prediction"
RANDOM_STATE = 42

TARGET = "yield_t_ha"

CATEGORICAL = ["crop", "region", "season", "soil_type", "irrigation"]
NUMERIC = ["ph", "nitrogen", "phosphorus", "potassium", "rainfall_mm", "temperature_c",
           "humidity_pct", "area_ha", "fertilizer_kg_ha", "year"]
RAW_FEATURES = CATEGORICAL + NUMERIC

CROPS = ["Rice", "Wheat", "Maize", "Cotton", "Groundnut", "Ragi", "Soybean", "Chickpea", "Pigeonpea", "Sorghum"]
REGIONS = ["Karnataka", "Punjab", "Maharashtra", "Uttar Pradesh", "Tamil Nadu", "Andhra Pradesh",
           "West Bengal", "Madhya Pradesh", "Gujarat", "Bihar"]
SEASONS = ["Kharif", "Rabi", "Summer"]
SOILS = ["Alluvial", "Black", "Red", "Laterite", "Sandy", "Loamy"]
IRRIGATION = ["Rainfed", "Canal", "Tubewell", "Sprinkler", "Drip"]

# Water supplied by each irrigation method, expressed as mm of rainfall-equivalent per season.
IRRIGATION_MM = {"Rainfed": 0, "Canal": 220, "Tubewell": 260, "Sprinkler": 200, "Drip": 280}
SEASON_ORDINAL = {"Kharif": 0, "Rabi": 1, "Summer": 2}

# Agronomic profile per crop. Used by the synthetic generator (as the "ground truth" physics)
# and by the recommendation engine (as agronomic reference ranges).
#   base: potential yield t/ha | temp: (optimum C, tolerance) | rain: (optimum mm, tolerance)
#   ph: optimum pH | npk: optimum nutrient supply kg/ha | seasons: allowed seasons
CROP_PROFILE = {
    "Rice":      dict(base=4.6, temp=(27, 5), rain=(1100, 380), ph=6.2, npk=(130, 55, 100), seasons=["Kharif", "Summer"]),
    "Wheat":     dict(base=4.2, temp=(20, 5), rain=(420, 160),  ph=6.8, npk=(130, 55, 90),  seasons=["Rabi"]),
    "Maize":     dict(base=4.8, temp=(25, 5), rain=(620, 220),  ph=6.5, npk=(150, 60, 90),  seasons=["Kharif", "Rabi"]),
    "Cotton":    dict(base=2.2, temp=(28, 5), rain=(700, 240),  ph=7.0, npk=(130, 55, 100), seasons=["Kharif"]),
    "Groundnut": dict(base=2.1, temp=(27, 4), rain=(560, 190),  ph=6.3, npk=(60, 50, 90),   seasons=["Kharif", "Summer"]),
    "Ragi":      dict(base=2.6, temp=(26, 4), rain=(480, 170),  ph=6.0, npk=(80, 45, 70),   seasons=["Kharif"]),
    "Soybean":   dict(base=2.4, temp=(26, 4), rain=(650, 190),  ph=6.5, npk=(60, 55, 90),   seasons=["Kharif"]),
    "Chickpea":  dict(base=1.7, temp=(21, 4), rain=(330, 140),  ph=7.0, npk=(50, 50, 70),   seasons=["Rabi"]),
    "Pigeonpea": dict(base=1.4, temp=(27, 5), rain=(700, 230),  ph=6.6, npk=(50, 50, 80),   seasons=["Kharif"]),
    "Sorghum":   dict(base=2.0, temp=(28, 5), rain=(500, 200),  ph=6.7, npk=(100, 50, 80),  seasons=["Kharif", "Rabi"]),
}

# Soil suitability multipliers (1.0 = ideal).
SOIL_FACTOR = {
    "Rice": {"Alluvial": 1.0, "Black": 0.95, "Red": 0.85, "Laterite": 0.8, "Sandy": 0.7, "Loamy": 0.98},
    "Wheat": {"Alluvial": 1.0, "Black": 0.95, "Red": 0.82, "Laterite": 0.75, "Sandy": 0.72, "Loamy": 1.0},
    "Maize": {"Alluvial": 0.97, "Black": 0.95, "Red": 0.9, "Laterite": 0.82, "Sandy": 0.8, "Loamy": 1.0},
    "Cotton": {"Alluvial": 0.92, "Black": 1.0, "Red": 0.85, "Laterite": 0.75, "Sandy": 0.72, "Loamy": 0.95},
    "Groundnut": {"Alluvial": 0.9, "Black": 0.85, "Red": 0.97, "Laterite": 0.9, "Sandy": 1.0, "Loamy": 0.98},
    "Ragi": {"Alluvial": 0.85, "Black": 0.82, "Red": 1.0, "Laterite": 0.95, "Sandy": 0.88, "Loamy": 0.95},
    "Soybean": {"Alluvial": 0.95, "Black": 1.0, "Red": 0.85, "Laterite": 0.78, "Sandy": 0.75, "Loamy": 0.97},
    "Chickpea": {"Alluvial": 0.95, "Black": 1.0, "Red": 0.85, "Laterite": 0.75, "Sandy": 0.8, "Loamy": 0.97},
    "Pigeonpea": {"Alluvial": 0.9, "Black": 1.0, "Red": 0.95, "Laterite": 0.85, "Sandy": 0.75, "Loamy": 0.95},
    "Sorghum": {"Alluvial": 0.92, "Black": 1.0, "Red": 0.93, "Laterite": 0.82, "Sandy": 0.8, "Loamy": 0.97},
}

# Indicative economics (INR). Used for profit estimates. Prices are illustrative and editable.
ECONOMICS = {
    "Rice":      dict(price_per_tonne=22000, cost_per_ha=58000),
    "Wheat":     dict(price_per_tonne=24000, cost_per_ha=46000),
    "Maize":     dict(price_per_tonne=21000, cost_per_ha=48000),
    "Cotton":    dict(price_per_tonne=70000, cost_per_ha=72000),
    "Groundnut": dict(price_per_tonne=65000, cost_per_ha=56000),
    "Ragi":      dict(price_per_tonne=42000, cost_per_ha=36000),
    "Soybean":   dict(price_per_tonne=47000, cost_per_ha=42000),
    "Chickpea":  dict(price_per_tonne=56000, cost_per_ha=38000),
    "Pigeonpea": dict(price_per_tonne=70000, cost_per_ha=40000),
    "Sorghum":   dict(price_per_tonne=31000, cost_per_ha=33000),
}

# Region climate offsets (rainfall multiplier, temperature offset C) - crude but plausible.
REGION_CLIMATE = {
    "Karnataka": (1.0, 0.0), "Punjab": (0.8, -1.5), "Maharashtra": (0.9, 1.0), "Uttar Pradesh": (0.95, -0.5),
    "Tamil Nadu": (0.95, 2.0), "Andhra Pradesh": (0.95, 2.0), "West Bengal": (1.3, 0.5),
    "Madhya Pradesh": (0.95, 0.5), "Gujarat": (0.75, 1.5), "Bihar": (1.1, 0.0),
}
