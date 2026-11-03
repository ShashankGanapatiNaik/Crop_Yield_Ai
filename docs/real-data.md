# Real district-level data pipeline

This adds a second, **real-data** training path next to the original synthetic one. Nothing from the synthetic path was removed:
`data/crop_yield_synthetic.csv`, `ml/artifacts/model_v1.*` and the synthetic training code are untouched, and tests/prototyping keep using them.

> **Honest status.** The pipeline, its tests and the serving/UI changes are built and tested **on fixtures and mocked API responses**.
> The development sandbox had no access to the real data hosts (NASA POWER, Open-Meteo, SoilGrids, data.gov.in, Kaggle), so **no real
> dataset was downloaded, no real model was trained, and no real-model artifact is shipped**. Real numbers will appear only after *you* run the
> steps below. Live API response shapes were written from provider documentation; check one district by hand on your first run (step 6).

## What is real, what is estimated, what is simulated

| Class | What | Where it comes from |
|---|---|---|
| **Observed** | District yield = production ÷ area | Government of India district-wise APY statistics |
| **Regional estimate** | Season rainfall, mean/max temperature, humidity | Gridded weather (NASA POWER or Open-Meteo/ERA5) at the **district centroid** |
| **Regional estimate** | Soil pH, organic carbon, sand %, clay % ; latitude/longitude | SoilGrids point value at the district centroid (or your own CSV) |
| **Simulated** | Everything in the original prototype model | `ml/src/data_gen.py` |

The app shows this on the Predict, Result and Model-insights pages (badge + legend), and the API returns `data_mode`, `estimated_inputs` and `data_notes` with every prediction.

**The real model predicts a district-level average yield, not one farm's yield.** It has no fertiliser, N-P-K, irrigation or soil-test inputs because the source data has none, and none are invented.

## 1. Data sources (verify each licence/terms yourself)

* **Yields (required).** District-wise, season-wise APY (area, production, yield) from the Ministry of Agriculture & Farmers Welfare (Directorate of Economics & Statistics). Public copies exist on the India Data Portal (columns include `area_unit`, `production_unit`, so units are explicit), data.gov.in, and a widely used Kaggle derivative (`State_Name, District_Name, Crop_Year, Season, Crop, Area, Production`, no unit columns). All three layouts are auto-detected.
* **Weather.** `nasa_power` (default: no key, one request per district for the whole period; daily `PRECTOTCORR`, `T2M`, `T2M_MAX`, `RH2M`) or `open_meteo` (ERA5 archive; **no humidity** is requested, so humidity is simply absent). Open-Meteo's free tier meters long date ranges heavily, which is why NASA POWER is the bulk default.
* **Soil.** SoilGrids v2 REST point queries (top 30 cm mean of pH, soil organic carbon, sand, clay; public API is limited to roughly 5 calls/minute, about 2 h for 600 districts), **or** your own `region,district,soil_ph,soil_organic_carbon,soil_sand_pct,soil_clay_pct` CSV (e.g. from Indian soil-survey data).
* **District coordinates.** Best: your own centroid CSV (`state,district,latitude,longitude`). Otherwise OpenStreetMap Nominatim (1 request/s; results are validated to lie in India **and** in the right state, else the district is dropped). Set `CROPAI_CONTACT=you@example.com` so requests identify you.

Known data hazards, all handled or reported (never silently): production units (cotton in bales, coconut in nuts), duplicate rows, zero production meaning "not reported", name spellings (`MYSORE`/`Mysuru`, `Orissa`/`Odisha`), district splits after 2000 (names that no longer match are dropped and listed in the merge report), pre-2014 Andhra Pradesh/Telangana boundaries.

## 2. Reproduce (first run: one state)

```bash
pip install -r requirements.txt          # Python 3.11+ (uses tomllib)
export CROPAI_CONTACT=you@example.com    # Windows PowerShell:  $env:CROPAI_CONTACT="you@example.com"

# edit ml/real_data.toml:  [apy] states = ["Karnataka"]   <- recommended for the FIRST run

python -m ml.src.data_download apy --src /path/to/apy.csv          # or an https URL; writes data/real/raw/apy_raw.csv + provenance (sha256)
python -m ml.src.data_download centroids --csv my_centroids.csv    # or omit --csv to geocode with Nominatim (resumable)
python -m ml.src.data_download weather --provider nasa_power       # resumable; cached under data/real/raw/weather/
python -m ml.src.data_download soil --source soilgrids             # resumable; or: --source csv --csv my_soil.csv

python -m ml.src.build_real_dataset      # clean -> validate -> merge -> validate  (offline). Writes data/real/processed/
python -m ml.src.train --data real --trials 30
mlflow ui --backend-store-uri sqlite:///ml/mlflow.db               # experiment "crop-yield-real-district"
```

Outputs: `data/real/processed/district_yield.csv`, `district_yield.meta.json` (dataset version hash, sources, merge accounting), `validation_report.json`, `apy_clean.csv`; model `ml/artifacts/real_district_vN.{joblib,json}`; plots in `ml/reports/real/real_district_vN/`.

**Serve it** (the synthetic model stays the default):

```bash
MODEL_VERSION=real_district_v1 uvicorn backend.app.main:app      # or real_latest
```

**Check one district by hand (step 6).** Open `data/real/raw/weather/nasa_power/<state>__<district>.csv`, sum `rain_mm` over one Kharif window and compare with a local/IMD figure; open `soil_district.csv` and sanity-check pH (4.5-8.5 typical). If a provider's payload shape ever changes, the parsers in `data_download.py` (`parse_power`, `parse_open_meteo`, `parse_soilgrids`) are the only places to adjust, and `tests/test_real_merge.py` shows the shape they expect.

## 3. Processing rules

* `yield_t_ha = production_t / area_ha`, always recomputed (a source yield column is only a cross-check, reported as `yield_crosscheck_agree_pct`).
* Dropped and **counted** at each step: unsupported crop, other seasons (default keeps Kharif/Rabi/Summer), unparseable year, missing/zero/negative area, missing production, **zero production** (config), area below 100 ha, unit that cannot be converted, yield outside crop-specific physical bounds. Duplicate keys are summed.
* **Cotton** is excluded when the file has no unit column unless you set `[cleaning].unit_assumptions = {Cotton = 0.17}` (APY bale = 170 kg). No silent unit assumptions.
* Season windows use the crop-year start `y` (e.g. `2015-16` -> 2015): Kharif Jun-Oct `y`; Rabi Nov `y`-Feb `y+1`; Summer Mar-May `y+1`. A season needs ≥ 90 % daily coverage; rainfall is scaled for a few missing days; otherwise the row is dropped, **never filled**.
* A candidate feature is used only if ≥ 50 % of rows have it (`make_district_spec`). Missing values are imputed per crop (median, learned on training years only).

## 4. Training and evaluation protocol

| Step | Rule |
|---|---|
| Split | Chronological by crop year: train = earliest, validation = next `val_years` (2), test = latest `test_years` (3). Asserted: train < val < test. |
| Tuning | Optuna on **training years only**, with expanding-window year blocks (each fold validates on years after its training years). |
| Selection | Lowest **validation** RMSE. The test years are used only for reporting. |
| Uncertainty | Ratio-quantile regression fitted on training years; conformal correction `q̂` computed on the (later) validation years; coverage reported on test years. |
| Deployed model | Refit on `final_fit` data: `train_val` by default (the evaluated model is the train-only fit; test years stay held out). |
| Reporting | R², MAE, RMSE, MAPE overall; **per crop and per state** (groups under `min_rows_per_group_report` are counted, not reported); residual and SHAP plots. |

**Comparison.** Each real-data model is listed next to two no-weather/no-soil baselines (crop × state × season mean, and district-crop history mean) so you can see what weather and soil actually add. The synthetic model's own test metrics are shown for reference with an explicit note that they are **not comparable** (different data, different inputs).

**MLflow** logs, per run: dataset version/hash, weather and soil source, split years, feature spec (`feature_spec.json`), Optuna history, parameters, CV/validation/test metrics, tags (`data_mode=real`, provenance classes), and the model + metadata artifacts. Real models are versioned `real_district_vN` and **never overwrite** `model_vN`.

**Lineage guard.** `train_real` refuses to build a "real" model unless every row has `yield_source == "apy_observed"`, so fixtures or synthetic rows can never be labelled real.

## 5. Serving and API compatibility

* `MODEL_VERSION`: `latest` (synthetic `model_vN`, unchanged default), `N`, `real_latest`, or `real_district_vN`.
* `POST /predict`, `/compare-crops`, `/what-if` keep their shape. Fields a model does not use become optional; the **active model decides what is required** (`GET /crops` returns `input_schema`, `data_mode`, `districts`). Missing required inputs and unknown categories return the same `422 {"detail": "Validation failed", "errors": [...]}`.
* Real model inputs: `crop, region (state), season, rainfall_mm, temperature_c` (+ optional `humidity_pct, district, latitude/longitude, area_ha, year`). Soil and location are filled server-side from the district (or nearest district, or the state median) and listed in `estimated_inputs`.
* Responses add `model_name`, `data_mode`, `estimated_inputs`, `data_notes`. A value outside the range the model saw **for that crop and season** produces a data note (trees cannot extrapolate).
* Real-model recommendations: no N-P-K, fertiliser or irrigation-lever advice (no such inputs). Rainfall is judged against what districts actually experienced for that crop and season, not simulator constants.

## 6. Limitations (read before quoting any number)

1. **District averages, not farms** (ecological fallacy). Accuracy for an individual field is unknown.
2. **Weather and soil are gridded estimates at a centroid**, not the farm. Reanalysis tends to under-estimate orographic rain (Western Ghats, coast). Large districts are poorly summarised by one point.
3. **No irrigation, fertiliser, variety or sowing-date information.** Irrigated districts are partly explained by weather they do not depend on.
4. **Season windows are approximations** of true sowing-to-harvest periods, and crop-year labels follow the APY start-year convention.
5. **APY quality varies** (reporting errors, unit quirks, district re-organisation). Dropped rows are listed, but unnoticed errors remain.
6. **Time trend:** `year` is a feature; trees cannot extrapolate beyond the last training year.
7. **Profit figures** use placeholder prices and costs in `ml/src/config.py`.
8. **The 10 supported crops** are the app's list; other APY crops are dropped (extend `CROP_ALIASES`, `YIELD_BOUNDS`, `ECONOMICS`, `CROP_PROFILE` together).

## 7. Tests (no internet needed)

`tests/test_real_cleaning.py` (normalisation, units, yield formula, invalid/duplicate rows, validation), `test_real_merge.py` (season windows, coverage rule, join accounting, mocked NASA POWER / Open-Meteo / SoilGrids / Nominatim payloads, resumable downloads), `test_real_train.py` (no temporal leakage, blocked CV, baselines, feature schema, end-to-end training, versioning/no-overwrite, strict-JSON metadata, lineage guard), `test_real_serving.py` (schema, district lookup, estimate flags, out-of-range notes, API contract, synthetic contract unchanged). All fixtures are labelled and fabricated (`tests/real_fixtures.py`) and are never written under `data/real/`.