# Methodology

## 1. Data
**Source: synthetic.** `ml/src/data_gen.py` simulates 9,000 field-seasons for 10 Indian crops (Rice, Wheat, Maize, Cotton, Groundnut, Ragi, Soybean, Chickpea, Pigeonpea, Sorghum), 10 states, 3 seasons, 6 soil types and 5 irrigation methods, years 2000-2023. Sugarcane was left out because its ~70 t/ha yield would dominate every error metric.

Yield = crop potential × product of response factors:
- Gaussian optimum curves for temperature and pH; water is **asymmetric** (steep penalty for deficit, gentle for surplus), where water = rainfall + 0.9 × irrigation-equivalent;
- Mitscherlich nutrient response with a Liebig-style limiting term (half minimum, half mean) and an excess penalty;
- soil suitability, heat stress above optimum + 7 °C, ~1.1 %/year technology trend, mild area effect, humidity disease penalty for some crops;
- multiplicative log-normal noise (σ = 0.08).

Then defects were injected: ~3 % missing values in 7 columns and ~0.6 % gross yield outliers (×0.1 or ×3.5). Sugar-coating avoided: the generator is smooth and low-noise, so **a high R² is partly built in**. A real dataset will score lower.

## 2. EDA (`ml/reports/eda/`)
Missing-value bars, distributions, correlation heatmap, yield vs rainfall and temperature, yield by crop and season, irrigation effect by season, yearly trend, soil effect. Because crop identity dominates raw yield, several plots use *relative yield* (`yield / crop median`) so that climate effects are visible. The irrigation plot shows the expected pattern: rainfed fields lose most in Rabi and Summer.

## 3. Cleaning and features
- **Outliers (target):** per-crop robust z-score using the median absolute deviation; rows with |z| > 6 removed (21 of 9,000). 
- **Missing values:** per-crop median imputation learned in `fit` (global median fallback), so test rows can never leak information into the imputer.
- **Winsorising:** features clipped to Q1 − 3·IQR … Q3 + 3·IQR (learned on train).
- **Engineered features (13):** rainfall × temperature interaction, total water (rain + irrigation), aridity index, heat-humidity index, N:P, N:K, P:K ratios, total NPK + fertiliser, fertiliser per soil N, pH deviation from 6.8, ordinal season code, log area, years since 2000. Season is additionally one-hot encoded.

## 4. Models and validation
Linear, Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, and a **stacking ensemble** (XGB + LGBM + CatBoost → Ridge meta-learner, 3 inner folds).
- Split: 80/20 stratified by crop, **test set used once** after all decisions.
- Hyper-parameters: Optuna TPE, objective = 3-fold CV RMSE on a 4,000-row subsample, **6 trials/model** (sandbox limit; use `--trials 30+` for a serious run).
- Final evaluation: 5-fold CV on the full train split (mean ± std of R², MAE, RMSE, MAPE) and test metrics.
- **Model selection uses CV RMSE only.** Every run is logged to MLflow (nested runs per model, Optuna trial history, metrics, plots, artifact).

## 5. Uncertainty (80% interval)
First attempt: LightGBM quantile regression (α = 0.1/0.9) on yield + conformalised quantile regression (CQR). Raw coverage was only 63.6 % and in sparse high-yield regions the band collapsed onto the point estimate.
Final design (`ml/src/uncertainty.py`):
1. out-of-fold LightGBM predictions `ŷ_oof` on the training split;
2. quantile regressors for the **ratio** `y / ŷ_oof` (multiplicative noise makes the ratio far more homoscedastic);
3. CQR on a held-out 20 % calibration split gives a correction `q̂`;
4. interval = `ŷ × [r_lo − q̂, r_hi + q̂]`, clamped to contain `ŷ`.
Result on unseen fields: coverage 76.6 % before and **81.4 % after** calibration (nominal 80 %), mean width 0.55 t/ha (vs 0.80 for the first design).

## 6. Explainability
`shap.TreeExplainer` on the best single tree model (CatBoost, by CV). SHAP values of one-hot columns are summed back to their parent feature (SHAP is additive, so this is exact). Global: beeswarm and grouped importance. Local: top-5 signed contributions (t/ha) per prediction, shown in the UI and PDF.

## 7. Recommendation engine
*Rule layer* (per-crop agronomic references): water balance vs crop optimum → irrigation need in mm; pH window → lime (≈2.5 t/ha per pH unit) or sulphur/gypsum; N/P/K supply vs need → kg/ha converted to urea (46 % N), single super phosphate (16 % P₂O₅), MOP (60 % K₂O); heat/cold stress; off-season warning; loss warning; profit sensitivity.
*Data layer:* top-quartile farm benchmarks mined from the data, and **model-simulated levers**: the model re-predicts yield after switching to the best irrigation, +30 % fertiliser, or optimum pH, and the gain is attached to the matching advice. A cross-crop comparison drives "a more profitable crop may exist".
Economics use indicative prices and costs (`ml/src/config.py`), which are placeholders, not market data.

## 8. Testing
48 pytest tests: auth flows and RBAC, prediction payload and interval ordering, an agronomic sanity check (drought yields less), validation matrix, history ownership isolation, PDF generation, compare ordering, what-if and sweep, weather parsing and failure (mocked), every recommendation rule, feature-engineering robustness, unseen-category handling. Tests build a small model via `ml.src.quick_train`.
