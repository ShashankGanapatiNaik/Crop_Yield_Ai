# Limitations and future scope

## Limitations (be upfront about these in your viva)

**Data**
1. **Synthetic training data (shipped model).** A real-data pipeline now exists (see [real-data.md](real-data.md)) but no real model is shipped; until you build one, everything below about synthetic data applies.
   Original note: **Synthetic training data.** The model has learned the simulator, not real farms. Reported R² (0.94) is optimistic; real yield data typically gives R² 0.5-0.8 because of unmeasured factors (variety, sowing date, pests, management).
2. **Random split, not temporal or spatial.** The test set shares years and regions with training. Performance on a future season or a new district is untested. A leave-one-year-out or leave-one-state-out evaluation is needed.
3. **Year extrapolation.** Trees cannot extrapolate: predictions for years after 2023 reuse the 2023 trend level.
4. **Sparse optimum region.** High-yield, ideal-condition fields are rare in the data, so estimates there are less certain (one reason intervals use relative error).
5. **Soil nutrient scale.** N/P/K are "available kg/ha" with agronomic weights chosen for the simulator. Real soil-test units and methods (Olsen P, ammonium acetate K) differ and need mapping.

**Modelling**
6. **Stacking gain is within noise** (0.266 vs CatBoost 0.270 CV RMSE, std ≈ 0.03). It is selected by rule, not proven better.
7. **Explainer ≠ predictor.** SHAP explains CatBoost, the strongest single tree model, not the stacked blend.
8. **Hyper-parameter search was small** (6 Optuna trials, 4k-row subsample) because of sandbox limits.
9. **Interval coverage is marginal**, and is not guaranteed per crop, region or season.
10. **Residual skew −1.88**: occasional large over-predictions; likely linked to remaining outliers (hypothesis, not tested).
11. **SHAP is associative, not causal.** "Rainfall raised the estimate" does not prove that adding water will raise real yield. What-if answers describe the *model*, not the field.

**Product**
12. **Recommendations are guidance.** Lime and fertiliser doses use generic conversion factors; users are told to confirm with a soil test and extension officer.
13. **Prices and costs are placeholders** (MSP-like constants), so profit figures are indicative.
14. **Recommendation text is English-only**; the surrounding UI is localised. Hindi and Kannada strings were written by the developer and should be reviewed by native speakers and agronomists.
15. **Voice input** uses the browser Web Speech API: works in Chromium-based browsers and Safari with network access; absent or unreliable elsewhere, and numbers spoken in words are only partly parsed.
16. **Seasonal rainfall from weather auto-fill** is a 90-day total, not the true season total.
17. **Not load-tested**; the PostgreSQL path and Docker images were written but not executed in the development sandbox (no Docker there). Run `docker compose up --build` and the test-suite against Postgres before relying on it.
18. **No password reset, email verification, refresh tokens or audit trail.** The JWT lives in `localStorage` (simple, but exposed to XSS; httpOnly cookies are safer).

## Future scope
- Train and validate on real data (ICRISAT district-level yields, state agriculture departments, Kaggle India crop yield), joined with IMD / NASA POWER weather and Soil Health Card data.
- Temporal/spatial cross-validation; per-crop or hierarchical models; monotonic constraints (more drought → not more yield) via LightGBM/XGBoost.
- Satellite features (NDVI/EVI from Sentinel-2), soil maps (SoilGrids), and in-season updating as weather unfolds.
- Causal or quasi-experimental evidence behind recommendations; optimisation (best fertiliser dose under a budget).
- Live mandi prices (Agmarknet) for profit; price uncertainty.
- Localised, agronomist-reviewed advice text; text-to-speech output for low-literacy users; SMS/WhatsApp delivery for low connectivity.
- Model monitoring and drift detection, scheduled retraining, A/B comparison of model versions (the version-tagged history already supports this).
- Security hardening: httpOnly cookies, refresh tokens, email verification, audit logs; background jobs for PDF generation.