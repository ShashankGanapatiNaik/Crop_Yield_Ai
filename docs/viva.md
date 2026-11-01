# Viva cheat sheet: every major decision, with a defensible answer

Numbers are from this project (synthetic data, test set of 1,796 rows).

## The one-minute pitch
Farmers estimate yield by experience. This system predicts yield from soil, weather, water and crop, **shows why**, **gives a range**, and **recommends actions**, in three languages. ML core: stacking ensemble (R² 0.938). Explanations: SHAP. Uncertainty: quantile regression + conformal calibration (81.4 % coverage for an 80 % interval). Served by FastAPI, used through a React PWA.

## Data and honesty
**Q. Why synthetic data? Isn't that cheating?** No public dataset with all required columns (soil N/P/K, pH, weather, irrigation) was available offline. I built an agronomic simulator and said so everywhere. It validates the *pipeline*; accuracy claims on real farms would be invalid. The code reads `data/crop_yield.csv` automatically if you provide one.
**Q. Your R² is 0.94. Is that believable?** For this data, yes: the generator is smooth with 8 % noise. On real data expect lower. Saying this unprompted is stronger than defending the number.
**Q. How do you prevent data leakage?** Split first (stratified by crop); imputation medians, winsorising bounds and scaler are fitted *inside* the pipeline, hence inside each CV fold; the test set is used once, after selection; model selection uses CV RMSE not test.
**Q. Why a random split instead of by year?** Simplicity; I list it as a limitation. A leave-one-year-out check is the right next test.

## Features and preprocessing
**Q. Why these engineered features?** Rainfall × temperature (heat-moisture interaction), total water (rain + irrigation, since crops respond to total supply), aridity index, nutrient ratios (balance matters, not just amount), pH deviation (distance from optimum, trees can learn it but it helps), season code, log area.
**Q. How are missing values handled?** Per-crop median (a rice field's missing pH is best guessed from other rice fields), global median fallback; learned on train only.
**Q. Outliers?** Target: removed with a robust MAD z-score > 6 per crop (not plain z-score, because outliers inflate the std and hide themselves). Features: winsorised at 3 IQR so the API tolerates typos.

## Models
**Q. Why gradient boosting over linear regression?** The response is non-linear (optimum curves, saturation, interactions). Linear/Ridge stopped at R² 0.73; boosting reached 0.93-0.94. Ridge was identical to OLS, which shows the error is bias, not variance.
**Q. Why not deep learning?** 9k tabular rows: boosted trees usually beat neural nets on small tabular data, train in seconds, and work with SHAP.
**Q. Why stacking? Is it really better?** It blends three strong, differently-biased learners with a Ridge meta-learner. Honest answer: 0.266 vs 0.270 CV RMSE is inside the fold std (~0.03), so the gain is not statistically established. I selected it by the pre-declared rule (lowest CV RMSE). CatBoost alone would be a fine production choice.
**Q. Why Optuna?** TPE concentrates trials where the objective is good, so it beats grid search at equal budget. I used only 6 trials/model (CPU limit) and a subsample; I would use 50+ on a real run.
**Q. Why 5-fold CV?** Gives mean ± std of each metric, and uses data efficiently. The std is what lets me say stacking ≈ CatBoost.
**Q. Why did you remove Random Forest from the stack?** It was the slowest base learner and the weakest of the four; boosting models carry the blend.
**Q. What do the metrics mean?** R² = share of variance explained. MAE = typical error (0.17 t/ha). RMSE penalises big misses (0.265). MAPE = relative error (13 %), unreliable at small yields, so I decide with MAE/RMSE.

## Uncertainty
**Q. How do you get the interval?** Quantile regression (10th/90th percentile) on the **ratio y/ŷ**, using out-of-fold predictions, then conformal calibration so coverage holds in finite samples.
**Q. Why not plain quantile regression on yield?** I tried. Raw coverage was 63.6 % and in sparse high-yield regions the band collapsed onto the prediction. Noise is multiplicative, so absolute-scale quantiles were poorly behaved. Modelling the ratio fixed it: 81.4 % coverage, 31 % narrower.
**Q. What does "80 % interval" promise?** On average, 8 of 10 real fields fall inside. Not per-crop, not per-farm.

## Explainability
**Q. Why SHAP? Why not feature importance, LIME or permutation importance?** SHAP gives *signed, per-prediction* contributions in t/ha that sum to the prediction minus the baseline (additivity), is consistent, and is exact and fast for trees. Impurity importance is global, biased to high-cardinality features, and unsigned. LIME is unstable (random sampling).
**Q. Which model does SHAP explain?** CatBoost, the best single tree model, because TreeExplainer needs a tree and the stack is a blend. Stated openly.
**Q. Does SHAP prove causation?** No. It explains the model. "More rain raised the estimate" is not evidence that irrigating will raise real yield.
**Q. How do you show 20+ one-hot columns sensibly?** Sum SHAP values of one-hot columns back to the parent feature (exact because SHAP is additive).

## Backend
**Q. Why FastAPI?** Async-capable, Pydantic validation built in (range checks and category checks run before the model), automatic OpenAPI docs, type hints, and very fast. Flask would need extra libraries for each of these.
**Q. Why JWT?** Stateless auth: any API replica can verify tokens. Trade-offs: cannot revoke before expiry (12 h here), and storing in localStorage is XSS-exposed; cookies are the hardening step.
**Q. How are passwords stored?** bcrypt (salted, slow by design), never plain.
**Q. Why SQLAlchemy and PostgreSQL with SQLite fallback?** One ORM, two backends: SQLite for zero-setup dev and tests, PostgreSQL for concurrent production use. JSON columns store the full result.
**Q. Why pickle the whole pipeline?** So preprocessing at serving is *identical* to training (no train/serve skew). Cost: library versions must match, hence exact pins for ML libraries.
**Q. How do you secure the API?** Validation, RBAC (`/admin` = 403 for farmers), ownership checks (other users get 404 on your records), rate limiting, CORS allow-list, whitelisted file names, non-root container, secrets via environment.
**Q. What is the recommendation engine?** Hybrid. Rules encode agronomy (water balance, pH window, nutrient need in kg/ha → fertiliser quantities). The data layer uses benchmarks and **model-simulated levers**: the model re-predicts after each change, so advice carries an expected gain.
**Q. How does what-if work?** `POST /what-if` re-predicts with overrides and optionally sweeps one parameter to draw a response curve; the UI debounces sliders (250 ms) and ignores stale responses.

## Frontend
**Q. Why React + Vite + TypeScript + Tailwind?** Component model for many pages, fast dev builds, type safety against API shape drift, utility CSS with theme tokens (CSS variables) giving dark mode with no duplicated styles.
**Q. Why Recharts?** Declarative, React-native charts; fine for these sizes.
**Q. How is i18n done?** react-i18next with three JSON catalogues (230 keys each, parity checked by script); categories are translated through `opt.*` keys; fonts include Noto Devanagari/Kannada fallbacks.
**Q. How does voice input work?** Browser Web Speech API with `en-IN`/`hi-IN`/`kn-IN`; a parser converts "six point five", Devanagari or Kannada digits to numbers; for selects, the transcript is matched against option names. Limitations: Chromium/Safari, needs network.
**Q. How is it a PWA?** Web manifest plus a Workbox service worker (precache of the app shell, network-first for reference endpoints), so it installs and the shell opens offline. Predictions need the network.
**Q. Accessibility?** Skip link, labels bound to inputs, `aria-invalid` and described errors, `aria-live` toasts, visible focus ring, 44 px touch targets, reduced-motion respected, colour never the only signal.

## DevOps and testing
**Q. Why Docker?** Same runtime everywhere; compose wires Postgres + API + nginx; the SPA is built once and served statically, with `/api` proxied (no CORS in production).
**Q. What does CI do?** Ruff lint and pytest (with a quickly-trained model so CI needs no big artifacts), ESLint + type-check + production build, and a Docker build.
**Q. What do the tests prove?** 48 tests: auth and roles, validation matrix, ownership isolation, PDF output, every recommendation rule, feature-engineering robustness, plus an agronomic sanity test (a drought-stressed field must not out-yield a well-watered one).
**Q. What did you NOT verify?** Docker images and the PostgreSQL path were not executed in my sandbox; training used few Optuna trials; no load test.

## "What would you do next?"
Real data with temporal and spatial validation, monotonic constraints, satellite features, agronomist-reviewed localised advice, live prices, drift monitoring. (See `limitations.md`.)

## Likely trap questions
- *"Your model says rice yields 5 t/ha. Is that real?"* It's the simulator's potential. Real averages are lower, and that's why real data is first on my future-work list.
- *"Why is 'crop' the top SHAP feature, is the weather irrelevant?"* No. Crops differ in potential by more than weather moves any single crop; within a crop the weather and soil terms dominate (see the what-if curves).
- *"Why not just use R² to pick the model?"* Test-set selection leaks; CV RMSE on train is used, and RMSE is in agronomic units.
