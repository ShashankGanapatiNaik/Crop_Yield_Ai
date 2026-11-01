# Data flow

## Training-time

```mermaid
flowchart TD
  A[data/crop_yield.csv if present<br/>else synthetic generator] --> B[Clean: drop missing target,<br/>per-crop MAD outlier removal]
  B --> C[Stratified 80/20 split by crop]
  C --> D[Train 7183 rows]
  C --> T[Test 1796 rows - touched once]
  D --> E[Optuna tuning<br/>3-fold CV on 4k subsample]
  E --> F[5-fold CV on full train<br/>R2 / MAE / RMSE / MAPE]
  F --> G[Stacking: XGB + LGBM + CatBoost, Ridge meta-learner]
  F --> H[Select best by CV RMSE]
  G --> H
  H --> I[Out-of-fold surrogate predictions]
  I --> J[Quantile regression on relative error y / y_hat]
  J --> K[Conformal calibration q_hat]
  H --> L[SHAP on best tree model]
  H --> T
  T --> M[Test metrics, residual plots, coverage]
  H & K & L --> N[(model_vN.joblib + model_vN.json)]
  F & M --> O[(MLflow runs + plots)]
```

Inside every pipeline: `FeatureEngineer` (per-crop median imputation → IQR winsorising → 13 engineered features) → `ColumnTransformer` (StandardScaler for numerics, one-hot for categoricals) → estimator.

## Inference-time

```mermaid
flowchart LR
  F[Form values / voice / weather auto-fill] --> V[Client validation]
  V --> P[POST /predict]
  P --> S[Pydantic validation]
  S --> Pipe[Pipeline -> yield t/ha]
  Pipe --> I[Interval = yield x band]
  Pipe --> X[SHAP -> top-5 factors]
  Pipe --> L[Lever simulation + crop comparison]
  L --> R[Recommendation engine]
  I & X & R --> Out[JSON response]
  Out --> H[(predictions table)]
  Out --> UI[Result page / PDF]
```

## Weather auto-fill

Browser geolocation → `GET /weather?lat&lon` → Open-Meteo (90 past days + 7 forecast days) → mean temperature, mean humidity, **90-day rainfall total** → pre-fills the form. Responses are cached 30 minutes per ~1 km cell. The 90-day total approximates, but is not equal to, seasonal rainfall, so the UI says the user should adjust.
