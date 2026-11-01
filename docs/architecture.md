# System architecture

## Components

| Layer | Technology | Responsibility |
|---|---|---|
| Presentation | React 18, Vite, TypeScript, Tailwind, Recharts, react-i18next, vite-plugin-pwa | Forms with validation and voice input, charts, EN/HI/KN, dark mode, offline shell |
| API | FastAPI, Pydantic v2, slowapi, python-jose, bcrypt | Auth, validation, rate limiting, orchestration, PDF reports |
| Inference | scikit-learn pipeline, LightGBM/XGBoost/CatBoost, SHAP | Point prediction, interval, explanation |
| Decision support | `services/recommendations.py` | Agronomic rules plus model-simulated levers plus profit logic |
| Persistence | SQLAlchemy 2, PostgreSQL (SQLite fallback) | Users, predictions, API usage |
| Experiment tracking | MLflow (SQLite backend) | Every model run: params, CV and test metrics, artifacts |
| DevOps | Docker, docker-compose, GitHub Actions, pytest, ruff, ESLint | Reproducible runtime and CI |

## Key design decisions

1. **One pipeline object does everything.** `FeatureEngineer` (imputation, winsorising, engineered features) and encoding live *inside* the scikit-learn pipeline that is pickled. The API cannot drift from training (no train/serve skew).
2. **Model bundle, not a bare model.** `model_vN.joblib` holds the selected pipeline, two quantile pipelines plus conformal `q_hat`, and a separate explainer pipeline. `model_vN.json` holds metrics, comparison table, benchmarks and category lists. The API loads the highest version (or `MODEL_VERSION`).
3. **Explainer ≠ predictor when needed.** SHAP's TreeExplainer needs a tree model. The stacking ensemble is the predictor; the explainer is the best single tree model by CV (CatBoost). Both reach similar accuracy, and the explanation is an explanation of the strongest base learner, not of the blend. This is stated in the UI docs and the viva sheet.
4. **Recommendations are partly model-simulated.** For irrigation, fertiliser and pH the service asks the model "what if?" and attaches the gain to the rule-based advice.
5. **Stateless API, stateful DB.** Horizontal scaling is possible because only the DB holds state. Model and explainer load once per process.
6. **SQLAlchemy `JSON` columns** store inputs and the full result payload, so history and PDF reports are reproducible even after the model changes (the stored `model_version` says which model produced them).

## Request lifecycle (`POST /predict`)

```mermaid
sequenceDiagram
  participant U as Browser
  participant A as FastAPI
  participant M as ModelService
  participant D as Database
  U->>A: POST /predict (JWT, JSON)
  A->>A: rate limit, JWT decode, Pydantic validation (ranges, categories)
  A->>M: full_prediction(inputs, crop)
  M->>M: pipeline.predict -> point
  M->>M: quantile models x relative band + q_hat -> 80% interval
  M->>M: SHAP (explainer) -> grouped top-5 factors
  M->>M: compare all crops -> alternatives; levers -> expected gains
  M->>M: recommendation engine -> sorted advice
  A->>D: INSERT prediction (inputs + result JSON)
  A-->>U: 200 PredictionOut
```

## Deployment topology (docker-compose)

`frontend` (nginx, static SPA, `/api/*` reverse-proxied) → `backend` (uvicorn, 2 workers) → `db` (PostgreSQL 16, volume). `ml/artifacts` is mounted read-only into the backend so retraining needs only a restart.
