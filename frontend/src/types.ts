export interface User { id: number; name: string; email: string; role: 'farmer' | 'admin' }
export interface TokenOut { access_token: string; token_type: string; user: User }

export interface Inputs {
  crop: string; region: string; season: string; soil_type: string; irrigation: string
  ph: number; nitrogen: number; phosphorus: number; potassium: number
  rainfall_mm: number; temperature_c: number; humidity_pct: number
  area_ha: number; fertilizer_kg_ha: number; year: number
}
export type NumKey = 'ph' | 'nitrogen' | 'phosphorus' | 'potassium' | 'rainfall_mm' | 'temperature_c' | 'humidity_pct' | 'area_ha' | 'fertilizer_kg_ha' | 'year'
export type CatKey = 'crop' | 'region' | 'season' | 'soil_type' | 'irrigation'
export type FormState = Record<CatKey, string> & Record<NumKey, string>

export interface Factor { feature: string; label: string; value: number | string; shap: number; direction: 'increases' | 'decreases' }
export type Severity = 'critical' | 'warning' | 'info' | 'success'
export interface Recommendation { category: string; severity: Severity; title: string; detail: string; expected_gain_t_ha: number | null }
export interface Economics { price_per_tonne: number; revenue: number; cost: number; profit: number; roi_pct: number }
export interface Prediction {
  id: number; crop: string; inputs: Inputs; predicted_yield_t_ha: number; lower_t_ha: number; upper_t_ha: number; confidence_level: number
  total_production_t: number; economics: Economics; base_value_t_ha: number; top_factors: Factor[]; recommendations: Recommendation[]
  model_version: number; created_at: string
}
export interface HistoryItem { id: number; crop: string; predicted_yield_t_ha: number; lower_t_ha: number; upper_t_ha: number; total_production_t: number; profit: number; area_ha: number; created_at: string; model_version: number }
export interface CompareRow { crop: string; yield_t_ha: number; lower_t_ha: number; upper_t_ha: number; total_production_t: number; season_suitable: boolean; revenue: number; cost: number; profit: number; roi_pct: number }
export interface Scenario { yield_t_ha: number; lower_t_ha: number; upper_t_ha: number; revenue: number; cost: number; profit: number; roi_pct: number }
export interface WhatIf {
  crop: string; baseline: Scenario; scenario: Scenario & { inputs: Inputs; top_factors: Factor[] }
  delta_yield_t_ha: number; delta_pct: number; delta_profit: number
  sweep?: { param: string; points: { x: number; yield_t_ha: number; lower_t_ha: number; upper_t_ha: number }[] }
}
export interface Weather { temperature_c: number; humidity_pct: number; rainfall_mm: number; period_days: number; next_7d_rainfall_mm: number; note: string }
export interface CropMeta { name: string; seasons: string[]; optimum_ph: number; optimum_temp_c: number; optimum_water_mm: number; price_per_tonne: number; cost_per_ha: number }
export interface CropsMeta { crops: CropMeta[]; regions: string[]; seasons: string[]; soil_types: string[]; irrigation: string[] }
export interface ComparisonRow { model: string; cv_rmse: number; cv_rmse_std: number; test_r2: number; test_mae: number; test_rmse: number; test_mape: number }
export interface ModelInfo {
  version: number; trained_at: string; data_source: string; best_model: string; explainer_model: string; n_train: number; n_test: number; cv_folds: number; optuna_trials: number
  metrics_test: { r2: number; mae: number; rmse: number; mape: number }
  comparison: ComparisonRow[]
  per_crop_metrics: Record<string, { r2: number; mae: number; rmse: number; mape: number }>
  uncertainty: { coverage_calibrated: number; coverage_raw_quantile: number; mean_interval_width_t_ha: number; nominal_coverage: number }
  global_importance: Record<string, number>
  residual_summary: { mean: number; std: number; skew: number }
}
export interface AdminStats {
  users: number; predictions: number; requests: number; server_errors: number
  predictions_per_day: { date: string; count: number }[]
  by_crop: { crop: string; count: number; avg_yield: number }[]
  endpoints: { path: string; count: number; avg_latency_ms: number }[]
}
export interface AdminUser { id: number; name: string; email: string; role: string; created_at: string; predictions: number }

export interface Place { lat: number; lon: number; label: string; display_name: string; locality: string; district: string; state: string; country: string; region: string | null }