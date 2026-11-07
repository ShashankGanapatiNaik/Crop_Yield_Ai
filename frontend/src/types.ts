export interface User { id: number; name: string; email: string; role: 'farmer' | 'admin' }
export interface TokenOut { access_token: string; token_type: string; user: User }

export interface Inputs {
  crop: string; region: string; season: string; soil_type: string; irrigation: string
  ph: number; nitrogen: number; phosphorus: number; potassium: number
  rainfall_mm: number; temperature_c: number; humidity_pct: number
  area_ha: number; fertilizer_kg_ha: number; year: number; district?: string
}
export type NumKey = 'ph' | 'nitrogen' | 'phosphorus' | 'potassium' | 'rainfall_mm' | 'temperature_c' | 'humidity_pct' | 'area_ha' | 'fertilizer_kg_ha' | 'year'
export type CatKey = 'crop' | 'region' | 'season' | 'soil_type' | 'irrigation'
export type FormState = Record<CatKey, string> & Record<NumKey, string> & { district: string }

export interface Factor { feature: string; label: string; value: number | string; shap: number; direction: 'increases' | 'decreases' }
export type Severity = 'critical' | 'warning' | 'info' | 'success'
export interface Recommendation { category: string; severity: Severity; title: string; detail: string; expected_gain_t_ha: number | null }
export interface Economics { price_per_tonne: number; revenue: number; cost: number; profit: number; roi_pct: number }
export interface Prediction {
  id: number; crop: string; inputs: Inputs; predicted_yield_t_ha: number; lower_t_ha: number; upper_t_ha: number; confidence_level: number
  total_production_t: number; economics: Economics; base_value_t_ha: number; top_factors: Factor[]; recommendations: Recommendation[]
  model_version: number; created_at: string
  model_name?: string | null; data_mode?: DataMode | null; estimated_inputs?: string[]; data_notes?: string[]
}
export interface HistoryItem { id: number; crop: string; predicted_yield_t_ha: number; lower_t_ha: number; upper_t_ha: number; total_production_t: number; profit: number; area_ha: number; created_at: string; model_version: number }
export interface CompareRow { crop: string; yield_t_ha: number; lower_t_ha: number; upper_t_ha: number; total_production_t: number; season_suitable: boolean; revenue: number; cost: number; profit: number; roi_pct: number }
export interface Scenario { yield_t_ha: number; lower_t_ha: number; upper_t_ha: number; revenue: number; cost: number; profit: number; roi_pct: number }
export interface WhatIf {
  crop: string; baseline: Scenario; scenario: Scenario & { inputs: Inputs; top_factors: Factor[] }
  delta_yield_t_ha: number; delta_pct: number; delta_profit: number
  sweep?: { param: string; points: { x: number; yield_t_ha: number; lower_t_ha: number; upper_t_ha: number }[] }
}
export interface Weather { temperature_c: number; humidity_pct: number; rainfall_mm: number; rainfall_last_90d_mm: number; period_days: number; next_7d_rainfall_mm: number; season: string | null; basis: string; seasons_used: { year: number; rainfall_mm: number; temperature_c: number | null }[]; note: string }
export interface CropMeta { name: string; seasons: string[]; optimum_ph: number; optimum_temp_c: number; optimum_water_mm: number; price_per_tonne: number; cost_per_ha: number }
export type DataMode = 'synthetic' | 'real'
export interface InputSchema { required: string[]; used: string[]; optional: string[] }
export interface Provenance { yield: string; weather: string; soil: string; sources?: Record<string, string>; dataset_version?: string | null; statement?: string }
export interface CropsMeta {
  crops: CropMeta[]; regions: string[]; seasons: string[]; soil_types: string[]; irrigation: string[]
  data_mode?: DataMode; model_name?: string | null; input_schema?: InputSchema | null; provenance?: Provenance | null; districts?: Record<string, string[]> | null
}
export interface ComparisonRow { model: string; is_baseline?: boolean; cv_rmse: number; cv_rmse_std: number; test_r2: number; test_mae: number; test_rmse: number; test_mape: number }
export interface ModelInfo {
  data_mode?: DataMode; name?: string; provenance?: Provenance; split?: { train_years: number[]; val_years: number[]; test_years: number[]; n_train: number; n_val: number; n_test: number }
  synthetic_reference?: { best_model: string; metrics_test: { r2: number }; note: string } | null
  version: number; trained_at: string; data_source: string; best_model: string; explainer_model: string; n_train: number; n_test: number; cv_folds: number; optuna_trials: number
  metrics_test: { r2: number; mae: number; rmse: number; mape: number }
  comparison: ComparisonRow[]
  per_crop_metrics: Record<string, { r2: number | null; mae: number; rmse: number; mape: number; n?: number }>
  per_region_metrics?: Record<string, { r2: number | null; mae: number; rmse: number; mape: number; n: number }>
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

// ---- satellite crop health
export interface GeoPolygon { type: 'Polygon'; coordinates: number[][][] }
export type NdviCategory = 'low' | 'moderate' | 'high'
export interface FarmField {
  id: number; name: string; geometry: GeoPolygon; centroid_lat: number; centroid_lon: number; area_ha: number
  source: 'polygon' | 'point_radius'; radius_m: number | null; created_at: string; last_synced_at: string | null; last_sync_status: string | null
}
export interface NdviObs { date: string; ndvi_mean: number; valid_fraction: number; scenes: number; category: NdviCategory; color: string; label: string }
export interface LegendItem { category: NdviCategory; label: string; color: string; range: string }
export interface CropHealthCommon {
  status: 'ok' | 'no_data'; reason: string | null; field: { id: number; name: string; area_ha: number }
  window: { start: string; end: string; days: number }; thresholds: { low: number; high: number; note: string }; legend: LegendItem[]
  from_cache: boolean; stale: boolean; warning: string | null; last_synced_at: string | null
  source: { satellite: string; dataset: string; index: string; resolution_m: number; cloud_mask: string; revisit: string; attribution: string }
}
export interface CropHealthLatest extends CropHealthCommon {
  observation: NdviObs | null; previous: NdviObs | null
  interpretation: { category: NdviCategory; summary: string; change: number | null; caveats: string[]; disclaimer: string } | null
  layer: { tile_url: string; style: string; date: string; attribution: string; expires_note: string; bounds: [[number, number], [number, number]] } | null
  layer_error: string | null
}
export interface CropHealthHistory extends CropHealthCommon { count: number; observations: NdviObs[] }
export interface CropHealthStatus {
  configured: boolean; provider: string; dataset: string; thresholds: { low: number; high: number }; legend: LegendItem[]
  limits: { min_field_ha: number; max_field_ha: number; max_fields_per_user: number; default_days: number; max_days: number }
}
export interface FieldCreateBody { name: string; geometry?: GeoPolygon; latitude?: number; longitude?: number; radius_m?: number }