import type { CatKey, FormState, Inputs, NumKey } from './types'

export interface NumSpec { key: NumKey; min: number; max: number; step: number; unit: string; sliderMin?: number; sliderMax?: number }
export const NUM_SPECS: Record<NumKey, NumSpec> = {
  ph: { key: 'ph', min: 3.5, max: 10, step: 0.1, unit: '', sliderMin: 4, sliderMax: 9 },
  nitrogen: { key: 'nitrogen', min: 0, max: 600, step: 1, unit: 'kg/ha', sliderMin: 20, sliderMax: 250 },
  phosphorus: { key: 'phosphorus', min: 0, max: 250, step: 1, unit: 'kg/ha', sliderMin: 5, sliderMax: 80 },
  potassium: { key: 'potassium', min: 0, max: 800, step: 1, unit: 'kg/ha', sliderMin: 30, sliderMax: 300 },
  rainfall_mm: { key: 'rainfall_mm', min: 0, max: 6000, step: 10, unit: 'mm', sliderMin: 0, sliderMax: 2500 },
  temperature_c: { key: 'temperature_c', min: -5, max: 50, step: 0.5, unit: '°C', sliderMin: 10, sliderMax: 42 },
  humidity_pct: { key: 'humidity_pct', min: 0, max: 100, step: 1, unit: '%', sliderMin: 20, sliderMax: 98 },
  area_ha: { key: 'area_ha', min: 0.01, max: 5000, step: 0.1, unit: 'ha' },
  fertilizer_kg_ha: { key: 'fertilizer_kg_ha', min: 0, max: 1000, step: 5, unit: 'kg/ha', sliderMin: 0, sliderMax: 300 },
  year: { key: 'year', min: 1990, max: 2100, step: 1, unit: '' },
}
export const CAT_KEYS: CatKey[] = ['crop', 'region', 'season', 'soil_type', 'irrigation']

export const DEFAULT_FORM: FormState = {
  crop: 'Rice', region: 'Karnataka', season: 'Kharif', soil_type: 'Loamy', irrigation: 'Canal',
  ph: '6.5', nitrogen: '100', phosphorus: '30', potassium: '120', rainfall_mm: '800', temperature_c: '27',
  humidity_pct: '70', area_ha: '1', fertilizer_kg_ha: '90', year: String(new Date().getFullYear()), district: '',
}

export type Errors = Partial<Record<keyof FormState, 'required' | 'range'>>

export function validate(f: FormState, skip: (keyof FormState)[] = []): Errors {
  const e: Errors = {}
  for (const k of Object.keys(NUM_SPECS) as NumKey[]) {
    if (skip.includes(k)) continue
    const s = f[k].trim()
    if (s === '') { e[k] = 'required'; continue }
    const n = Number(s)
    if (!Number.isFinite(n) || n < NUM_SPECS[k].min || n > NUM_SPECS[k].max) e[k] = 'range'
  }
  for (const k of CAT_KEYS) if (!f[k]) e[k] = 'required'
  return e
}

/** used = the active model's input schema (undefined -> send everything). Fields the model does not use are never sent. */
export function toInputs(f: FormState, used?: string[]): Inputs {
  const n = (k: NumKey) => Number(f[k])
  const full: Inputs = {
    crop: f.crop, region: f.region, season: f.season, soil_type: f.soil_type, irrigation: f.irrigation,
    ph: n('ph'), nitrogen: n('nitrogen'), phosphorus: n('phosphorus'), potassium: n('potassium'), rainfall_mm: n('rainfall_mm'),
    temperature_c: n('temperature_c'), humidity_pct: n('humidity_pct'), area_ha: n('area_ha'), fertilizer_kg_ha: n('fertilizer_kg_ha'), year: n('year'),
  }
  if (!used) return full
  const keep = new Set([...used, 'crop', 'region', 'season', 'area_ha', 'year'])
  const out = Object.fromEntries(Object.entries(full).filter(([k]) => keep.has(k))) as unknown as Inputs
  if (f.district) out.district = f.district
  return out
}

/** Numeric form keys the active model does not use: skipped by validation. */
export function unusedKeys(used?: string[]): (keyof FormState)[] {
  return used ? (Object.keys(NUM_SPECS) as NumKey[]).filter((k) => !used.includes(k) && k !== 'area_ha' && k !== 'year') : []
}

export function fromInputs(i: Partial<Inputs>): FormState {
  // merge over defaults: a saved prediction from a different model may lack some fields
  const given = Object.entries(i).filter(([, v]) => v !== undefined && v !== null).map(([k, v]) => [k, String(v)])
  return { ...DEFAULT_FORM, ...Object.fromEntries(given) } as FormState
}

const LAST = 'cropai_last_inputs'
export const lastInputs = {
  load: (): Inputs | null => { try { const s = localStorage.getItem(LAST); return s ? (JSON.parse(s) as Inputs) : null } catch { return null } },
  save: (i: Inputs) => { try { localStorage.setItem(LAST, JSON.stringify(i)) } catch { /* storage unavailable */ } },
}