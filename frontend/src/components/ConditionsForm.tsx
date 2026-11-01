import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, ApiError } from '../api'
import { useToast } from '../toast'
import type { Errors } from '../ranges'
import type { CropsMeta, FormState, Place } from '../types'
import LocationPicker from './LocationPicker'
import { Icon, NumField, SelField, Spinner } from './ui'

interface Props { form: FormState; setForm: (f: FormState) => void; errors: Errors; meta: CropsMeta; hideCrop?: boolean }

export default function ConditionsForm({ form, setForm, errors, meta, hideCrop }: Props) {
  const { t } = useTranslation()
  const toast = useToast()
  const [locating, setLocating] = useState(false)
  const [picker, setPicker] = useState<null | { auto: boolean }>(null)
  const [place, setPlace] = useState<Place | null>(null)
  const set = (k: keyof FormState) => (v: string) => setForm({ ...form, [k]: v })
  const allowed = meta.crops.find((c) => c.name === form.crop)?.seasons ?? meta.seasons

  const setCrop = (crop: string) => {
    const seasons = meta.crops.find((c) => c.name === crop)?.seasons ?? meta.seasons
    setForm({ ...form, crop, season: seasons.includes(form.season) ? form.season : seasons[0] })
  }

  const applyLocation = async (lat: number, lon: number, p: Place | null) => {
    setPicker(null); setLocating(true)
    try {
      const w = await api.weather(lat, lon, form.season)
      const region = p?.region && meta.regions.includes(p.region) ? p.region : null
      setForm({ ...form, temperature_c: String(w.temperature_c), humidity_pct: String(Math.round(w.humidity_pct)), rainfall_mm: String(Math.round(w.rainfall_mm)), ...(region ? { region } : {}) })
      setPlace(p)
      toast('success', t('form.weather_filled', { basis: w.basis, mm: Math.round(w.rainfall_mm) }) + (region ? ` ${t('form.filled_region', { region: t(`opt.${region}`, region) })}` : ''))
    } catch (e) {
      toast('error', e instanceof ApiError ? e.message : t('form.weather_fail'))
    } finally { setLocating(false) }
  }

  const Section = ({ title, children }: { title: string; children: React.ReactNode }) => (
    <fieldset className="panel p-5 sm:p-6">
      <legend className="px-2 font-display text-lg font-semibold text-brand-strong">{title}</legend>
      <div className="mt-2 grid gap-x-5 gap-y-1 sm:grid-cols-2 lg:grid-cols-3">{children}</div>
    </fieldset>
  )

  return (
    <div className="space-y-5">
      <Section title={t('form.s_field')}>
        {!hideCrop && <SelField label={t('form.crop')} value={form.crop} options={meta.crops.map((c) => c.name)} onChange={setCrop} error={!!errors.crop} />}
        <SelField label={t('form.region')} value={form.region} options={meta.regions} onChange={set('region')} />
        <SelField label={t('form.season')} value={form.season} options={meta.seasons} onChange={set('season')} disabledOptions={meta.seasons.filter((s) => !allowed.includes(s))} />
        <NumField k="area_ha" label={t('form.area_ha')} value={form.area_ha} onChange={set('area_ha')} errors={errors} />
        <NumField k="year" label={t('form.year')} value={form.year} onChange={set('year')} errors={errors} />
      </Section>
      <Section title={t('form.s_soil')}>
        <SelField label={t('form.soil_type')} value={form.soil_type} options={meta.soil_types} onChange={set('soil_type')} />
        <NumField k="ph" label={t('form.ph')} value={form.ph} onChange={set('ph')} errors={errors} />
        <span className="hidden lg:block" />
        <NumField k="nitrogen" label={t('form.nitrogen')} value={form.nitrogen} onChange={set('nitrogen')} errors={errors} />
        <NumField k="phosphorus" label={t('form.phosphorus')} value={form.phosphorus} onChange={set('phosphorus')} errors={errors} />
        <NumField k="potassium" label={t('form.potassium')} value={form.potassium} onChange={set('potassium')} errors={errors} />
      </Section>
      <Section title={t('form.s_water')}>
        <div className="sm:col-span-2 lg:col-span-3">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <button type="button" className="btn btn-outline" onClick={() => setPicker({ auto: true })} disabled={locating}>
              {locating ? <Spinner /> : <Icon name="pin" />} {locating ? t('form.locating') : t('form.use_location')}
            </button>
            <button type="button" className="btn btn-ghost" onClick={() => setPicker({ auto: false })} disabled={locating}>{t('form.choose_map')}</button>
            {place && <span className="rounded-full bg-brand-soft px-3 py-1 text-sm font-semibold text-brand-strong" title={place.display_name}><Icon name="pin" className="mr-1 inline h-4 w-4" />{place.label}</span>}
          </div>
        </div>
        <NumField k="rainfall_mm" label={t('form.rainfall_mm')} value={form.rainfall_mm} onChange={set('rainfall_mm')} errors={errors} hint={t('form.rain_hint')} />
        <NumField k="temperature_c" label={t('form.temperature_c')} value={form.temperature_c} onChange={set('temperature_c')} errors={errors} />
        <NumField k="humidity_pct" label={t('form.humidity_pct')} value={form.humidity_pct} onChange={set('humidity_pct')} errors={errors} />
        <SelField label={t('form.irrigation')} value={form.irrigation} options={meta.irrigation} onChange={set('irrigation')} />
        <NumField k="fertilizer_kg_ha" label={t('form.fertilizer_kg_ha')} value={form.fertilizer_kg_ha} onChange={set('fertilizer_kg_ha')} errors={errors} />
      </Section>
      {picker && <LocationPicker initial={place ? { lat: place.lat, lon: place.lon } : null} autoLocate={picker.auto} onClose={() => setPicker(null)} onConfirm={applyLocation} />}
    </div>
  )
}