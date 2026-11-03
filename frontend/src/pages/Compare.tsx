import { useState, type FormEvent } from 'react'
import { useTranslation } from 'react-i18next'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, ApiError } from '../api'
import ConditionsForm from '../components/ConditionsForm'
import YieldRange from '../components/YieldRange'
import DataBadge from '../components/DataBadge'
import { ErrorBox, PageTitle, Skeleton, Spinner } from '../components/ui'
import { useFetch } from '../hooks/useFetch'
import { DEFAULT_FORM, fromInputs, lastInputs, toInputs, unusedKeys, validate, type Errors } from '../ranges'
import { useTheme } from '../theme'
import { useToast } from '../toast'
import type { CompareRow, FormState } from '../types'
import { cssColor, fmt, inr } from '../util'

export default function Compare() {
  const { t } = useTranslation()
  useTheme()
  const toast = useToast()
  const meta = useFetch(api.crops)
  const [form, setForm] = useState<FormState>(() => { const l = lastInputs.load(); return l ? fromInputs(l) : DEFAULT_FORM })
  const [errors, setErrors] = useState<Errors>({})
  const [busy, setBusy] = useState(false)
  const [rows, setRows] = useState<CompareRow[] | null>(null)
  const [best, setBest] = useState('')
  const [picked, setPicked] = useState<string[]>([])

  const run = async (e: FormEvent) => {
    e.preventDefault()
    const v = validate(form, unusedKeys(meta.data?.input_schema?.used))
    setErrors(v)
    if (Object.keys(v).length) return toast('error', t('form.fix'))
    setBusy(true)
    try {
      const inputs = toInputs(form, meta.data?.input_schema?.used)
      const { crop: _c, ...rest } = inputs
      void _c
      const r = await api.compare(rest)
      setRows(r.results); setBest(r.best_crop); setPicked(r.results.slice(0, 3).map((x) => x.crop))
    } catch (err) { toast('error', err instanceof ApiError ? err.message : t('common.error')) } finally { setBusy(false) }
  }
  const toggle = (c: string) => setPicked((p) => (p.includes(c) ? p.filter((x) => x !== c) : p.length >= 4 ? p : [...p, c]))
  const chart = rows?.map((r) => ({ ...r, name: t(`opt.${r.crop}`, r.crop).split(' (')[0] }))
  const tipStyle = { background: cssColor('surface'), border: `1px solid ${cssColor('line')}`, borderRadius: 8 }

  return (
    <>
      <PageTitle title={t('compare.title')} sub={t('compare.subtitle')} />
      {meta.data && <div className="mb-5"><DataBadge mode={meta.data.data_mode} compact /></div>}
      {meta.error && <ErrorBox message={meta.error} onRetry={meta.reload} />}
      {meta.data && (
        <form onSubmit={run} noValidate className="mb-8">
          <ConditionsForm form={form} setForm={setForm} errors={errors} meta={meta.data} hideCrop />
          <button className="btn btn-primary mt-5" disabled={busy}>{busy && <Spinner />} {busy ? t('compare.running') : t('compare.run')}</button>
        </form>
      )}
      {busy && !rows && <Skeleton className="h-64" />}
      {rows && chart && (
        <div className="space-y-8">
          <div className="grid gap-6 lg:grid-cols-2">
            {([['profit', 'chart_profit'], ['yield_t_ha', 'chart_yield']] as const).map(([key, title]) => (
              <section key={key} className="panel p-5" aria-label={t(`compare.${title}`)}>
                <h2 className="mb-2 text-lg font-bold">{t(`compare.${title}`)}</h2>
                <div className="h-72"><ResponsiveContainer><BarChart data={chart} layout="vertical" margin={{ left: 10, right: 16 }}>
                  <CartesianGrid stroke={cssColor('line')} strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" stroke={cssColor('muted')} tickFormatter={(v: number) => (key === 'profit' ? `${Math.round(v / 1000)}k` : String(v))} />
                  <YAxis type="category" dataKey="name" width={86} stroke={cssColor('muted')} tick={{ fill: cssColor('ink'), fontSize: 12 }} />
                  <Tooltip contentStyle={tipStyle} formatter={(v: number) => [key === 'profit' ? inr(v) : `${fmt(v)} ${t('common.tha')}`, '']} />
                  <Bar dataKey={key} radius={4}>
                    {chart.map((r) => <Cell key={r.crop} fill={r.crop === best ? cssColor('marigold') : r.season_suitable ? cssColor('brand') : cssColor('muted')} fillOpacity={r.season_suitable ? 1 : 0.5} />)}
                  </Bar>
                </BarChart></ResponsiveContainer></div>
              </section>
            ))}
          </div>

          <section className="panel overflow-x-auto" aria-label="Table">
            <table className="w-full min-w-[560px] text-left text-sm">
              <thead className="border-b border-line text-muted"><tr>{['crop', 'yield', 'production', 'profit'].map((h) => <th key={h} className="p-3 font-semibold">{h === 'crop' ? t('form.crop') : t(`compare.${h}`)}</th>)}<th className="p-3" /></tr></thead>
              <tbody>{rows.map((r) => (
                <tr key={r.crop} className="border-b border-line last:border-0">
                  <td className="p-3 font-semibold">{t(`opt.${r.crop}`, r.crop)} {r.crop === best && <span className="ml-1 rounded-full bg-marigold/25 px-2 py-0.5 text-xs">{t('compare.best')}</span>}
                    {!r.season_suitable && <span className="block text-xs font-normal text-warn">{t('compare.off_season', { season: t(`opt.${form.season}`, form.season) })}</span>}</td>
                  <td className="p-3">{fmt(r.yield_t_ha)} <span className="text-muted">({fmt(r.lower_t_ha)}–{fmt(r.upper_t_ha)})</span></td>
                  <td className="p-3">{fmt(r.total_production_t, 1)} t</td>
                  <td className={`p-3 font-bold ${r.profit >= 0 ? 'text-good' : 'text-danger'}`}>{inr(r.profit)}</td>
                  <td className="p-3"><button type="button" onClick={() => toggle(r.crop)} aria-pressed={picked.includes(r.crop)} className={`btn !min-h-9 !px-3 text-xs ${picked.includes(r.crop) ? 'btn-primary' : 'btn-outline'}`}>{t('compare.side')}</button></td>
                </tr>))}</tbody>
            </table>
          </section>

          <section aria-label={t('compare.side')}>
            <h2 className="mb-1 text-2xl font-bold">{t('compare.side')}</h2><p className="mb-3 text-sm text-muted">{t('compare.pick')}</p>
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              {rows.filter((r) => picked.includes(r.crop)).map((r) => (
                <article key={r.crop} className={`panel p-5 ${r.crop === best ? 'ring-2 ring-marigold' : ''}`}>
                  <h3 className="text-xl font-bold text-brand-strong">{t(`opt.${r.crop}`, r.crop)}</h3>
                  <p className="mt-2 font-display text-4xl font-bold">{fmt(r.yield_t_ha)}<span className="ml-1 text-base text-muted">{t('common.tha')}</span></p>
                  <div className="my-3"><YieldRange lo={r.lower_t_ha} point={r.yield_t_ha} hi={r.upper_t_ha} max={Math.max(...rows.map((x) => x.upper_t_ha)) * 1.1} /></div>
                  <p className="text-sm text-muted">{t('compare.profit')}</p><p className={`text-xl font-bold ${r.profit >= 0 ? 'text-good' : 'text-danger'}`}>{inr(r.profit)}</p>
                  <p className="text-xs text-muted">ROI {fmt(r.roi_pct, 0)}% · {fmt(r.total_production_t, 1)} t</p>
                </article>))}
            </div>
          </section>
        </div>
      )}
    </>
  )
}