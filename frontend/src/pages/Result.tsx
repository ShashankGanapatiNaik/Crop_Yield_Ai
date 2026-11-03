import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { api } from '../api'
import RecoCard from '../components/RecoCard'
import ShapChart from '../components/ShapChart'
import DataBadge from '../components/DataBadge'
import YieldRange from '../components/YieldRange'
import { ErrorBox, Icon, Skeleton, Spinner } from '../components/ui'
import { useFetch } from '../hooks/useFetch'
import { lastInputs } from '../ranges'
import { useToast } from '../toast'
import { dateShort, fmt, inr } from '../util'

export default function Result() {
  const { id } = useParams()
  const { t, i18n } = useTranslation()
  const toast = useToast()
  const nav = useNavigate()
  const { data: p, error, loading, reload } = useFetch(() => api.historyItem(Number(id)), [id])
  const [pdfBusy, setPdfBusy] = useState(false)

  if (loading) return <div className="space-y-4"><Skeleton className="h-10 w-1/2" /><Skeleton className="h-40" /><Skeleton className="h-72" /></div>
  if (error || !p) return <ErrorBox message={error ?? t('result.not_found')} onRetry={reload} />

  const go = (to: string) => { lastInputs.save(p.inputs); nav(to) }
  const pdf = async () => {
    setPdfBusy(true)
    try { await api.downloadReport(p.id); toast('success', t('result.pdf_ok')) } catch { toast('error', t('common.error')) } finally { setPdfBusy(false) }
  }
  const e = p.economics

  return (
    <div className="space-y-8">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold text-brand-strong sm:text-4xl">{t(`opt.${p.crop}`, p.crop)} · {t(`opt.${p.inputs.region}`, p.inputs.region)}</h1>
          <p className="text-sm text-muted">{t(`opt.${p.inputs.season}`, p.inputs.season)} · {fmt(p.inputs.area_ha, 1)} {t('common.ha')} · {dateShort(p.created_at, i18n.language)} · {t('result.model', { v: p.model_version })}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button className="btn btn-outline" onClick={() => go('/whatif')}>{t('result.whatif')}</button>
          <button className="btn btn-outline" onClick={() => go('/compare')}>{t('result.compare')}</button>
          <button className="btn btn-primary" onClick={pdf} disabled={pdfBusy}>{pdfBusy ? <Spinner /> : <Icon name="down" />} {t('result.pdf')}</button>
        </div>
      </header>

      <DataBadge mode={p.data_mode ?? 'synthetic'} compact />
      {(p.data_notes?.length ?? 0) > 0 && (
        <ul className="space-y-1 rounded-xl bg-sunken p-4 text-sm" aria-label="Data notes">
          {p.estimated_inputs && p.estimated_inputs.length > 0 && <li className="font-semibold">{t('data.estimated_from', { fields: p.estimated_inputs.join(', ') })}</li>}
          {p.data_notes!.map((n, i) => <li key={i} className="text-muted">{n}</li>)}
        </ul>
      )}

      <section className="panel p-6 sm:p-8" aria-labelledby="yl">
        <div className="grid gap-8 md:grid-cols-[auto_1fr] md:items-center">
          <div>
            <h2 id="yl" className="text-sm font-semibold text-muted">{t('result.yield')}</h2>
            <p className="font-display text-6xl font-bold text-brand-strong">{fmt(p.predicted_yield_t_ha)}<span className="ml-2 text-xl text-muted">{t('common.tha')}</span></p>
            <p className="mt-1 text-muted">{t('result.range')}: <b className="text-ink">{fmt(p.lower_t_ha)} – {fmt(p.upper_t_ha)}</b></p>
          </div>
          <YieldRange lo={p.lower_t_ha} point={p.predicted_yield_t_ha} hi={p.upper_t_ha} />
        </div>
        <dl className="mt-8 grid grid-cols-2 gap-4 border-t border-line pt-5 md:grid-cols-4">
          {[[t('result.production'), `${fmt(p.total_production_t, 1)} t`], [t('result.revenue'), inr(e.revenue)], [t('result.cost'), inr(e.cost)], [t('result.profit'), inr(e.profit)]].map(([k, v], i) => (
            <div key={k}><dt className="text-sm text-muted">{k}</dt><dd className={`text-xl font-bold ${i === 3 ? (e.profit >= 0 ? 'text-good' : 'text-danger') : ''}`}>{v}</dd></div>
          ))}
        </dl>
      </section>

      <section aria-labelledby="dr">
        <h2 id="dr" className="text-2xl font-bold">{t('result.drivers')}</h2>
        <p className="mb-2 text-sm text-muted">{t('result.drivers_note', { base: fmt(p.base_value_t_ha) })}</p>
        <div className="panel p-4"><ShapChart factors={p.top_factors} /></div>
      </section>

      <section aria-labelledby="rc">
        <h2 id="rc" className="mb-3 text-2xl font-bold">{t('result.recs')}</h2>
        <div className="grid gap-3 md:grid-cols-2">{p.recommendations.map((r, i) => <RecoCard key={i} r={r} />)}</div>
      </section>
      <p><Link to="/history" className="text-brand underline">← {t('nav.history')}</Link></p>
    </div>
  )
}