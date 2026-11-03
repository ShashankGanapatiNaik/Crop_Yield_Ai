import { useTranslation } from 'react-i18next'
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, API_BASE } from '../api'
import DataBadge, { ProvenanceLegend } from '../components/DataBadge'
import { ErrorBox, PageTitle, Skeleton } from '../components/ui'
import { useFetch } from '../hooks/useFetch'
import { useTheme } from '../theme'
import { cssColor, fmt } from '../util'

const PLOTS = ['residual_analysis', 'shap_summary_beeswarm', 'prediction_intervals', 'model_comparison']

export default function Insights() {
  const { t } = useTranslation()
  useTheme()
  const { data: m, error, loading, reload } = useFetch(api.modelInfo)
  if (error) return <ErrorBox message={error} onRetry={reload} />
  if (loading || !m) return <div className="space-y-4"><Skeleton className="h-10 w-1/2" /><Skeleton className="h-32" /><Skeleton className="h-72" /></div>

  const tip = { background: cssColor('surface'), border: `1px solid ${cssColor('line')}`, borderRadius: 8 }
  const cmp = [...m.comparison].sort((a, b) => b.test_r2 - a.test_r2)
  const imp = Object.entries(m.global_importance).slice(0, 12).map(([k, v]) => ({ k, v: Number(v.toFixed(3)) })).reverse()
  const pc = Object.entries(m.per_crop_metrics).map(([c, v]) => ({ crop: t(`opt.${c}`, c).split(' (')[0], r2: Number((v.r2 ?? 0).toFixed(3)), mae: Number(v.mae.toFixed(3)) }))
  const u = m.uncertainty

  return (
    <div className="space-y-8">
      <PageTitle title={t('insights.title')} sub={t('insights.subtitle')} />
      <DataBadge mode={m.data_mode ?? 'synthetic'} />
      <section className="panel p-5"><ProvenanceLegend real={m.data_mode === 'real'} />
        {m.split && <p className="mt-3 text-sm text-muted">{t('data.split', { train: `${m.split.train_years[0]}-${m.split.train_years.at(-1)}`, val: m.split.val_years.join(', '), test: m.split.test_years.join(', ') })}</p>}
        {m.provenance?.statement && <p className="mt-2 text-sm text-muted">{m.provenance.statement}</p>}</section>
      <section className="panel p-5">
        <p className="text-sm text-muted">{t('insights.best')} · v{m.version} · {t('insights.source')}: {m.data_source}</p>
        <h2 className="text-2xl font-bold text-brand-strong">{m.best_model}</h2>
        <dl className="mt-4 grid grid-cols-2 gap-4 md:grid-cols-4">
          {[['r2', m.metrics_test.r2, 3], ['mae', m.metrics_test.mae, 3], ['rmse', m.metrics_test.rmse, 3], ['mape', m.metrics_test.mape, 1]].map(([k, v, d]) => (
            <div key={String(k)}><dt className="text-sm text-muted">{t(`insights.${k}`)}</dt><dd className="font-display text-3xl font-bold">{fmt(Number(v), Number(d))}</dd></div>))}
        </dl>
        <p className="mt-3 text-xs text-muted">n_train {m.n_train} · n_test {m.n_test} · {m.cv_folds}-fold CV · {m.optuna_trials} Optuna trials/model</p>
      </section>

      <section className="panel p-5">
        <h2 className="mb-2 text-lg font-bold">{t('insights.compare')}</h2>
        <div className="h-80"><ResponsiveContainer><BarChart data={cmp} margin={{ bottom: 40 }}>
          <CartesianGrid stroke={cssColor('line')} strokeDasharray="3 3" /><XAxis dataKey="model" interval={0} angle={-30} textAnchor="end" stroke={cssColor('muted')} tick={{ fontSize: 11, fill: cssColor('ink') }} />
          <YAxis yAxisId="a" domain={m.data_mode === 'real' ? ['auto', 'auto'] : [0.6, 1]} stroke={cssColor('muted')} /><YAxis yAxisId="b" orientation="right" stroke={cssColor('muted')} />
          <Tooltip contentStyle={tip} /><Legend verticalAlign="top" />
          <Bar yAxisId="a" dataKey="test_r2" name={t('insights.test_r2')} fill={cssColor('brand')} radius={[4, 4, 0, 0]} />
          <Bar yAxisId="b" dataKey="test_rmse" name="RMSE" fill={cssColor('marigold')} radius={[4, 4, 0, 0]} />
        </BarChart></ResponsiveContainer></div>
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="panel p-5"><h2 className="mb-2 text-lg font-bold">{t('insights.importance')}</h2>
          <div className="h-80"><ResponsiveContainer><BarChart data={imp} layout="vertical" margin={{ left: 20 }}>
            <XAxis type="number" stroke={cssColor('muted')} /><YAxis type="category" dataKey="k" width={130} tick={{ fontSize: 11, fill: cssColor('ink') }} stroke={cssColor('muted')} />
            <Tooltip contentStyle={tip} /><Bar dataKey="v" radius={4}>{imp.map((_, i) => <Cell key={i} fill={cssColor('brand')} />)}</Bar></BarChart></ResponsiveContainer></div></section>
        <section className="panel p-5"><h2 className="mb-2 text-lg font-bold">{t('insights.per_crop')}</h2>
          <div className="h-80"><ResponsiveContainer><BarChart data={pc} layout="vertical" margin={{ left: 20 }}>
            <XAxis type="number" domain={m.data_mode === 'real' ? ['auto', 'auto'] : [0.6, 1]} stroke={cssColor('muted')} /><YAxis type="category" dataKey="crop" width={90} tick={{ fontSize: 11, fill: cssColor('ink') }} stroke={cssColor('muted')} />
            <Tooltip contentStyle={tip} /><Bar dataKey="r2" name="R²" fill={cssColor('marigold')} radius={4} /></BarChart></ResponsiveContainer></div></section>
      </div>

      {m.per_region_metrics && Object.keys(m.per_region_metrics).length > 0 && (
        <section className="panel overflow-x-auto p-5"><h2 className="mb-2 text-lg font-bold">{t('data.per_region')}</h2>
          <table className="w-full min-w-[420px] text-left text-sm"><thead className="text-muted"><tr><th className="py-2">{t('form.region')}</th><th>n</th><th>R²</th><th>{t('insights.mae')}</th><th>{t('insights.rmse')}</th></tr></thead>
            <tbody>{Object.entries(m.per_region_metrics).sort((a, b) => b[1].n - a[1].n).map(([r, v]) => (
              <tr key={r} className="border-t border-line"><td className="py-2 font-semibold">{r}</td><td>{v.n}</td><td>{v.r2 == null ? '—' : fmt(v.r2, 2)}</td><td>{fmt(v.mae, 2)}</td><td>{fmt(v.rmse, 2)}</td></tr>))}</tbody></table></section>
      )}

      {m.synthetic_reference && (
        <section className="panel p-5"><h2 className="text-lg font-bold">{t('data.ref_title')}</h2>
          <p className="mt-1 text-muted">{m.synthetic_reference.best_model}: R² {fmt(m.synthetic_reference.metrics_test.r2, 3)}. {m.synthetic_reference.note}</p></section>
      )}

      <section className="panel p-5">
        <h2 className="text-lg font-bold">{t('insights.coverage')}</h2>
        <p className="mt-1 text-muted">{t('insights.cov_text', { c: fmt(u.coverage_calibrated * 100, 1), r: fmt(u.coverage_raw_quantile * 100, 1), w: fmt(u.mean_interval_width_t_ha) })}</p>
        <h2 className="mt-4 text-lg font-bold">{t('insights.residuals')}</h2>
        <p className="mt-1 text-muted">{t('insights.resid_text', { m: fmt(m.residual_summary.mean, 3), s: fmt(m.residual_summary.std, 3), k: fmt(m.residual_summary.skew, 2) })}</p>
      </section>

      <section aria-label={t('insights.plots')}>
        <h2 className="mb-3 text-2xl font-bold">{t('insights.plots')}</h2>
        <div className="grid gap-5 lg:grid-cols-2">
          {PLOTS.map((p) => (
            <figure key={p} className="panel overflow-hidden bg-white p-2">
              <img src={`${API_BASE}/model-plots/${p}`} alt={p.replace(/_/g, ' ')} loading="lazy" className="w-full" />
            </figure>))}
        </div>
      </section>
    </div>
  )
}