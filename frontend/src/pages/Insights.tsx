import { useTranslation } from 'react-i18next'
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, API_BASE } from '../api'
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
  const pc = Object.entries(m.per_crop_metrics).map(([c, v]) => ({ crop: t(`opt.${c}`, c).split(' (')[0], r2: Number(v.r2.toFixed(3)), mae: Number(v.mae.toFixed(3)) }))
  const u = m.uncertainty

  return (
    <div className="space-y-8">
      <PageTitle title={t('insights.title')} sub={t('insights.subtitle')} />
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
          <YAxis yAxisId="a" domain={[0.6, 1]} stroke={cssColor('muted')} /><YAxis yAxisId="b" orientation="right" stroke={cssColor('muted')} />
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
            <XAxis type="number" domain={[0.6, 1]} stroke={cssColor('muted')} /><YAxis type="category" dataKey="crop" width={90} tick={{ fontSize: 11, fill: cssColor('ink') }} stroke={cssColor('muted')} />
            <Tooltip contentStyle={tip} /><Bar dataKey="r2" name="R²" fill={cssColor('marigold')} radius={4} /></BarChart></ResponsiveContainer></div></section>
      </div>

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
