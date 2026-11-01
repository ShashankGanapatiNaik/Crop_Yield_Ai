import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api'
import { useAuth } from '../auth'
import { ErrorBox, Icon, PageTitle, Skeleton } from '../components/ui'
import { useFetch } from '../hooks/useFetch'
import { useTheme } from '../theme'
import { cssColor, dateShort, fmt } from '../util'

export default function Dashboard() {
  const { t, i18n } = useTranslation()
  const { user } = useAuth()
  useTheme()
  const { data, error, loading, reload } = useFetch(api.history)
  const items = data?.items ?? []
  const avg = items.length ? items.reduce((s, i) => s + i.predicted_yield_t_ha, 0) / items.length : 0
  const counts = items.reduce<Record<string, number>>((m, i) => ({ ...m, [i.crop]: (m[i.crop] ?? 0) + 1 }), {})
  const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0]
  const series = [...items].reverse().map((i, n) => ({ n: n + 1, d: dateShort(i.created_at, i.created_at && i18n.language), y: i.predicted_yield_t_ha }))

  return (
    <>
      <PageTitle title={t('dash.hello', { name: user?.name.split(' ')[0] })} sub={t('dash.sub')}>
        <Link to="/predict" className="btn btn-primary"><Icon name="leaf" /> {t('dash.new')}</Link>
      </PageTitle>
      {error && <ErrorBox message={error} onRetry={reload} />}
      <dl className="grid gap-4 sm:grid-cols-3">
        {[[t('dash.total'), data?.total ?? 0], [t('dash.avg'), items.length ? `${fmt(avg)} ${t('common.tha')}` : t('dash.none')], [t('dash.top'), top ? t(`opt.${top}`, top) : t('dash.none')]].map(([k, v]) => (
          <div key={String(k)} className="panel p-5"><dt className="text-sm text-muted">{k}</dt><dd className="mt-1 font-display text-3xl font-bold text-brand-strong">{loading ? <Skeleton className="h-9 w-24" /> : v}</dd></div>
        ))}
      </dl>
      <div className="mt-6 grid gap-6 lg:grid-cols-5">
        <section className="panel p-5 lg:col-span-3" aria-labelledby="tr">
          <h2 id="tr" className="mb-3 text-lg font-bold">{t('dash.trend')}</h2>
          {loading ? <Skeleton className="h-60" /> : series.length < 2 ? <p className="py-16 text-center text-muted">{t('history.empty')}</p> : (
            <div className="h-60"><ResponsiveContainer><LineChart data={series}>
              <CartesianGrid stroke={cssColor('line')} strokeDasharray="3 3" />
              <XAxis dataKey="n" stroke={cssColor('muted')} /><YAxis stroke={cssColor('muted')} unit="" />
              <Tooltip contentStyle={{ background: cssColor('surface'), border: `1px solid ${cssColor('line')}`, borderRadius: 8 }} formatter={(v: number) => [`${fmt(v)} ${t('common.tha')}`, '']} />
              <Line type="monotone" dataKey="y" stroke={cssColor('brand')} strokeWidth={2.5} dot={{ r: 3, fill: cssColor('marigold') }} />
            </LineChart></ResponsiveContainer></div>
          )}
        </section>
        <section className="panel p-5 lg:col-span-2" aria-labelledby="rp">
          <h2 id="rp" className="mb-3 text-lg font-bold">{t('dash.recent')}</h2>
          {loading ? <div className="space-y-2"><Skeleton className="h-10" /><Skeleton className="h-10" /><Skeleton className="h-10" /></div> : items.length === 0 ? (
            <Link to="/predict" className="text-brand underline">{t('history.empty_cta')}</Link>
          ) : (
            <ul className="divide-y divide-line">{items.slice(0, 6).map((i) => (
              <li key={i.id}><Link to={`/result/${i.id}`} className="flex items-center justify-between gap-2 py-2.5 hover:text-brand">
                <span className="font-semibold">{t(`opt.${i.crop}`, i.crop)}</span><span className="text-sm text-muted">{fmt(i.predicted_yield_t_ha)} {t('common.tha')}</span>
              </Link></li>))}</ul>
          )}
        </section>
      </div>
    </>
  )
}
