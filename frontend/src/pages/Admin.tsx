import { useTranslation } from 'react-i18next'
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api'
import { ErrorBox, PageTitle, Skeleton } from '../components/ui'
import { useFetch } from '../hooks/useFetch'
import { useTheme } from '../theme'
import { cssColor, dateShort, fmt } from '../util'

export default function Admin() {
  const { t, i18n } = useTranslation()
  useTheme()
  const stats = useFetch(api.adminStats)
  const users = useFetch(api.adminUsers)
  const s = stats.data
  const tip = { background: cssColor('surface'), border: `1px solid ${cssColor('line')}`, borderRadius: 8 }
  return (
    <>
      <PageTitle title={t('admin.title')} />
      {stats.error && <ErrorBox message={stats.error} onRetry={stats.reload} />}
      <dl className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {[['users', s?.users], ['predictions', s?.predictions], ['requests', s?.requests], ['errors', s?.server_errors]].map(([k, v]) => (
          <div key={String(k)} className="panel p-5"><dt className="text-sm text-muted">{t(`admin.${k}`)}</dt><dd className="font-display text-3xl font-bold text-brand-strong">{stats.loading ? <Skeleton className="h-9 w-16" /> : v}</dd></div>))}
      </dl>
      {s && (
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <section className="panel p-5"><h2 className="mb-2 text-lg font-bold">{t('admin.per_day')}</h2>
            <div className="h-64"><ResponsiveContainer><LineChart data={s.predictions_per_day.map((d) => ({ ...d, label: d.date.slice(5) }))}>
              <CartesianGrid stroke={cssColor('line')} strokeDasharray="3 3" /><XAxis dataKey="label" stroke={cssColor('muted')} /><YAxis allowDecimals={false} stroke={cssColor('muted')} />
              <Tooltip contentStyle={tip} /><Line dataKey="count" stroke={cssColor('brand')} strokeWidth={2.5} dot={{ r: 3 }} /></LineChart></ResponsiveContainer></div></section>
          <section className="panel p-5"><h2 className="mb-2 text-lg font-bold">{t('admin.by_crop')}</h2>
            <div className="h-64"><ResponsiveContainer><BarChart data={s.by_crop}>
              <CartesianGrid stroke={cssColor('line')} strokeDasharray="3 3" /><XAxis dataKey="crop" stroke={cssColor('muted')} tick={{ fontSize: 11 }} /><YAxis allowDecimals={false} stroke={cssColor('muted')} />
              <Tooltip contentStyle={tip} /><Bar dataKey="count" fill={cssColor('marigold')} radius={4} /></BarChart></ResponsiveContainer></div></section>
          <section className="panel overflow-x-auto p-5 lg:col-span-2"><h2 className="mb-2 text-lg font-bold">{t('admin.endpoints')}</h2>
            <table className="w-full text-left text-sm"><thead className="text-muted"><tr><th className="py-2">{t('admin.path')}</th><th>{t('admin.count')}</th><th>{t('admin.latency')}</th></tr></thead>
              <tbody>{s.endpoints.map((e) => <tr key={e.path} className="border-t border-line"><td className="py-2 font-mono text-xs">{e.path}</td><td>{e.count}</td><td>{fmt(e.avg_latency_ms, 0)}</td></tr>)}</tbody></table></section>
        </div>
      )}
      <section className="panel mt-6 overflow-x-auto p-5"><h2 className="mb-2 text-lg font-bold">{t('admin.users_table')}</h2>
        {users.loading ? <Skeleton className="h-24" /> : (
          <table className="w-full text-left text-sm"><thead className="text-muted"><tr>{['name', 'email', 'role', 'predictions', 'joined'].map((h) => <th key={h} className="py-2">{t(`admin.${h}`)}</th>)}</tr></thead>
            <tbody>{users.data?.map((u) => <tr key={u.id} className="border-t border-line"><td className="py-2">{u.name}</td><td>{u.email}</td><td>{u.role}</td><td>{u.predictions}</td><td>{dateShort(u.created_at, i18n.language)}</td></tr>)}</tbody></table>)}
      </section>
    </>
  )
}
