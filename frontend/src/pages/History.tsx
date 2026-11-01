import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { api } from '../api'
import { ErrorBox, Icon, PageTitle, Skeleton } from '../components/ui'
import { useFetch } from '../hooks/useFetch'
import { useToast } from '../toast'
import { dateShort, fmt, inr } from '../util'

export default function History() {
  const { t, i18n } = useTranslation()
  const toast = useToast()
  const { data, error, loading, reload } = useFetch(api.history)

  const del = async (id: number) => {
    if (!window.confirm(t('history.confirm'))) return
    try { await api.deleteHistory(id); toast('success', t('history.deleted')); reload() } catch { toast('error', t('common.error')) }
  }
  const pdf = (id: number) => api.downloadReport(id).catch(() => toast('error', t('common.error')))

  return (
    <>
      <PageTitle title={t('history.title')} />
      {error && <ErrorBox message={error} onRetry={reload} />}
      {loading && <div className="space-y-3"><Skeleton className="h-16" /><Skeleton className="h-16" /><Skeleton className="h-16" /></div>}
      {data && data.items.length === 0 && (
        <div className="panel p-10 text-center"><p className="text-muted">{t('history.empty')}</p><Link to="/predict" className="btn btn-primary mt-4">{t('history.empty_cta')}</Link></div>
      )}
      {data && data.items.length > 0 && (
        <ul className="space-y-3">
          {data.items.map((i) => (
            <li key={i.id} className="panel flex flex-wrap items-center gap-x-6 gap-y-2 p-4">
              <Link to={`/result/${i.id}`} className="min-w-40 flex-1">
                <span className="block text-lg font-bold text-brand-strong">{t(`opt.${i.crop}`, i.crop)}</span>
                <span className="text-sm text-muted">{dateShort(i.created_at, i18n.language)} · {fmt(i.area_ha, 1)} {t('common.ha')}</span>
              </Link>
              <div><span className="block text-xs text-muted">{t('result.yield')}</span><b>{fmt(i.predicted_yield_t_ha)}</b> <span className="text-sm text-muted">{t('common.tha')} ({fmt(i.lower_t_ha)}–{fmt(i.upper_t_ha)})</span></div>
              <div><span className="block text-xs text-muted">{t('result.profit')}</span><b className={i.profit >= 0 ? 'text-good' : 'text-danger'}>{inr(i.profit)}</b></div>
              <div className="ml-auto flex gap-1">
                <Link to={`/result/${i.id}`} className="btn btn-ghost !px-3">{t('history.open')}</Link>
                <button className="btn btn-ghost !px-3" onClick={() => pdf(i.id)} aria-label={`${t('history.pdf')} #${i.id}`}><Icon name="down" /></button>
                <button className="btn btn-ghost !px-3 text-danger" onClick={() => del(i.id)} aria-label={`${t('common.delete')} #${i.id}`}><Icon name="trash" /></button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </>
  )
}
