import { useTranslation } from 'react-i18next'
import type { Recommendation, Severity } from '../types'
import { Icon } from './ui'

const STYLE: Record<Severity, { bar: string; text: string; icon: string }> = {
  critical: { bar: 'border-danger', text: 'text-danger', icon: 'alert' },
  warning: { bar: 'border-warn', text: 'text-warn', icon: 'alert' },
  info: { bar: 'border-info', text: 'text-info', icon: 'info' },
  success: { bar: 'border-good', text: 'text-good', icon: 'check' },
}
const CAT_ICON: Record<string, string> = { irrigation: 'drop', soil_ph: 'leaf', nitrogen: 'leaf', phosphorus: 'leaf', potassium: 'leaf', climate: 'sun', crop: 'leaf', economics: 'info' }

export default function RecoCard({ r }: { r: Recommendation }) {
  const { t } = useTranslation()
  const s = STYLE[r.severity]
  return (
    <article className={`panel border-l-4 ${s.bar} p-4`}>
      <header className={`flex items-center gap-2 text-sm font-semibold ${s.text}`}>
        <Icon name={s.icon} className="h-4 w-4" /> {t(`sev.${r.severity}`)}
        <Icon name={CAT_ICON[r.category] ?? 'leaf'} className="ml-auto h-4 w-4 opacity-60" />
      </header>
      <h3 className="mt-1 text-base font-bold">{r.title}</h3>
      <p className="mt-1 text-sm text-muted">{r.detail}</p>
      {r.expected_gain_t_ha != null && (
        <p className="mt-2 inline-block rounded-full bg-brand-soft px-3 py-1 text-xs font-semibold text-brand-strong">{t('result.gain', { g: r.expected_gain_t_ha.toFixed(2) })}</p>
      )}
    </article>
  )
}
