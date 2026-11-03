import { useTranslation } from 'react-i18next'
import type { DataMode } from '../types'
import { Icon } from './ui'

/** Says plainly what the model was trained on, so nobody mistakes a prototype for field evidence (or vice versa). */
export default function DataBadge({ mode, compact }: { mode?: DataMode | null; compact?: boolean }) {
    const { t } = useTranslation()
    const real = mode === 'real'
    const cls = real ? 'border-good/50 bg-good/10 text-good' : 'border-warn/50 bg-warn/10 text-warn'
    return (
        <aside className={`rounded-xl border px-4 py-3 ${cls}`} role="note" aria-label={t(real ? 'data.real_title' : 'data.synthetic_title')}>
            <p className="flex items-center gap-2 font-semibold"><Icon name={real ? 'check' : 'alert'} className="h-4 w-4" />{t(real ? 'data.real_title' : 'data.synthetic_title')}</p>
            {!compact && <p className="mt-1 text-sm text-ink">{t(real ? 'data.real_text' : 'data.synthetic_text')}</p>}
        </aside>
    )
}

export function ProvenanceLegend({ real }: { real: boolean }) {
    const { t } = useTranslation()
    const rows: [string, string][] = real
        ? [['data.observed', 'data.observed_d'], ['data.estimate', 'data.estimate_d'], ['data.yours', 'data.yours_d']]
        : [['data.simulated', 'data.simulated_d'], ['data.yours', 'data.yours_d']]
    return (
        <dl className="grid gap-x-6 gap-y-1 text-sm sm:grid-cols-[auto_1fr]">
            {rows.map(([k, d]) => (<div key={k} className="contents"><dt className="font-semibold">{t(k)}</dt><dd className="text-muted">{t(d)}</dd></div>))}
        </dl>
    )
}