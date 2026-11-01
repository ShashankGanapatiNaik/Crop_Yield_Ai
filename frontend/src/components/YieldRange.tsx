import { useTranslation } from 'react-i18next'
import { fmt } from '../util'

/** The "furrow ruler": a measuring bar with the 80% band and a marigold marker for the point estimate. */
export default function YieldRange({ lo, point, hi, max }: { lo: number; point: number; hi: number; max?: number }) {
  const { t } = useTranslation()
  const top = max ?? Math.max(hi * 1.25, point * 1.4, 1)
  const pct = (v: number) => `${Math.min(100, Math.max(0, (v / top) * 100))}%`
  return (
    <figure aria-label={`${t('result.range')}: ${fmt(lo)} – ${fmt(hi)} ${t('common.tha')}`} className="select-none">
      <div className="relative h-14">
        <div className="absolute inset-x-0 top-6 h-3 rounded-full bg-sunken" style={{ backgroundImage: 'repeating-linear-gradient(90deg, transparent 0 calc(10% - 1px), rgb(var(--line)) calc(10% - 1px) 10%)' }} />
        <div className="absolute top-6 h-3 rounded-full bg-brand/45" style={{ left: pct(lo), width: `calc(${pct(hi)} - ${pct(lo)})` }} />
        <div className="absolute top-3 h-9 w-1.5 -translate-x-1/2 rounded-full bg-marigold ring-2 ring-surface" style={{ left: pct(point) }} />
        <span className="absolute top-0 -translate-x-1/2 whitespace-nowrap text-xs font-semibold" style={{ left: pct(point) }}>{fmt(point)}</span>
      </div>
      <div className="relative mt-1 h-5 text-xs text-muted">
        <span className="absolute -translate-x-1/2" style={{ left: pct(lo) }}>{fmt(lo)}</span>
        <span className="absolute -translate-x-1/2" style={{ left: pct(hi) }}>{fmt(hi)}</span>
      </div>
      <figcaption className="sr-only">{fmt(point)} {t('common.tha')}</figcaption>
    </figure>
  )
}
