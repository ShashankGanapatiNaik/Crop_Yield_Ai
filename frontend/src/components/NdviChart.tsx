import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useTranslation } from 'react-i18next'
import { useTheme } from '../theme'
import type { NdviObs } from '../types'
import { cssColor, dateShort, fmt } from '../util'

interface Props { observations: NdviObs[]; low: number; high: number }

export default function NdviChart({ observations, low, high }: Props) {
    const { t, i18n } = useTranslation()
    useTheme() // re-read CSS colours when the theme changes
    const data = observations.map((o) => ({ ...o, label: dateShort(o.date, i18n.language) }))
    const min = Math.min(0, ...observations.map((o) => o.ndvi_mean))
    const ink = cssColor('ink'), muted = cssColor('muted'), line = cssColor('line')
    const dot = (p: { cx?: number; cy?: number; payload?: NdviObs; index?: number }) => (
        <circle key={p.index} cx={p.cx} cy={p.cy} r={5} fill={p.payload?.color ?? muted} stroke={cssColor('surface')} strokeWidth={2} />
    )
    return (
        <div className="h-64 w-full" role="img" aria-label={data.map((d) => `${d.label}: ${fmt(d.ndvi_mean)}`).join(', ')}>
            <ResponsiveContainer>
                <LineChart data={data} margin={{ left: 0, right: 16, top: 10, bottom: 4 }}>
                    <CartesianGrid stroke={line} strokeDasharray="3 3" />
                    <XAxis dataKey="label" stroke={muted} tick={{ fill: muted, fontSize: 12 }} />
                    <YAxis domain={[min, 1]} stroke={muted} tick={{ fill: muted, fontSize: 12 }} tickFormatter={(v: number) => fmt(v, 1)} width={38} />
                    <ReferenceLine y={low} stroke="#dc2626" strokeDasharray="5 4" />
                    <ReferenceLine y={high} stroke="#16a34a" strokeDasharray="5 4" />
                    <Tooltip contentStyle={{ background: cssColor('surface'), border: `1px solid ${line}`, borderRadius: 8, color: ink }} labelStyle={{ color: ink }}
                        formatter={(v: number, _n: string, item) => [`${fmt(v)} · ${t('health.valid_pct', { p: Math.round(((item?.payload as NdviObs)?.valid_fraction ?? 0) * 100) })}`, t('health.ndvi')]} />
                    <Line type="monotone" dataKey="ndvi_mean" stroke={cssColor('brand')} strokeWidth={2.5} dot={dot} activeDot={{ r: 7 }} isAnimationActive={false} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    )
}