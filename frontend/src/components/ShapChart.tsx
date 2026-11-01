import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useTranslation } from 'react-i18next'
import { useTheme } from '../theme'
import type { Factor } from '../types'
import { cssColor, fmt } from '../util'

export default function ShapChart({ factors }: { factors: Factor[] }) {
  const { t } = useTranslation()
  useTheme() // re-render (and re-read CSS colours) when the theme flips
  const good = cssColor('good'), bad = cssColor('danger'), ink = cssColor('ink'), muted = cssColor('muted'), line = cssColor('line')
  const data = factors.map((f) => ({ ...f, name: `${f.label}${typeof f.value === 'number' ? ` = ${f.value}` : `: ${t(`opt.${f.value}`, String(f.value))}`}` }))
  return (
    <div className="h-72 w-full" role="img" aria-label={data.map((d) => `${d.label} ${d.shap > 0 ? '+' : ''}${fmt(d.shap, 2)}`).join(', ')}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 8, bottom: 8 }}>
          <XAxis type="number" stroke={muted} tick={{ fill: muted, fontSize: 12 }} tickFormatter={(v: number) => (v > 0 ? `+${v}` : `${v}`)} />
          <YAxis type="category" dataKey="name" width={170} stroke={muted} tick={{ fill: ink, fontSize: 12 }} />
          <ReferenceLine x={0} stroke={line} />
          <Tooltip cursor={{ fill: 'transparent' }} contentStyle={{ background: cssColor('surface'), border: `1px solid ${line}`, borderRadius: 8, color: ink }}
            formatter={(v: number) => [`${v > 0 ? '+' : ''}${fmt(v, 3)} ${t('common.tha')}`, '']} labelStyle={{ color: ink }} />
          <Bar dataKey="shap" radius={4} barSize={22}>
            {data.map((d, i) => <Cell key={i} fill={d.shap >= 0 ? good : bad} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
