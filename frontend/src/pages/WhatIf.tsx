import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { Area, ComposedChart, CartesianGrid, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, ApiError } from '../api'
import YieldRange from '../components/YieldRange'
import { ErrorBox, PageTitle, SelField, Skeleton } from '../components/ui'
import { useFetch } from '../hooks/useFetch'
import { lastInputs, NUM_SPECS } from '../ranges'
import { useTheme } from '../theme'
import type { Inputs, NumKey, WhatIf as WhatIfRes } from '../types'
import { cssColor, fmt, inr } from '../util'

const SLIDERS: NumKey[] = ['rainfall_mm', 'temperature_c', 'ph', 'fertilizer_kg_ha', 'nitrogen', 'phosphorus', 'potassium', 'humidity_pct']

export default function WhatIf() {
  const { t } = useTranslation()
  useTheme()
  const base = useRef<Inputs | null>(lastInputs.load()).current
  const meta = useFetch(api.crops)
  const [vals, setVals] = useState<Inputs | null>(base)
  const [focus, setFocus] = useState<NumKey>('rainfall_mm')
  const [res, setRes] = useState<WhatIfRes | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const seq = useRef(0)

  useEffect(() => {
    if (!base || !vals) return
    const id = ++seq.current
    const timer = setTimeout(async () => {
      const changes: Record<string, number | string> = {}
      for (const k of [...SLIDERS, 'irrigation'] as (NumKey | 'irrigation')[]) if (vals[k] !== base[k]) changes[k] = vals[k]
      const s = NUM_SPECS[focus]
      try {
        const r = await api.whatIf(base, changes, { param: focus, min: s.sliderMin ?? s.min, max: s.sliderMax ?? s.max, steps: 24 })
        if (id === seq.current) { setRes(r); setErr(null) }
      } catch (e) { if (id === seq.current) setErr(e instanceof ApiError ? e.message : t('common.error')) }
    }, 250)
    return () => clearTimeout(timer)
  }, [vals, focus, base, t])

  if (!base || !vals) return (<><PageTitle title={t('whatif.title')} /><div className="panel p-8 text-center"><p className="text-muted">{t('whatif.need_base')}</p><Link to="/predict" className="btn btn-primary mt-4">{t('nav.predict')}</Link></div></>)

  const change = (k: NumKey, v: number) => { setFocus(k); setVals({ ...vals, [k]: v }) }
  const used = meta.data?.input_schema?.used
  // only fields the active model uses AND that exist in the saved inputs (they may come from a different model)
  const sliders = meta.data ? SLIDERS.filter((k) => (!used || used.includes(k)) && vals[k] !== undefined) : []
  const d = res?.delta_yield_t_ha ?? 0
  const curve = res?.sweep?.points.map((p) => ({ x: p.x, y: p.yield_t_ha, band: [p.lower_t_ha, p.upper_t_ha] as [number, number] }))

  return (
    <>
      <PageTitle title={t('whatif.title')} sub={`${t(`opt.${base.crop}`, base.crop)} · ${t('whatif.subtitle')}`}>
        <button className="btn btn-outline" onClick={() => setVals(base)}>{t('whatif.reset')}</button>
      </PageTitle>
      {err && <ErrorBox message={err} />}
      <div className="grid gap-6 lg:grid-cols-5">
        <section className="panel space-y-4 p-5 lg:col-span-2" aria-label="Controls">
          {sliders.map((k) => {
            const s = NUM_SPECS[k]
            const changed = vals[k] !== base[k]
            return (
              <div key={k}>
                <label htmlFor={`s-${k}`} className="flex justify-between text-sm font-semibold">
                  <span>{t(`form.${k}`)}</span>
                  <span className={changed ? 'text-marigold' : 'text-muted'}>{fmt(vals[k], s.step < 1 ? 1 : 0)} {s.unit}{changed && ` (${fmt(base[k], s.step < 1 ? 1 : 0)})`}</span>
                </label>
                <input id={`s-${k}`} type="range" className="mt-1 w-full" min={Math.min(s.sliderMin ?? s.min, base[k])} max={Math.max(s.sliderMax ?? s.max, base[k])} step={s.step}
                  value={vals[k]} onChange={(e) => change(k, Number(e.target.value))} onFocus={() => setFocus(k)} />
              </div>
            )
          })}
          {meta.data && (!used || used.includes('irrigation')) && <SelField label={t('form.irrigation')} value={vals.irrigation} options={meta.data.irrigation} onChange={(v) => setVals({ ...vals, irrigation: v })} />}
        </section>

        <section className="space-y-6 lg:col-span-3" aria-live="polite">
          {!res ? <Skeleton className="h-64" /> : (
            <>
              <div className="panel p-6">
                <div className="flex flex-wrap items-end justify-between gap-4">
                  <div><p className="text-sm text-muted">{t('whatif.scenario')}</p>
                    <p className="font-display text-5xl font-bold text-brand-strong">{fmt(res.scenario.yield_t_ha)}<span className="ml-2 text-lg text-muted">{t('common.tha')}</span></p></div>
                  <div className="text-right"><p className="text-sm text-muted">{t('whatif.delta')} ({t('whatif.baseline')} {fmt(res.baseline.yield_t_ha)})</p>
                    <p className={`text-2xl font-bold ${d >= 0 ? 'text-good' : 'text-danger'}`}>{d >= 0 ? '+' : ''}{fmt(d)} ({res.delta_pct >= 0 ? '+' : ''}{fmt(res.delta_pct, 1)}%)</p>
                    <p className="text-sm text-muted">{t('result.profit')}: {res.delta_profit >= 0 ? '+' : ''}{inr(res.delta_profit)}</p></div>
                </div>
                <div className="mt-4"><YieldRange lo={res.scenario.lower_t_ha} point={res.scenario.yield_t_ha} hi={res.scenario.upper_t_ha} max={Math.max(res.baseline.upper_t_ha, res.scenario.upper_t_ha) * 1.3} /></div>
              </div>
              {curve && (
                <div className="panel p-5">
                  <h2 className="text-lg font-bold">{t('whatif.curve', { param: t(`form.${focus}`) })}</h2>
                  <p className="mb-2 text-xs text-muted">{t('whatif.curve_note')}</p>
                  <div className="h-60"><ResponsiveContainer><ComposedChart data={curve} margin={{ left: 0, right: 12, top: 8 }}>
                    <CartesianGrid stroke={cssColor('line')} strokeDasharray="3 3" />
                    <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} stroke={cssColor('muted')} /><YAxis stroke={cssColor('muted')} />
                    <Tooltip contentStyle={{ background: cssColor('surface'), border: `1px solid ${cssColor('line')}`, borderRadius: 8 }} formatter={(v: number | number[]) => [Array.isArray(v) ? `${fmt(v[0])} – ${fmt(v[1])}` : fmt(v), '']} />
                    <Area dataKey="band" stroke="none" fill={cssColor('brand')} fillOpacity={0.2} isAnimationActive={false} />
                    <Line dataKey="y" stroke={cssColor('brand')} strokeWidth={2.5} dot={false} isAnimationActive={false} />
                    <ReferenceLine x={vals[focus]} stroke={cssColor('marigold')} strokeWidth={2} />
                  </ComposedChart></ResponsiveContainer></div>
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </>
  )
}