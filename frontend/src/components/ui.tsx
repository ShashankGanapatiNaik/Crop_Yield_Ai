import { useId, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { useToast } from '../toast'
import { parseSpokenNumber, useVoice } from '../hooks/useVoice'
import { NUM_SPECS, type Errors } from '../ranges'
import type { NumKey } from '../types'

const paths: Record<string, string> = {
  mic: 'M12 14a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0M12 16v4m-4 0h8',
  pin: 'M12 21s7-6.2 7-11a7 7 0 1 0-14 0c0 4.8 7 11 7 11zm0-8.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5z',
  sun: 'M12 4V2m0 20v-2m8-8h2M2 12h2m13.7-5.7 1.4-1.4M4.9 19.1l1.4-1.4m0-11.4L4.9 4.9m14.2 14.2-1.4-1.4M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10z',
  moon: 'M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z',
  menu: 'M4 7h16M4 12h16M4 17h16', close: 'M6 6l12 12M18 6 6 18',
  check: 'M5 12.5l4.5 4.5L19 7.5', alert: 'M12 9v4m0 4h.01M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z',
  info: 'M12 11v5m0-8h.01M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z', down: 'M12 4v12m0 0 5-5m-5 5-5-5M5 20h14',
  trash: 'M4 7h16M9 7V4h6v3m-8 0 1 13h8l1-13', arrow: 'M5 12h14m0 0-5-5m5 5-5 5', flame: 'M12 3c1 4 5 5 5 10a5 5 0 0 1-10 0c0-2 1-3 2-4 0 2 1 3 2 3 0-3-1-6 1-9z',
  drop: 'M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z', leaf: 'M5 19c0-8 5-14 15-14 0 10-6 15-14 15M5 19c3-5 6-7 10-9',
}
export function Icon({ name, className = 'h-5 w-5' }: { name: keyof typeof paths | string; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={paths[name]} />
    </svg>
  )
}

export const Skeleton = ({ className = 'h-6 w-full' }: { className?: string }) => <div className={`skeleton ${className}`} aria-hidden="true" />
export const Spinner = () => <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" aria-hidden="true" />

export function PageTitle({ title, sub, children }: { title: string; sub?: string; children?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-3xl font-bold text-brand-strong sm:text-4xl">{title}</h1>
        {sub && <p className="mt-1 max-w-2xl text-muted">{sub}</p>}
      </div>
      {children}
    </div>
  )
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  const { t } = useTranslation()
  return (
    <div role="alert" className="panel flex items-center gap-3 border-danger/40 p-4 text-danger">
      <Icon name="alert" /> <span className="flex-1">{message}</span>
      {onRetry && <button className="btn btn-outline" onClick={onRetry}>{t('common.retry')}</button>}
    </div>
  )
}

interface NumFieldProps { k: NumKey; label: string; value: string; onChange: (v: string) => void; errors?: Errors; hint?: string }
export function NumField({ k, label, value, onChange, errors, hint }: NumFieldProps) {
  const { t } = useTranslation()
  const toast = useToast()
  const id = useId()
  const { supported, listening, listen } = useVoice()
  const spec = NUM_SPECS[k]
  const err = errors?.[k]
  const msg = err === 'required' ? t('form.required') : err ? t('form.range', { min: spec.min, max: spec.max }) : undefined
  const speak = () => {
    if (!supported) return toast('info', t('form.mic_unsupported'))
    toast('info', t('form.mic_listening'))
    listen((txt) => { const n = parseSpokenNumber(txt); if (n) onChange(n); else toast('error', t('form.mic_noresult')) }, () => toast('error', t('form.mic_noresult')))
  }
  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-sm font-semibold">{label}{spec.unit && <span className="ml-1 font-normal text-muted">({spec.unit})</span>}</label>
      <div className="relative">
        <input id={id} className="field-input pr-12" type="text" inputMode="decimal" value={value} onChange={(e) => onChange(e.target.value.replace(',', '.'))}
          aria-invalid={!!err} aria-describedby={`${id}-m`} autoComplete="off" />
        <button type="button" onClick={speak} aria-label={`${t('form.mic')}: ${label}`} aria-pressed={listening}
          className={`absolute right-1 top-1/2 flex h-9 w-9 -translate-y-1/2 items-center justify-center rounded-md text-muted hover:bg-sunken hover:text-brand ${listening ? 'bg-marigold/20 text-marigold' : ''}`}>
          <Icon name="mic" className="h-[18px] w-[18px]" />
        </button>
      </div>
      <p id={`${id}-m`} className={`mt-1 min-h-[1.1rem] text-xs ${msg ? 'text-danger' : 'text-muted'}`}>{msg ?? hint ?? ''}</p>
    </div>
  )
}

interface SelFieldProps { label: string; value: string; options: string[]; onChange: (v: string) => void; error?: boolean; disabledOptions?: string[] }
export function SelField({ label, value, options, onChange, error, disabledOptions = [] }: SelFieldProps) {
  const { t } = useTranslation()
  const toast = useToast()
  const id = useId()
  const { supported, listening, listen } = useVoice()
  const speak = () => {
    if (!supported) return toast('info', t('form.mic_unsupported'))
    toast('info', t('form.mic_listening'))
    listen((txt) => {
      const s = txt.toLowerCase()
      const hit = options.find((o) => s.includes(o.toLowerCase()) || s.includes(t(`opt.${o}`).toLowerCase().split(' ')[0]))
      if (hit) onChange(hit)
      else toast('error', t('form.mic_noresult'))
    }, () => toast('error', t('form.mic_noresult')))
  }
  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-sm font-semibold">{label}</label>
      <div className="relative">
        <select id={id} className="field-input appearance-none pr-14" value={value} onChange={(e) => onChange(e.target.value)} aria-invalid={error}>
          {options.map((o) => <option key={o} value={o}>{t(`opt.${o}`, o)}{disabledOptions.includes(o) ? ' •' : ''}</option>)}
        </select>
        <button type="button" onClick={speak} aria-label={`${t('form.mic')}: ${label}`} aria-pressed={listening}
          className={`absolute right-7 top-1/2 flex h-9 w-9 -translate-y-1/2 items-center justify-center rounded-md text-muted hover:bg-sunken hover:text-brand ${listening ? 'bg-marigold/20 text-marigold' : ''}`}>
          <Icon name="mic" className="h-[18px] w-[18px]" />
        </button>
      </div>
      <p className="mt-1 min-h-[1.1rem] text-xs" />
    </div>
  )
}
