import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { ApiError } from '../api'
import { useAuth } from '../auth'
import { Spinner } from '../components/ui'
import { useToast } from '../toast'

export default function Login({ mode }: { mode: 'login' | 'register' }) {
  const { t } = useTranslation()
  const { login, register } = useAuth()
  const toast = useToast()
  const nav = useNavigate()
  const [f, setF] = useState({ name: '', email: '', password: '' })
  const [errs, setErrs] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const reg = mode === 'register'

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    const v: Record<string, string> = {}
    if (reg && f.name.trim().length < 2) v.name = t('auth.e_name')
    if (!/^\S+@\S+\.\S+$/.test(f.email)) v.email = t('auth.e_email')
    if (f.password.length < 8) v.password = t('auth.e_pw')
    setErrs(v)
    if (Object.keys(v).length) return
    setBusy(true)
    try {
      const u = reg ? await register(f.name.trim(), f.email, f.password) : await login(f.email, f.password)
      toast('success', t('auth.welcome', { name: u.name }))
      nav('/dashboard')
    } catch (err) {
      toast('error', err instanceof ApiError ? err.message : t('common.error'))
    } finally { setBusy(false) }
  }

  const field = (k: 'name' | 'email' | 'password', label: string, type: string, ac: string) => (
    <div>
      <label htmlFor={k} className="mb-1 block text-sm font-semibold">{label}</label>
      <input id={k} type={type} autoComplete={ac} className="field-input" value={f[k]} onChange={(e) => setF({ ...f, [k]: e.target.value })} aria-invalid={!!errs[k]} aria-describedby={`${k}-e`} />
      <p id={`${k}-e`} className="mt-1 min-h-[1.1rem] text-xs text-danger">{errs[k]}</p>
    </div>
  )
  return (
    <div className="mx-auto max-w-md">
      <form onSubmit={submit} noValidate className="panel space-y-2 p-6 sm:p-8">
        <h1 className="mb-3 text-3xl font-bold text-brand-strong">{reg ? t('auth.register_t') : t('auth.login_t')}</h1>
        {reg && field('name', t('auth.name'), 'text', 'name')}
        {field('email', t('auth.email'), 'email', 'email')}
        {field('password', t('auth.password'), 'password', reg ? 'new-password' : 'current-password')}
        <button className="btn btn-primary w-full" disabled={busy}>{busy && <Spinner />} {reg ? t('auth.register_btn') : t('auth.login_btn')}</button>
        <p className="pt-2 text-center text-sm text-muted">{reg ? t('auth.have') : t('auth.none')}{' '}
          <Link className="font-semibold text-brand underline" to={reg ? '/login' : '/register'}>{reg ? t('nav.login') : t('nav.register')}</Link></p>
      </form>
    </div>
  )
}
