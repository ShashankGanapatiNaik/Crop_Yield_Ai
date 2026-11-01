import { useEffect, useState } from 'react'
import { Link, NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useAuth } from '../auth'
import { useTheme } from '../theme'
import { useToast } from '../toast'
import { LANGS } from '../i18n'
import { Icon } from './ui'

const Logo = () => (
  <span className="flex items-center gap-2 font-display text-lg font-bold text-brand-strong">
    <svg viewBox="0 0 32 32" className="h-8 w-8" aria-hidden="true"><rect width="32" height="32" rx="8" fill="rgb(var(--brand-strong))" />
      <path d="M16 22V12" stroke="#e9a21b" strokeWidth="2.2" strokeLinecap="round" /><path d="M16 14c-4 0-5.5-2-5.5-5 3.4 0 5.5 1.8 5.5 5zM16 16c3.5 0 5-2 5-4.6-3.2 0-5 1.7-5 4.6z" fill="#e9a21b" />
      <path d="M6 25c3-1.3 6-1.3 10 0s7 1.3 10 0" stroke="#7bc47f" strokeWidth="2" strokeLinecap="round" fill="none" /></svg>
    Crop Yield AI
  </span>
)

export default function Layout() {
  const { t, i18n } = useTranslation()
  const { user, logout } = useAuth()
  const { theme, toggle } = useTheme()
  const toast = useToast()
  const nav = useNavigate()
  const [open, setOpen] = useState(false)
  const [online, setOnline] = useState(navigator.onLine)
  useEffect(() => {
    const on = () => setOnline(true), off = () => setOnline(false)
    window.addEventListener('online', on); window.addEventListener('offline', off)
    return () => { window.removeEventListener('online', on); window.removeEventListener('offline', off) }
  }, [])
  useEffect(() => {
    const h = () => { toast('error', t('auth.expired')); nav('/login') }
    window.addEventListener('auth:expired', h)
    return () => window.removeEventListener('auth:expired', h)
  }, [nav, t, toast])

  const links = user
    ? [['/dashboard', 'dashboard'], ['/predict', 'predict'], ['/whatif', 'whatif'], ['/compare', 'compare'], ['/history', 'history'], ['/insights', 'insights'], ...(user.role === 'admin' ? [['/admin', 'admin']] : [])]
    : [['/', 'home'], ['/insights', 'insights']]
  const cls = ({ isActive }: { isActive: boolean }) => `rounded-lg px-3 py-2 text-sm font-semibold ${isActive ? 'bg-brand-soft text-brand-strong' : 'text-muted hover:bg-sunken hover:text-ink'}`

  return (
    <div className="flex min-h-screen flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-lg focus:bg-surface focus:px-4 focus:py-2">{t('app.skip')}</a>
      <header className="sticky top-0 z-40 border-b border-line bg-bg/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-2.5">
          <Link to={user ? '/dashboard' : '/'} aria-label={t('app.name')}><Logo /></Link>
          <nav className="ml-4 hidden flex-1 gap-1 lg:flex" aria-label="Main">{links.map(([to, k]) => <NavLink key={to} to={to} className={cls} end>{t(`nav.${k}`)}</NavLink>)}</nav>
          <div className="ml-auto flex items-center gap-1.5">
            <select aria-label={t('nav.language')} value={i18n.language} onChange={(e) => i18n.changeLanguage(e.target.value)} className="field-input !w-auto !py-1.5 text-sm">
              {LANGS.map((l) => <option key={l.code} value={l.code}>{l.label}</option>)}
            </select>
            <button className="btn btn-ghost !px-2.5" onClick={toggle} aria-label={t('nav.theme')}><Icon name={theme === 'dark' ? 'sun' : 'moon'} /></button>
            {user ? (
              <button className="btn btn-outline hidden sm:inline-flex" onClick={() => { logout(); toast('info', t('auth.bye')); nav('/') }}>{t('nav.logout')}</button>
            ) : (
              <Link to="/login" className="btn btn-primary hidden sm:inline-flex">{t('nav.login')}</Link>
            )}
            <button className="btn btn-ghost !px-2.5 lg:hidden" onClick={() => setOpen(!open)} aria-expanded={open} aria-label={t('nav.menu')}><Icon name={open ? 'close' : 'menu'} /></button>
          </div>
        </div>
        {open && (
          <nav className="border-t border-line bg-surface px-4 py-3 lg:hidden" aria-label="Mobile">
            <div className="flex flex-col gap-1">
              {links.map(([to, k]) => <NavLink key={to} to={to} className={cls} onClick={() => setOpen(false)} end>{t(`nav.${k}`)}</NavLink>)}
              {user ? <button className="btn btn-outline mt-2" onClick={() => { logout(); setOpen(false); nav('/') }}>{t('nav.logout')}</button>
                : <Link to="/login" className="btn btn-primary mt-2" onClick={() => setOpen(false)}>{t('nav.login')}</Link>}
            </div>
          </nav>
        )}
        {!online && <p role="status" className="bg-warn/15 px-4 py-1.5 text-center text-sm text-warn">{t('common.offline')}</p>}
      </header>
      <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 py-8" tabIndex={-1}><Outlet /></main>
      <footer className="border-t border-line px-4 py-6 text-center text-sm text-muted">{t('home.disclaimer')}</footer>
    </div>
  )
}
