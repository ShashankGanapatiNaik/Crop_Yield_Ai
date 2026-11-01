import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useAuth } from '../auth'
import { Icon } from '../components/ui'

export default function Home() {
  const { t } = useTranslation()
  const { user } = useAuth()
  const feats = [['f1', 'leaf'], ['f2', 'info'], ['f3', 'drop'], ['f4', 'flame']] as const
  return (
    <div className="space-y-16">
      <section className="relative overflow-hidden rounded-3xl bg-brand-strong px-6 py-14 text-on-brand sm:px-12 sm:py-20">
        <svg className="absolute inset-0 h-full w-full opacity-40" preserveAspectRatio="none" viewBox="0 0 800 400" aria-hidden="true">
          {Array.from({ length: 9 }).map((_, i) => (
            <path key={i} className="furrow" d={`M-20 ${150 + i * 28} C 200 ${110 + i * 30}, 420 ${200 + i * 26}, 820 ${130 + i * 30}`} fill="none" stroke={i % 3 === 0 ? '#7bc47f' : '#2e7d4f'} strokeWidth="3" strokeLinecap="round" style={{ animationDuration: `${20 + i * 3}s` }} />
          ))}
        </svg>
        <div className="relative max-w-2xl">
          <h1 className="text-4xl font-bold leading-[1.1] sm:text-6xl">{t('home.title')}</h1>
          <p className="mt-5 max-w-xl text-lg opacity-90">{t('home.subtitle')}</p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link to={user ? '/predict' : '/register'} className="btn bg-marigold text-[#1b1b10] hover:brightness-110">{t('home.cta_start')} <Icon name="arrow" /></Link>
            {!user && <Link to="/login" className="btn border border-white/40 text-on-brand hover:bg-white/10">{t('home.cta_login')}</Link>}
          </div>
        </div>
      </section>

      <section className="grid gap-x-12 gap-y-8 md:grid-cols-2" aria-label="Features">
        {feats.map(([k, icon]) => (
          <div key={k} className="flex gap-4 border-t border-line pt-5">
            <span className="mt-1 text-brand"><Icon name={icon} className="h-6 w-6" /></span>
            <div><h2 className="text-xl font-bold">{t(`home.${k}_t`)}</h2><p className="mt-1 text-muted">{t(`home.${k}_d`)}</p></div>
          </div>
        ))}
      </section>

      <section>
        <h2 className="mb-5 text-2xl font-bold text-brand-strong">{t('home.how_t')}</h2>
        <ol className="grid gap-4 md:grid-cols-3">
          {['s1', 's2', 's3'].map((s, i) => (
            <li key={s} className="panel flex gap-3 p-5"><span className="font-display text-3xl font-bold text-marigold">{i + 1}</span><p>{t(`home.${s}`)}</p></li>
          ))}
        </ol>
        <p className="mt-6 text-muted">{t('home.langs')}</p>
      </section>
    </div>
  )
}
