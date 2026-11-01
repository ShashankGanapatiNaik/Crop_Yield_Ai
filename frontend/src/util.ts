export const fmt = (n: number, d = 2) => n.toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d })
export const inr = (n: number) => `₹${Math.round(n).toLocaleString('en-IN')}`
export const dateShort = (iso: string, lang = 'en') => new Date(iso).toLocaleDateString(lang === 'en' ? 'en-IN' : `${lang}-IN`, { day: 'numeric', month: 'short', year: 'numeric' })
export const cssColor = (name: string) => `rgb(${getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim()})`
