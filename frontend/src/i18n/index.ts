import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import en from './en.json'
import hi from './hi.json'
import kn from './kn.json'

export const LANGS = [
  { code: 'en', label: 'English', speech: 'en-IN' },
  { code: 'hi', label: 'हिन्दी', speech: 'hi-IN' },
  { code: 'kn', label: 'ಕನ್ನಡ', speech: 'kn-IN' },
] as const

const saved = typeof localStorage !== 'undefined' ? localStorage.getItem('cropai_lang') : null

i18n.use(initReactI18next).init({
  resources: { en: { translation: en }, hi: { translation: hi }, kn: { translation: kn } },
  lng: saved ?? 'en',
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
})

i18n.on('languageChanged', (l) => {
  document.documentElement.lang = l
  localStorage.setItem('cropai_lang', l)
})
document.documentElement.lang = i18n.language

export default i18n
