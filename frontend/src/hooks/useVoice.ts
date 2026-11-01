import { useCallback, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { LANGS } from '../i18n'

/* Minimal typing for the (still vendor-prefixed) Web Speech API. */
interface SpeechRec {
  lang: string; interimResults: boolean; maxAlternatives: number
  onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null
  onerror: (() => void) | null; onend: (() => void) | null
  start: () => void; stop: () => void
}
type Ctor = new () => SpeechRec
const getCtor = (): Ctor | undefined => {
  const w = window as unknown as { SpeechRecognition?: Ctor; webkitSpeechRecognition?: Ctor }
  return w.SpeechRecognition ?? w.webkitSpeechRecognition
}

const WORD_NUMS: Record<string, string> = { zero: '0', one: '1', two: '2', three: '3', four: '4', five: '5', six: '6', seven: '7', eight: '8', nine: '9', ten: '10' }
const DEVANAGARI = '०१२३४५६७८९'
const KANNADA = '೦೧೨೩೪೫೬೭೮೯'

/** Turn a spoken phrase ("six point five", "६.५", "650") into a numeric string, or null. */
export function parseSpokenNumber(text: string): string | null {
  let s = text.toLowerCase().trim()
  s = s.replace(/[०-९]/g, (d) => String(DEVANAGARI.indexOf(d))).replace(/[೦-೯]/g, (d) => String(KANNADA.indexOf(d)))
  s = s.replace(/\b(point|dot|दशमलव|ದಶಮಾಂಶ)\b/g, '.').replace(/\b(zero|one|two|three|four|five|six|seven|eight|nine|ten)\b/g, (w) => WORD_NUMS[w])
  s = s.replace(/(\d)\s*\.\s*(\d)/g, '$1.$2').replace(/(\d)\s+(?=\d\b)/g, '$1')
  const m = s.match(/-?\d+(\.\d+)?/)
  return m ? m[0] : null
}

export function useVoice() {
  const { i18n } = useTranslation()
  const [listening, setListening] = useState(false)
  const rec = useRef<SpeechRec | null>(null)
  const supported = typeof window !== 'undefined' && !!getCtor()

  const listen = useCallback((onText: (t: string) => void, onFail: () => void) => {
    const C = getCtor()
    if (!C) return onFail()
    rec.current?.stop()
    const r = new C()
    r.lang = LANGS.find((l) => l.code === i18n.language)?.speech ?? 'en-IN'
    r.interimResults = false
    r.maxAlternatives = 1
    let got = false
    r.onresult = (e) => { got = true; onText(e.results[0][0].transcript) }
    r.onerror = () => { setListening(false); onFail() }
    r.onend = () => { setListening(false); if (!got) onFail() }
    rec.current = r
    setListening(true)
    r.start()
  }, [i18n.language])

  return { supported, listening, listen }
}
