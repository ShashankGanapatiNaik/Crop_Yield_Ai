import { createContext, useCallback, useContext, useState, type ReactNode } from 'react'

type Kind = 'success' | 'error' | 'info'
interface Toast { id: number; kind: Kind; msg: string }
const Ctx = createContext<(kind: Kind, msg: string) => void>(() => {})
let seq = 0

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([])
  const push = useCallback((kind: Kind, msg: string) => {
    const id = ++seq
    setItems((s) => [...s, { id, kind, msg }])
    setTimeout(() => setItems((s) => s.filter((t) => t.id !== id)), kind === 'error' ? 6500 : 4000)
  }, [])
  const color = { success: 'border-good text-good', error: 'border-danger text-danger', info: 'border-info text-info' }
  return (
    <Ctx.Provider value={push}>
      {children}
      <div className="pointer-events-none fixed inset-x-0 bottom-4 z-50 flex flex-col items-center gap-2 px-4" role="status" aria-live="polite">
        {items.map((t) => (
          <div key={t.id} className={`pointer-events-auto max-w-md rounded-xl border-l-4 bg-surface px-4 py-3 text-sm shadow-lg ring-1 ring-line ${color[t.kind]}`}>
            <span className="text-ink">{t.msg}</span>
          </div>
        ))}
      </div>
    </Ctx.Provider>
  )
}
export const useToast = () => useContext(Ctx)
