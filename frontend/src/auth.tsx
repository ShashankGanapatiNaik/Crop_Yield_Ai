import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import { api, tokenStore } from './api'
import type { User } from './types'

interface AuthCtx {
  user: User | null; loading: boolean
  login: (email: string, password: string) => Promise<User>
  register: (name: string, email: string, password: string) => Promise<User>
  logout: () => void
}
const Ctx = createContext<AuthCtx>(null as unknown as AuthCtx)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(!!tokenStore.get())

  useEffect(() => {
    if (!tokenStore.get()) return
    api.me().then(setUser).catch(() => tokenStore.clear()).finally(() => setLoading(false))
  }, [])
  useEffect(() => {
    const h = () => setUser(null)
    window.addEventListener('auth:expired', h)
    return () => window.removeEventListener('auth:expired', h)
  }, [])

  const login = useCallback(async (e: string, p: string) => { const r = await api.login(e, p); tokenStore.set(r.access_token); setUser(r.user); return r.user }, [])
  const register = useCallback(async (n: string, e: string, p: string) => { const r = await api.register(n, e, p); tokenStore.set(r.access_token); setUser(r.user); return r.user }, [])
  const logout = useCallback(() => { tokenStore.clear(); setUser(null) }, [])
  return <Ctx.Provider value={{ user, loading, login, register, logout }}>{children}</Ctx.Provider>
}
export const useAuth = () => useContext(Ctx)
