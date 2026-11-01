import type { AdminStats, AdminUser, CompareRow, CropsMeta, HistoryItem, Inputs, ModelInfo, Place, Prediction, TokenOut, User, Weather, WhatIf } from './types'

export const API_BASE: string = import.meta.env.VITE_API_URL ?? '/api'
const KEY = 'cropai_token'
export const tokenStore = {
  get: () => localStorage.getItem(KEY),
  set: (t: string) => localStorage.setItem(KEY, t),
  clear: () => localStorage.removeItem(KEY),
}

export class ApiError extends Error {
  status: number
  errors: { field: string; message: string }[]
  constructor(status: number, message: string, errors: { field: string; message: string }[] = []) {
    super(message)
    this.status = status
    this.errors = errors
  }
}

async function raw(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers)
  const token = tokenStore.get()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, headers })
  } catch {
    throw new ApiError(0, 'Network error. Check your connection.')
  }
  if (res.status === 401 && token) {
    tokenStore.clear()
    window.dispatchEvent(new Event('auth:expired'))
  }
  if (!res.ok) {
    let detail = res.statusText
    let errors: { field: string; message: string }[] = []
    try {
      const j = await res.json()
      detail = typeof j.detail === 'string' ? j.detail : detail
      errors = j.errors ?? []
    } catch { /* non-JSON error body */ }
    throw new ApiError(res.status, detail, errors)
  }
  return res
}

async function req<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await raw(path, init)
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T)
}
const post = <T>(path: string, body: unknown) => req<T>(path, { method: 'POST', body: JSON.stringify(body) })

export const api = {
  login: (email: string, password: string) => post<TokenOut>('/auth/login', { email, password }),
  register: (name: string, email: string, password: string) => post<TokenOut>('/auth/register', { name, email, password }),
  me: () => req<User>('/auth/me'),
  predict: (inputs: Inputs) => post<Prediction>('/predict', inputs),
  compare: (inputs: Omit<Inputs, 'crop'>, crops?: string[]) => post<{ results: CompareRow[]; best_crop: string }>('/compare-crops', { ...inputs, crops }),
  whatIf: (base: Inputs, changes: Record<string, number | string>, sweep?: { param: string; min: number; max: number; steps: number }) =>
    post<WhatIf>('/what-if', { base, changes, sweep }),
  weather: (lat: number, lon: number) => req<Weather>(`/weather?lat=${lat}&lon=${lon}`),
  geoReverse: (lat: number, lon: number) => req<Place>(`/geocode/reverse?lat=${lat}&lon=${lon}`),
  geoSearch: (q: string) => req<{ results: Place[] }>(`/geocode/search?q=${encodeURIComponent(q)}`),
  history: () => req<{ total: number; items: HistoryItem[] }>('/history?limit=100'),
  historyItem: (id: number) => req<Prediction>(`/history/${id}`),
  deleteHistory: (id: number) => req<void>(`/history/${id}`, { method: 'DELETE' }),
  crops: () => req<CropsMeta>('/crops'),
  modelInfo: () => req<ModelInfo>('/model-info'),
  adminStats: () => req<AdminStats>('/admin/stats'),
  adminUsers: () => req<AdminUser[]>('/admin/users'),
  async downloadReport(id: number) {
    const blob = await (await raw(`/report/${id}`)).blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `yield-report-${id}.pdf`
    a.click()
    setTimeout(() => URL.revokeObjectURL(url), 2000)
  },
}