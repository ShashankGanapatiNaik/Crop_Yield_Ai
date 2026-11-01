import { useCallback, useEffect, useState } from 'react'
import { ApiError } from '../api'

export function useFetch<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const run = useCallback(() => {
    setLoading(true); setError(null)
    fn().then(setData).catch((e) => setError(e instanceof ApiError ? e.message : String(e))).finally(() => setLoading(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  useEffect(run, [run])
  return { data, error, loading, reload: run, setData }
}
