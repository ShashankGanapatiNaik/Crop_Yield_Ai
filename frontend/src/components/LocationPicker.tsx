import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api } from '../api'
import type { Place } from '../types'
import { Icon, Spinner } from './ui'

interface Props {
    initial?: { lat: number; lon: number } | null
    autoLocate?: boolean
    onClose: () => void
    onConfirm: (lat: number, lon: number, place: Place | null) => void
}
const INDIA: [number, number] = [20.59, 78.96]
const APPROX_METRES = 1500

const pin = L.divIcon({
    className: '',
    html: '<div style="width:28px;height:28px;border-radius:50% 50% 50% 0;background:#d68f0c;border:3px solid #fff;transform:rotate(-45deg);box-shadow:0 2px 6px rgba(0,0,0,.4)"></div>',
    iconSize: [28, 28],
    iconAnchor: [14, 28],
})

export default function LocationPicker({ initial, autoLocate, onClose, onConfirm }: Props) {
    const { t } = useTranslation()
    const el = useRef<HTMLDivElement>(null)
    const map = useRef<L.Map | null>(null)
    const marker = useRef<L.Marker | null>(null)
    const circle = useRef<L.Circle | null>(null)
    const seq = useRef(0)
    const [pos, setPos] = useState<{ lat: number; lon: number } | null>(initial ?? null)
    const [place, setPlace] = useState<Place | null>(null)
    const [looking, setLooking] = useState(false)
    const [geoFail, setGeoFail] = useState(false)
    const [accuracy, setAccuracy] = useState<number | null>(null)
    const [locating, setLocating] = useState(false)
    const [locateErr, setLocateErr] = useState(false)
    const [query, setQuery] = useState('')
    const [results, setResults] = useState<Place[] | null>(null)
    const [searching, setSearching] = useState(false)

    const resolveName = useCallback(async (lat: number, lon: number) => {
        const id = ++seq.current
        setLooking(true); setGeoFail(false); setPlace(null)
        try {
            const p = await api.geoReverse(lat, lon)
            if (id === seq.current) setPlace(p)
        } catch {
            if (id === seq.current) setGeoFail(true)
        } finally {
            if (id === seq.current) setLooking(false)
        }
    }, [])

    const moveTo = useCallback((lat: number, lon: number, opts: { zoom?: number; acc?: number | null } = {}) => {
        setPos({ lat, lon })
        setAccuracy(opts.acc ?? null)
        marker.current?.setLatLng([lat, lon])
        circle.current?.remove(); circle.current = null
        if (opts.acc && map.current) circle.current = L.circle([lat, lon], { radius: opts.acc, color: '#1f6b45', weight: 1, fillOpacity: 0.12 }).addTo(map.current)
        if (opts.zoom && map.current) map.current.setView([lat, lon], opts.zoom)
        resolveName(lat, lon)
    }, [resolveName])

    // Create the map once.
    useEffect(() => {
        if (!el.current) return
        const m = L.map(el.current, { zoomControl: true }).setView(initial ? [initial.lat, initial.lon] : INDIA, initial ? 14 : 5)
        L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: t('form.map_attr') }).addTo(m)
        const mk = L.marker(initial ? [initial.lat, initial.lon] : INDIA, { icon: pin, draggable: true, keyboard: true, opacity: initial ? 1 : 0 }).addTo(m)
        mk.on('dragend', () => { const ll = mk.getLatLng(); moveTo(ll.lat, ll.lng) })
        m.on('click', (e: L.LeafletMouseEvent) => { mk.setOpacity(1); moveTo(e.latlng.lat, e.latlng.lng) })
        map.current = m; marker.current = mk
        setTimeout(() => m.invalidateSize(), 50)
        if (initial) resolveName(initial.lat, initial.lon)
        return () => { m.remove(); map.current = null; marker.current = null; circle.current = null }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    const locate = useCallback(() => {
        if (!navigator.geolocation) { setLocateErr(true); return }
        setLocating(true); setLocateErr(false)
        navigator.geolocation.getCurrentPosition(
            (g) => { setLocating(false); marker.current?.setOpacity(1); moveTo(g.coords.latitude, g.coords.longitude, { zoom: g.coords.accuracy > APPROX_METRES ? 11 : 16, acc: g.coords.accuracy }) },
            () => { setLocating(false); setLocateErr(true) },
            { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 },
        )
    }, [moveTo])

    useEffect(() => { if (autoLocate && !initial) locate() }, [autoLocate, initial, locate])
    useEffect(() => { const k = (e: KeyboardEvent) => e.key === 'Escape' && onClose(); window.addEventListener('keydown', k); return () => window.removeEventListener('keydown', k) }, [onClose])

    // Debounced place search (Nominatim asks for <= 1 request/second).
    useEffect(() => {
        if (query.trim().length < 3) { setResults(null); return }
        setSearching(true)
        const id = setTimeout(async () => {
            try { setResults((await api.geoSearch(query.trim())).results) } catch { setResults([]) } finally { setSearching(false) }
        }, 650)
        return () => clearTimeout(id)
    }, [query])

    const pick = (p: Place) => { setResults(null); setQuery(''); marker.current?.setOpacity(1); moveTo(p.lat, p.lon, { zoom: 14 }) }

    return (
        <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/50 p-0 sm:items-center sm:p-4" role="dialog" aria-modal="true" aria-labelledby="lp-title">
            <div className="flex max-h-[95vh] w-full max-w-3xl flex-col overflow-hidden rounded-t-2xl bg-surface sm:rounded-2xl">
                <header className="flex items-center justify-between gap-3 border-b border-line px-5 py-3">
                    <h2 id="lp-title" className="text-xl font-bold text-brand-strong">{t('form.picker_title')}</h2>
                    <button className="btn btn-ghost !px-2.5" onClick={onClose} aria-label={t('form.close')}><Icon name="close" /></button>
                </header>

                <div className="space-y-3 overflow-y-auto p-5">
                    <div className="flex flex-wrap items-end gap-2">
                        <div className="relative min-w-52 flex-1">
                            <label htmlFor="lp-q" className="mb-1 block text-sm font-semibold">{t('form.search_label')}</label>
                            <input id="lp-q" className="field-input" value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t('form.search_ph')} autoComplete="off" />
                            {(results || searching) && (
                                <ul className="absolute z-[1000] mt-1 max-h-56 w-full overflow-auto rounded-lg border border-line bg-surface shadow-lg" role="listbox">
                                    {searching && <li className="flex items-center gap-2 px-3 py-2 text-sm text-muted"><Spinner /> {t('common.loading')}</li>}
                                    {!searching && results?.length === 0 && <li className="px-3 py-2 text-sm text-muted">{t('form.no_results')}</li>}
                                    {results?.map((p, i) => (
                                        <li key={i} role="option" aria-selected="false"><button type="button" className="w-full px-3 py-2 text-left text-sm hover:bg-sunken" onClick={() => pick(p)}>
                                            <span className="font-semibold">{p.label}</span><span className="block truncate text-xs text-muted">{p.display_name}</span></button></li>
                                    ))}
                                </ul>
                            )}
                        </div>
                        <button type="button" className="btn btn-outline" onClick={locate} disabled={locating}>{locating ? <Spinner /> : <Icon name="pin" />} {locating ? t('form.locating') : t('form.locate_me')}</button>
                    </div>

                    {locateErr && <p role="alert" className="text-sm text-danger">{t('form.locate_fail')}</p>}
                    <div ref={el} className="h-[320px] w-full overflow-hidden rounded-xl border border-line" aria-label={t('form.drag_hint')} />
                    <p className="text-xs text-muted">{t('form.drag_hint')}</p>

                    <div className="rounded-xl bg-sunken p-4" aria-live="polite">
                        <p className="text-xs font-semibold text-muted">{t('form.selected_place')}</p>
                        {!pos && <p className="text-muted">{t('form.drag_hint')}</p>}
                        {pos && looking && <p className="flex items-center gap-2"><Spinner /> {t('form.place_loading')}</p>}
                        {pos && place && <><p className="text-lg font-bold text-brand-strong">{place.label}</p><p className="text-xs text-muted">{place.display_name}</p></>}
                        {pos && geoFail && <p className="text-sm text-warn">{t('form.geocode_fail')}</p>}
                        {pos && <p className="mt-1 text-xs text-muted">{t('form.coords', { lat: pos.lat.toFixed(5), lon: pos.lon.toFixed(5) })}{accuracy ? ` · ${t('form.accuracy', { m: Math.round(accuracy) })}` : ''}</p>}
                        {accuracy !== null && accuracy > APPROX_METRES && <p className="mt-2 text-sm text-warn">{t('form.approx_warn')}</p>}
                    </div>
                </div>

                <footer className="flex justify-end gap-2 border-t border-line px-5 py-3">
                    <button className="btn btn-outline" onClick={onClose}>{t('common.cancel')}</button>
                    <button className="btn btn-primary" disabled={!pos || looking} onClick={() => pos && onConfirm(pos.lat, pos.lon, place)}>{t('form.use_here')}</button>
                </footer>
            </div>
        </div>
    )
}