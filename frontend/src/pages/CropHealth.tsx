import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { api, ApiError } from '../api'
import FieldMap from '../components/FieldMap'
import NdviChart from '../components/NdviChart'
import { ErrorBox, Icon, PageTitle, Skeleton, Spinner } from '../components/ui'
import { useFetch } from '../hooks/useFetch'
import { useToast } from '../toast'
import type { CropHealthHistory, CropHealthLatest, FarmField, FieldCreateBody, NdviCategory } from '../types'
import { dateShort, fmt } from '../util'

const WINDOWS = [30, 60, 90, 180, 365]
type LatLon = [number, number]
type Method = 'point' | 'polygon'

export default function CropHealth() {
    const { t, i18n } = useTranslation()
    const toast = useToast()
    const status = useFetch(api.cropHealthStatus)
    const fieldsQ = useFetch(api.fields)
    const [selectedId, setSelectedId] = useState<number | null>(null)
    const [days, setDays] = useState(90)
    const [nonce, setNonce] = useState(0)
    const refreshRef = useRef(false)
    const [tileError, setTileError] = useState(false)

    const fields = fieldsQ.data ?? []
    const selected = fields.find((f) => f.id === selectedId) ?? null

    useEffect(() => { if (fieldsQ.data && fieldsQ.data.length && !fieldsQ.data.some((f) => f.id === selectedId)) setSelectedId(fieldsQ.data[0].id) }, [fieldsQ.data, selectedId])

    // latest and history are fetched one after the other so they share a single satellite request (the server caches the first)
    const health = useFetch<{ latest: CropHealthLatest; history: CropHealthHistory } | null>(async () => {
        if (selectedId == null) return null
        const refresh = refreshRef.current
        refreshRef.current = false
        const latest = await api.cropHealthLatest(selectedId, days, refresh)
        const history = await api.cropHealthHistory(selectedId, days)
        return { latest, history }
    }, [selectedId, days, nonce])
    useEffect(() => { setTileError(false) }, [health.data?.latest.layer?.tile_url])

    // ------------------------------------------------------------------ add-field form
    const [adding, setAdding] = useState(false)
    const [name, setName] = useState('')
    const [method, setMethod] = useState<Method>('point')
    const [center, setCenter] = useState<{ lat: number; lon: number } | null>(null)
    const [latText, setLatText] = useState('')
    const [lonText, setLonText] = useState('')
    const [radius, setRadius] = useState(100)
    const [vertices, setVertices] = useState<LatLon[]>([])
    const [formError, setFormError] = useState<string | null>(null)
    const [saving, setSaving] = useState(false)
    const [locating, setLocating] = useState(false)

    const resetForm = () => { setName(''); setMethod('point'); setCenter(null); setLatText(''); setLonText(''); setRadius(100); setVertices([]); setFormError(null) }
    const setPoint = useCallback((lat: number, lon: number) => { setCenter({ lat, lon }); setLatText(lat.toFixed(6)); setLonText(lon.toFixed(6)); setFormError(null) }, [])
    const addVertex = useCallback((lat: number, lon: number) => { setVertices((v) => [...v, [lat, lon]]); setFormError(null) }, [])
    const typed = (which: 'lat' | 'lon', v: string) => {
        const nl = which === 'lat' ? v : latText, no = which === 'lon' ? v : lonText
        if (which === 'lat') setLatText(v); else setLonText(v)
        const la = Number(nl), lo = Number(no)
        if (nl.trim() && no.trim() && Number.isFinite(la) && Number.isFinite(lo) && Math.abs(la) <= 90 && Math.abs(lo) <= 180) setCenter({ lat: la, lon: lo })
    }
    const useMyLocation = () => {
        if (!navigator.geolocation) return toast('error', t('form.loc_denied'))
        setLocating(true)
        navigator.geolocation.getCurrentPosition((p) => { setLocating(false); setPoint(p.coords.latitude, p.coords.longitude) },
            () => { setLocating(false); toast('error', t('form.loc_denied')) }, { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 })
    }

    const save = async () => {
        let body: FieldCreateBody
        if (!name.trim()) return setFormError(t('health.need_name'))
        if (method === 'point') {
            const la = Number(latText), lo = Number(lonText)
            if (!latText.trim() || !lonText.trim() || !Number.isFinite(la) || !Number.isFinite(lo) || Math.abs(la) > 90 || Math.abs(lo) > 180 || la < -56 || la > 84) return setFormError(center ? t('health.bad_coords') : t('health.need_point'))
            body = { name: name.trim(), latitude: la, longitude: lo, radius_m: radius }
        } else {
            if (vertices.length < 3) return setFormError(t('health.need_polygon'))
            const ring = vertices.map(([la, lo]) => [Number(lo.toFixed(7)), Number(la.toFixed(7))])
            body = { name: name.trim(), geometry: { type: 'Polygon', coordinates: [[...ring, ring[0]]] } }
        }
        setSaving(true); setFormError(null)
        try {
            const f = await api.createField(body)
            toast('success', t('health.saved'))
            await fieldsQ.reload()
            setSelectedId(f.id); setAdding(false); resetForm()
        } catch (e) {
            setFormError(e instanceof ApiError ? (e.errors[0]?.message ?? e.message) : t('common.error'))
        } finally { setSaving(false) }
    }

    const remove = async (f: FarmField) => {
        if (!window.confirm(t('health.delete_confirm'))) return
        try { await api.deleteField(f.id); toast('success', t('health.deleted')); if (selectedId === f.id) setSelectedId(null); fieldsQ.reload() } catch { toast('error', t('common.error')) }
    }
    const refresh = () => { refreshRef.current = true; setNonce((n) => n + 1) }

    const latest = health.data?.latest ?? null
    const history = health.data?.history ?? null
    const mapPolygon = useMemo(() => (adding ? null : selected?.geometry ?? null), [adding, selected])
    const catColor: Record<NdviCategory, string> = { low: '#dc2626', moderate: '#eab308', high: '#16a34a' }

    return (
        <>
            <PageTitle title={t('health.title')} sub={t('health.subtitle')}>
                {selected && !adding && (
                    <div className="flex flex-wrap items-end gap-2">
                        <div><label htmlFor="win" className="mb-1 block text-xs font-semibold text-muted">{t('health.window')}</label>
                            <select id="win" className="field-input !w-auto" value={days} onChange={(e) => setDays(Number(e.target.value))}>
                                {WINDOWS.map((w) => <option key={w} value={w}>{t('health.days_n', { n: w })}</option>)}</select></div>
                        <button className="btn btn-outline" onClick={refresh} disabled={health.loading}>{health.loading ? <Spinner /> : null} {t('health.refresh')}</button>
                    </div>
                )}
            </PageTitle>

            {status.data && !status.data.configured && (
                <div className="panel mb-5 border-warn/50 p-4" role="note"><p className="flex items-center gap-2 font-semibold text-warn"><Icon name="alert" className="h-4 w-4" />{t('health.unconfigured_title')}</p>
                    <p className="mt-1 text-sm">{t('health.unconfigured_text')}</p></div>
            )}
            {fieldsQ.error && <ErrorBox message={fieldsQ.error} onRetry={fieldsQ.reload} />}

            <div className="grid gap-6 lg:grid-cols-3">
                {/* ------------------------------------------------ left column: fields + add form */}
                <aside className="space-y-4 lg:col-span-1" aria-label={t('health.fields_title')}>
                    <section className="panel p-4">
                        <div className="mb-3 flex items-center justify-between"><h2 className="text-lg font-bold">{t('health.fields_title')}</h2>
                            {!adding && <button className="btn btn-primary !min-h-9 !px-3 text-sm" onClick={() => { resetForm(); setAdding(true) }}>+ {t('health.add_field')}</button>}</div>
                        {fieldsQ.loading && <div className="space-y-2"><Skeleton className="h-12" /><Skeleton className="h-12" /></div>}
                        {!fieldsQ.loading && fields.length === 0 && !adding && (
                            <div className="py-6 text-center"><Icon name="leaf" className="mx-auto h-8 w-8 text-brand" /><p className="mt-2 font-semibold">{t('health.no_fields')}</p><p className="text-sm text-muted">{t('health.no_fields_text')}</p></div>
                        )}
                        <ul className="space-y-1.5">
                            {fields.map((f) => (
                                <li key={f.id} className="flex items-center gap-1">
                                    <button onClick={() => { setAdding(false); setSelectedId(f.id) }} aria-pressed={f.id === selectedId && !adding}
                                        className={`flex-1 rounded-lg border px-3 py-2 text-left ${f.id === selectedId && !adding ? 'border-brand bg-brand-soft' : 'border-line hover:bg-sunken'}`}>
                                        <span className="block font-semibold">{f.name}</span><span className="text-xs text-muted">{t('health.area_ha', { ha: fmt(f.area_ha, f.area_ha < 10 ? 2 : 1) })}</span></button>
                                    <button className="btn btn-ghost !px-2.5 text-danger" onClick={() => remove(f)} aria-label={`${t('health.delete')}: ${f.name}`}><Icon name="trash" className="h-4 w-4" /></button>
                                </li>
                            ))}
                        </ul>
                    </section>

                    {adding && (
                        <section className="panel space-y-3 p-4" aria-label={t('health.add_field')}>
                            <h2 className="text-lg font-bold">{t('health.add_field')}</h2>
                            <div><label htmlFor="fname" className="mb-1 block text-sm font-semibold">{t('health.name')}</label>
                                <input id="fname" className="field-input" value={name} maxLength={120} placeholder={t('health.name_ph')} onChange={(e) => setName(e.target.value)} /></div>
                            <fieldset><legend className="mb-1 text-sm font-semibold">{t('health.method')}</legend>
                                <div className="grid grid-cols-2 gap-2">
                                    {(['point', 'polygon'] as Method[]).map((m) => (
                                        <button key={m} type="button" aria-pressed={method === m} onClick={() => { setMethod(m); setFormError(null) }}
                                            className={`btn !min-h-10 text-sm ${method === m ? 'btn-primary' : 'btn-outline'}`}>{t(m === 'point' ? 'health.m_point' : 'health.m_polygon')}</button>))}
                                </div></fieldset>
                            {method === 'point' ? (
                                <>
                                    <p className="text-sm text-muted">{t('health.pick_hint')}</p>
                                    <div className="grid grid-cols-2 gap-2">
                                        <div><label htmlFor="lat" className="mb-1 block text-xs font-semibold">{t('health.lat')}</label><input id="lat" inputMode="decimal" className="field-input" value={latText} onChange={(e) => typed('lat', e.target.value)} /></div>
                                        <div><label htmlFor="lon" className="mb-1 block text-xs font-semibold">{t('health.lon')}</label><input id="lon" inputMode="decimal" className="field-input" value={lonText} onChange={(e) => typed('lon', e.target.value)} /></div>
                                    </div>
                                    <p className="text-xs text-muted">{t('health.coords_hint')}</p>
                                    <button type="button" className="btn btn-outline w-full" onClick={useMyLocation} disabled={locating}>{locating ? <Spinner /> : <Icon name="pin" />} {t('form.use_location')}</button>
                                    <div><label htmlFor="rad" className="flex justify-between text-sm font-semibold"><span>{t('health.radius')}</span><span className="text-muted">{t('health.radius_m', { m: radius })}</span></label>
                                        <input id="rad" type="range" min={20} max={1000} step={10} value={radius} onChange={(e) => setRadius(Number(e.target.value))} className="w-full" /></div>
                                </>
                            ) : (
                                <>
                                    <p className="text-sm text-muted">{t('health.draw_hint')}</p>
                                    <p className="text-sm font-semibold" aria-live="polite">{t('health.vertices', { n: vertices.length })}</p>
                                    <div className="flex gap-2"><button type="button" className="btn btn-outline !min-h-9 text-sm" onClick={() => setVertices((v) => v.slice(0, -1))} disabled={!vertices.length}>{t('health.undo')}</button>
                                        <button type="button" className="btn btn-outline !min-h-9 text-sm" onClick={() => setVertices([])} disabled={!vertices.length}>{t('health.clear')}</button></div>
                                </>
                            )}
                            {formError && <p role="alert" className="text-sm text-danger">{formError}</p>}
                            <div className="flex gap-2"><button className="btn btn-primary flex-1" onClick={save} disabled={saving}>{saving && <Spinner />} {t('health.save_field')}</button>
                                <button className="btn btn-outline" onClick={() => { setAdding(false); resetForm() }}>{t('health.cancel')}</button></div>
                        </section>
                    )}
                </aside>

                {/* ------------------------------------------------ right column: map + results */}
                <div className="space-y-5 lg:col-span-2">
                    <FieldMap label={t('health.map_label')} height={380} polygon={mapPolygon} tileUrl={adding ? null : latest?.layer?.tile_url ?? null} onTileError={() => setTileError(true)}
                        editMode={adding ? method : null} center={center} radius={radius} vertices={vertices} onPoint={setPoint} onVertex={addVertex} />
                    {tileError && <p role="status" className="rounded-lg bg-warn/10 px-3 py-2 text-sm text-warn">{t('health.tiles_error')}</p>}

                    {!adding && !selected && !fieldsQ.loading && <p className="panel p-6 text-center text-muted">{fields.length ? t('health.select_field') : t('health.no_fields_text')}</p>}

                    {!adding && selected && health.loading && (
                        <div className="space-y-3" aria-busy="true"><p role="status" className="flex items-center gap-2 text-sm text-muted"><Spinner /> {t('health.refreshing')}</p><Skeleton className="h-40" /><Skeleton className="h-56" /></div>
                    )}
                    {!adding && selected && health.error && !health.loading && <ErrorBox message={`${t('health.load_error')}: ${health.error}`} onRetry={health.reload} />}

                    {!adding && selected && latest && !health.loading && (
                        <>
                            {latest.stale && latest.warning && <p role="status" className="panel border-warn/50 p-3 text-sm"><b className="text-warn">{t('health.stale_title')}.</b> {latest.warning}</p>}
                            {latest.status === 'no_data' || !latest.observation ? (
                                <section className="panel p-6" aria-live="polite"><h2 className="flex items-center gap-2 text-xl font-bold"><Icon name="info" className="text-info" />{t('health.no_data_title')}</h2>
                                    <p className="mt-2 text-muted">{t('health.no_data_text', { n: latest.window.days })}</p></section>
                            ) : (
                                <section className="panel p-5 sm:p-6" aria-labelledby="latest-h" aria-live="polite">
                                    <div className="flex flex-wrap items-start justify-between gap-4">
                                        <div><h2 id="latest-h" className="text-sm font-semibold text-muted">{t('health.latest')}</h2>
                                            <p className="font-display text-6xl font-bold" style={{ color: catColor[latest.observation.category] }}>{fmt(latest.observation.ndvi_mean)}<span className="ml-2 text-lg text-muted">{t('health.ndvi')}</span></p></div>
                                        <div className="text-right text-sm"><span className="inline-flex items-center gap-2 rounded-full px-3 py-1 font-semibold" style={{ background: `${catColor[latest.observation.category]}22`, color: 'inherit' }}>
                                            <span aria-hidden className="h-3 w-3 rounded-full" style={{ background: catColor[latest.observation.category] }} />{t(`health.cat_${latest.observation.category}`)}</span>
                                            <p className="mt-2 text-muted">{t('health.observed_on', { date: dateShort(latest.observation.date, i18n.language) })}</p>
                                            <p className="text-muted">{t('health.clear_sky', { p: Math.round(latest.observation.valid_fraction * 100) })}</p></div>
                                    </div>
                                    <p className="mt-4">{t(`health.cat_${latest.observation.category}_text`)}</p>
                                    {latest.previous && latest.interpretation?.change != null && (() => {
                                        const d = latest.interpretation.change, n = Math.round((new Date(latest.observation.date).getTime() - new Date(latest.previous.date).getTime()) / 86400000)
                                        return <p className="mt-2 text-sm text-muted">{t(d <= -0.1 ? 'health.change_down' : d >= 0.1 ? 'health.change_up' : 'health.change_flat', { d: Math.abs(d) >= 0.1 ? fmt(Math.abs(d)) : `${d > 0 ? '+' : ''}${fmt(d)}`, n })} <span className="opacity-80">{t('health.previous', { v: fmt(latest.previous.ndvi_mean), date: dateShort(latest.previous.date, i18n.language) })}</span></p>
                                    })()}
                                    {latest.layer_error && <p className="mt-3 rounded-lg bg-warn/10 px-3 py-2 text-sm text-warn">{t('health.layer_error')}</p>}
                                    <p className="mt-4 text-xs font-semibold text-muted">{t('health.disclaimer')}</p>
                                </section>
                            )}

                            <section className="panel p-5" aria-labelledby="legend-h"><h2 id="legend-h" className="mb-2 text-lg font-bold">{t('health.legend')}</h2>
                                <ul className="flex flex-wrap gap-x-6 gap-y-2">{latest.legend.map((l) => (
                                    <li key={l.category} className="flex items-center gap-2 text-sm"><span aria-hidden className="h-4 w-4 rounded" style={{ background: l.color }} /><b>{t(`health.cat_${l.category}`)}</b><span className="text-muted">{l.range}</span></li>))}</ul>
                                <p className="mt-2 text-xs text-muted">{t('health.legend_note', { low: latest.thresholds.low, high: latest.thresholds.high })}</p></section>

                            {history && (
                                <section className="panel p-5" aria-labelledby="hist-h"><h2 id="hist-h" className="text-lg font-bold">{t('health.history_title')}</h2>
                                    {history.count === 0 ? <p className="py-6 text-center text-muted">{t('health.no_data_title')}</p> : (
                                        <>{history.count === 1 && <p className="text-sm text-muted">{t('health.history_one')}</p>}
                                            <NdviChart observations={history.observations} low={history.thresholds.low} high={history.thresholds.high} />
                                            <p className="text-xs text-muted">{t('health.history_note')}</p></>)}
                                </section>
                            )}

                            <details className="panel p-5"><summary className="cursor-pointer text-lg font-bold">{t('health.caveats_title')}</summary>
                                <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-muted">{[1, 2, 3, 4, 5].map((n) => <li key={n}>{t(`health.caveat_${n}`)}</li>)}</ul></details>
                            <p className="text-xs text-muted">{t('health.source')}{latest.last_synced_at && ` ${t('health.last_sync', { date: dateShort(latest.last_synced_at, i18n.language) })}`}</p>
                        </>
                    )}
                </div>
            </div>
        </>
    )
}