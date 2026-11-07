import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import type { GeoPolygon } from '../types'

type LatLon = [number, number]
interface Props {
    height?: number
    /** Saved field boundary to show (view mode). */
    polygon?: GeoPolygon | null
    /** Satellite NDVI tile layer (temporary URL issued by the server). */
    tileUrl?: string | null
    onTileError?: () => void
    /** Editing: 'point' = click sets the centre (circle of `radius` metres), 'polygon' = clicks add corners. */
    editMode?: 'point' | 'polygon' | null
    center?: { lat: number; lon: number } | null
    radius?: number
    vertices?: LatLon[]
    onPoint?: (lat: number, lon: number) => void
    onVertex?: (lat: number, lon: number) => void
    label: string
}

const INDIA: LatLon = [20.59, 78.96]

export default function FieldMap({ height = 380, polygon, tileUrl, onTileError, editMode, center, radius = 100, vertices = [], onPoint, onVertex, label }: Props) {
    const { t } = useTranslation()
    const el = useRef<HTMLDivElement>(null)
    const map = useRef<L.Map | null>(null)
    const polyLayer = useRef<L.LayerGroup | null>(null)
    const tileLayer = useRef<L.TileLayer | null>(null)
    const editLayer = useRef<L.LayerGroup | null>(null)
    // latest callbacks/mode without re-creating the map
    const live = useRef({ editMode, onPoint, onVertex, onTileError })
    live.current = { editMode, onPoint, onVertex, onTileError }

    // create the map once
    useEffect(() => {
        if (!el.current) return
        const m = L.map(el.current, { zoomControl: true }).setView(INDIA, 5)
        L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: t('health.map_attr') }).addTo(m)
        polyLayer.current = L.layerGroup().addTo(m)
        editLayer.current = L.layerGroup().addTo(m)
        m.on('click', (e: L.LeafletMouseEvent) => {
            const { editMode: mode, onPoint: p, onVertex: v } = live.current
            if (mode === 'point') p?.(e.latlng.lat, e.latlng.lng)
            else if (mode === 'polygon') v?.(e.latlng.lat, e.latlng.lng)
        })
        map.current = m
        setTimeout(() => m.invalidateSize(), 50)
        return () => { m.remove(); map.current = null }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    // saved boundary: white halo + dark line so it stays visible on top of NDVI colours
    useEffect(() => {
        const g = polyLayer.current, m = map.current
        if (!g || !m) return
        g.clearLayers()
        if (!polygon) return
        const halo = L.geoJSON(polygon as never, { style: { color: '#ffffff', weight: 6, fill: false, opacity: 0.9 } })
        const line = L.geoJSON(polygon as never, { style: { color: '#0f172a', weight: 2.5, fill: false } })
        g.addLayer(halo); g.addLayer(line)
        m.fitBounds(line.getBounds(), { padding: [28, 28], maxZoom: 18 })
    }, [polygon])

    // NDVI tiles
    useEffect(() => {
        const m = map.current
        if (!m) return
        tileLayer.current?.remove()
        tileLayer.current = null
        if (!tileUrl) return
        const tl = L.tileLayer(tileUrl, { opacity: 0.85, maxZoom: 19, attribution: 'Contains modified Copernicus Sentinel data' })
        tl.on('tileerror', () => live.current.onTileError?.())
        tl.addTo(m)
        tileLayer.current = tl
        polyLayer.current?.eachLayer((l) => (l as L.GeoJSON).bringToFront())
    }, [tileUrl])

    // editing overlays
    useEffect(() => {
        const g = editLayer.current, m = map.current
        if (!g || !m) return
        g.clearLayers()
        m.getContainer().style.cursor = editMode ? 'crosshair' : ''
        const style = { color: '#d68f0c', weight: 2, fillOpacity: 0.18 }
        if (editMode === 'point' && center) {
            g.addLayer(L.circle([center.lat, center.lon], { radius, ...style }))
            g.addLayer(L.circleMarker([center.lat, center.lon], { radius: 5, color: '#fff', weight: 2, fillColor: '#d68f0c', fillOpacity: 1 }))
        }
        if (editMode === 'polygon' && vertices.length) {
            g.addLayer(vertices.length >= 3 ? L.polygon(vertices, style) : L.polyline(vertices, style))
            vertices.forEach((v) => g.addLayer(L.circleMarker(v, { radius: 5, color: '#fff', weight: 2, fillColor: '#d68f0c', fillOpacity: 1 })))
        }
    }, [editMode, center, radius, vertices])

    // move the view to a newly chosen centre (without zooming out if already close)
    useEffect(() => {
        const m = map.current
        if (m && editMode === 'point' && center) m.setView([center.lat, center.lon], Math.max(m.getZoom(), 16))
    }, [center, editMode])

    return <div ref={el} role="application" aria-label={label} style={{ height }} className="w-full overflow-hidden rounded-xl border border-line" />
}