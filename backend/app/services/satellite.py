"""Satellite crop-health logic that does NOT depend on any satellite provider (fully unit-tested).

* geometry validation (GeoJSON polygon / point+radius), area and centroid
* NDVI = (NIR - Red) / (NIR + Red)   (Sentinel-2: B8 = NIR, B4 = Red)
* configurable, APPROXIMATE thresholds for visualisation
* cautious interpretation text: NDVI measures vegetation greenness; it never diagnoses a disease
* aggregation of per-scene statistics into one observation per date
* the provider interface the Earth Engine module implements (and tests replace with a fake)

Nothing here fabricates satellite data: when no valid observation exists the functions say so.
"""
import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Protocol

import numpy as np


# ------------------------------------------------------------------------------------------------ errors
class SatelliteError(Exception):
    """The satellite service could not answer (mapped to HTTP 502)."""
    status_code = 502


class SatelliteNotConfigured(SatelliteError):
    """Earth Engine credentials / project are missing (HTTP 503)."""
    status_code = 503


class SatelliteTimeout(SatelliteError):
    """The computation took too long (HTTP 504)."""
    status_code = 504


class GeometryError(ValueError):
    """Invalid field geometry. `field` names the offending request field."""

    def __init__(self, message: str, field: str = "geometry"):
        super().__init__(message)
        self.field = field


# ------------------------------------------------------------------------------------------------ colours / thresholds
COLORS = {"low": "#dc2626", "moderate": "#eab308", "high": "#16a34a"}
LAYER_STYLES = ("classes", "gradient")
GRADIENT_PALETTE = ["#b91c1c", "#ef4444", "#f59e0b", "#fde047", "#a3e635", "#22c55e", "#166534"]
SENTINEL2_LAT_RANGE = (-56.0, 84.0)  # Sentinel-2 acquisition coverage


@dataclass(frozen=True)
class Thresholds:
    """Approximate NDVI class boundaries used for colouring only. Crop- and stage-specific values differ."""
    low: float = 0.2
    high: float = 0.5

    def __post_init__(self):
        if not (-1 < self.low < self.high < 1):
            raise ValueError(f"NDVI thresholds must satisfy -1 < low < high < 1 (got low={self.low}, high={self.high})")


def classify(value: float, th: Thresholds) -> str:
    return "low" if value < th.low else "high" if value >= th.high else "moderate"


def legend(th: Thresholds) -> list[dict]:
    return [{"category": "low", "label": "Low greenness", "color": COLORS["low"], "range": f"< {th.low:g}"},
            {"category": "moderate", "label": "Moderate or declining", "color": COLORS["moderate"], "range": f"{th.low:g} to {th.high:g}"},
            {"category": "high", "label": "High greenness", "color": COLORS["high"], "range": f">= {th.high:g}"}]


# ------------------------------------------------------------------------------------------------ NDVI
def ndvi(nir: float, red: float) -> float | None:
    """NDVI = (NIR - Red) / (NIR + Red). Returns None when undefined (zero sum or non-finite input).
    Scale-free: surface reflectance stored as DN x 10000 gives the same value as reflectance 0-1."""
    if not (math.isfinite(nir) and math.isfinite(red)):
        return None
    s = nir + red
    return None if s == 0 else (nir - red) / s


def ndvi_array(nir, red) -> np.ndarray:
    """Vectorised NDVI; undefined pixels are NaN."""
    nir, red = np.asarray(nir, dtype=float), np.asarray(red, dtype=float)
    s = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(s == 0, np.nan, (nir - red) / s)
    return np.where(np.isfinite(out), out, np.nan)


# ------------------------------------------------------------------------------------------------ geometry
_M_PER_DEG_LAT = 111_320.0


def _project(ring: list[list[float]]) -> list[tuple[float, float]]:
    """Local equirectangular projection to metres around the ring's mean latitude (accurate for field-sized shapes)."""
    lat0 = sum(p[1] for p in ring) / len(ring)
    kx = _M_PER_DEG_LAT * math.cos(math.radians(lat0))
    lon0 = ring[0][0]
    return [((p[0] - lon0) * kx, (p[1] - ring[0][1]) * _M_PER_DEG_LAT) for p in ring]


def _shoelace(pts: list[tuple[float, float]]) -> tuple[float, float, float]:
    """(signed area, cx, cy) of a closed polygon given WITHOUT repeating the first point."""
    a = cx = cy = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        cr = x0 * y1 - x1 * y0
        a += cr
        cx += (x0 + x1) * cr
        cy += (y0 + y1) * cr
    a /= 2
    return (a, cx / (6 * a), cy / (6 * a)) if a else (0.0, pts[0][0], pts[0][1])


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, c) -> bool:
    return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])


def _segments_intersect(p1, p2, p3, p4) -> bool:
    o1, o2, o3, o4 = _orient(p1, p2, p3), _orient(p1, p2, p4), _orient(p3, p4, p1), _orient(p3, p4, p2)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0) and 0 not in (o1, o2, o3, o4):
        return True
    return (o1 == 0 and _on_segment(p1, p2, p3)) or (o2 == 0 and _on_segment(p1, p2, p4)) or (o3 == 0 and _on_segment(p3, p4, p1)) or (o4 == 0 and _on_segment(p3, p4, p2))


def _self_intersects(pts: list[tuple[float, float]]) -> bool:
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):  # edges sharing a vertex
                continue
            if _segments_intersect(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                return True
    return False


def validate_coordinate(lat, lon) -> tuple[float, float]:
    """Validate a single lat/lon pair (degrees) and Sentinel-2 coverage."""
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError) as e:
        raise GeometryError("latitude and longitude must be numbers", "latitude") from e
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise GeometryError("latitude and longitude must be finite numbers", "latitude")
    if not -90 <= lat <= 90:
        raise GeometryError(f"latitude {lat} is outside -90..90", "latitude")
    if not -180 <= lon <= 180:
        raise GeometryError(f"longitude {lon} is outside -180..180 (GeoJSON order is [longitude, latitude])", "longitude")
    if not SENTINEL2_LAT_RANGE[0] <= lat <= SENTINEL2_LAT_RANGE[1]:
        raise GeometryError(f"latitude {lat} is outside Sentinel-2 coverage ({SENTINEL2_LAT_RANGE[0]:g} to {SENTINEL2_LAT_RANGE[1]:g})", "latitude")
    return lat, lon


def validate_polygon(geo, min_ha: float = 0.05, max_ha: float = 1000.0, max_vertices: int = 500) -> dict:
    """Validate a GeoJSON Polygon (or Feature wrapping one). Returns
    {"geometry": normalised closed polygon, "area_ha", "centroid": (lat, lon), "bbox": [w, s, e, n]}."""
    if isinstance(geo, dict) and geo.get("type") == "Feature":
        geo = geo.get("geometry")
    if not isinstance(geo, dict) or geo.get("type") != "Polygon":
        raise GeometryError("geometry must be a GeoJSON Polygon, e.g. {\"type\": \"Polygon\", \"coordinates\": [[[lon, lat], ...]]}")
    rings = geo.get("coordinates")
    if not isinstance(rings, list) or not rings:
        raise GeometryError("polygon has no coordinates")
    if len(rings) != 1:
        raise GeometryError("polygons with holes are not supported: provide one outer boundary")
    ring = []
    for p in rings[0]:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            raise GeometryError("each coordinate must be [longitude, latitude]")
        lat, lon = validate_coordinate(p[1], p[0])
        pt = [lon, lat]
        if not ring or ring[-1] != pt:  # drop consecutive duplicates
            ring.append(pt)
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]  # open ring; closed again on output
    if len(ring) < 3:
        raise GeometryError("a field boundary needs at least 3 distinct points")
    if len(ring) > max_vertices:
        raise GeometryError(f"too many vertices ({len(ring)}); the limit is {max_vertices}")
    lons, lats = [p[0] for p in ring], [p[1] for p in ring]
    if max(lons) - min(lons) > 0.25 or max(lats) - min(lats) > 0.25:
        raise GeometryError("the field extent is too large (over ~25 km); draw a single field")
    pts = _project(ring)
    if _self_intersects(pts):
        raise GeometryError("the boundary crosses itself; redraw it without overlapping edges")
    area_m2, cx, cy = _shoelace(pts)
    area_ha = abs(area_m2) / 10_000
    if area_ha < min_ha:
        raise GeometryError(f"field area {area_ha:.3f} ha is below the minimum {min_ha:g} ha (Sentinel-2 pixels are 10 m)")
    if area_ha > max_ha:
        raise GeometryError(f"field area {area_ha:.1f} ha exceeds the maximum {max_ha:g} ha")
    lat0 = sum(lats) / len(lats)
    centroid_lat = ring[0][1] + cy / _M_PER_DEG_LAT
    centroid_lon = ring[0][0] + cx / (_M_PER_DEG_LAT * math.cos(math.radians(lat0)))
    return {"geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]}, "area_ha": round(area_ha, 4),
            "centroid": (round(centroid_lat, 6), round(centroid_lon, 6)), "bbox": [min(lons), min(lats), max(lons), max(lats)]}


def circle_polygon(lat: float, lon: float, radius_m: float, n: int = 48) -> dict:
    """A regular n-gon approximating a circle (used for the 'point + radius' way of defining a field)."""
    lat, lon = validate_coordinate(lat, lon)
    if not 10 <= radius_m <= 3000:
        raise GeometryError("radius must be between 10 and 3000 metres", "radius_m")
    dlat = radius_m / _M_PER_DEG_LAT
    dlon = radius_m / (_M_PER_DEG_LAT * math.cos(math.radians(lat)))
    ring = [[round(lon + dlon * math.cos(2 * math.pi * i / n), 7), round(lat + dlat * math.sin(2 * math.pi * i / n), 7)] for i in range(n)]
    return {"type": "Polygon", "coordinates": [ring + [ring[0]]]}


# ------------------------------------------------------------------------------------------------ observations
@dataclass(frozen=True)
class Observation:
    date: date
    ndvi_mean: float
    valid_fraction: float   # share of the field covered by cloud-free pixels (0-1)
    scenes: int = 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["date"] = self.date.isoformat()
        return d


def aggregate_scene_rows(rows: list[dict], min_valid_fraction: float = 0.5) -> list[Observation]:
    """Per-scene statistics -> one Observation per date.
    A scene is discarded if its mean is missing / outside [-1, 1] or the cloud-free share of the field is below `min_valid_fraction`.
    Scenes of the same date (neighbouring tiles) are averaged weighted by their valid share; the combined share is capped at 1."""
    by_day: dict[date, list[tuple[float, float]]] = {}
    for r in rows:
        m, vf = r.get("ndvi_mean"), r.get("valid_fraction")
        if m is None or vf is None or not (math.isfinite(m) and math.isfinite(vf)) or not -1 <= m <= 1 or vf < min_valid_fraction:
            continue
        try:
            d = date.fromisoformat(str(r["date"])[:10])
        except (KeyError, ValueError):
            continue
        by_day.setdefault(d, []).append((float(m), min(1.0, float(vf))))
    out = []
    for d in sorted(by_day):
        parts = by_day[d]
        w = sum(v for _, v in parts)
        out.append(Observation(d, round(sum(m * v for m, v in parts) / w, 4), round(min(1.0, w), 3), len(parts)))
    return out


def trend_change(obs: list[Observation]) -> tuple[Observation | None, float | None]:
    """(previous observation, change in NDVI between the previous and the latest one)."""
    if len(obs) < 2:
        return None, None
    return obs[-2], round(obs[-1].ndvi_mean - obs[-2].ndvi_mean, 4)


# ------------------------------------------------------------------------------------------------ interpretation
CAVEATS = [
    "NDVI measures how green the vegetation looks from space. It is not a diagnosis of crop health.",
    "NDVI alone cannot identify a specific disease, pest, or nutrient or water problem.",
    "Low values can also mean bare soil, a recently sown or harvested field, a different growth stage, or a mixed pixel at the field edge.",
    "Satellite pixels are 10 m wide: small or narrow fields and edges are less reliable.",
    "Compare with the same crop at the same growth stage, and check the field on the ground before acting.",
]
_SUMMARY = {
    "high": "NDVI {v:.2f} indicates dense, green vegetation cover over most of the field.",
    "moderate": "NDVI {v:.2f} indicates moderate greenness. This is typical for an early or late growth stage or a partial canopy, and it can also reflect stress.",
    "low": "NDVI {v:.2f} indicates sparse or low greenness. Possible reasons: bare soil, a young or recently sown crop, a harvested or fallow field, or crop stress.",
}


def interpret(latest: Observation, previous: Observation | None, th: Thresholds) -> dict:
    cat = classify(latest.ndvi_mean, th)
    summary = _SUMMARY[cat].format(v=latest.ndvi_mean)
    change = None
    if previous is not None:
        change = round(latest.ndvi_mean - previous.ndvi_mean, 4)
        days = (latest.date - previous.date).days
        if change <= -0.10:
            summary += (f" Greenness fell by {abs(change):.2f} over {days} days. This can come from normal ripening, harvest, or stress; "
                        "inspect the field before drawing conclusions.")
        elif change >= 0.10:
            summary += f" Greenness rose by {change:.2f} over {days} days, consistent with canopy growth."
        else:
            summary += f" It is broadly unchanged ({change:+.2f}) since {previous.date.isoformat()}."
    return {"category": cat, "label": next(x["label"] for x in legend(th) if x["category"] == cat), "color": COLORS[cat], "summary": summary,
            "change": change, "caveats": CAVEATS, "disclaimer": "Vegetation greenness indicator, not a diagnosis."}


# ------------------------------------------------------------------------------------------------ provider interface
class SatelliteProvider(Protocol):
    def scene_stats(self, geometry: dict, start: date, end: date) -> list[dict]:
        """Per-scene rows {date, ndvi_mean, valid_fraction, cloud_pct?, tile?} for scenes intersecting `geometry`. Must raise SatelliteError on failure."""

    def map_layer(self, geometry: dict, obs_date: date, th: Thresholds, style: str) -> dict:
        """{tile_url, style, date, ...} for the NDVI of `obs_date` clipped to the field."""