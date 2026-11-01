"""Place-name lookup (reverse + search) via OpenStreetMap Nominatim. No API key needed.

Called from the backend (not the browser) because Nominatim's usage policy requires an identifying
User-Agent, which browsers cannot set, and because we can cache results (policy: max ~1 request/second).
"""
import logging
import time

import httpx

from ml.src.config import REGIONS

log = logging.getLogger("geocode")
BASE = "https://nominatim.openstreetmap.org"
HEADERS = {"User-Agent": "CropYieldAI/1.0 (final-year project; contact: admin@cropai.app)", "Accept-Language": "en"}
TTL = 3600
_cache: dict[tuple, tuple[float, object]] = {}
_REGION_BY_LOWER = {r.lower(): r for r in REGIONS}


class GeocodeError(Exception):
    pass


def parse_place(item: dict) -> dict:
    """Turn one Nominatim result into {lat, lon, label, ...}. Pure function (unit-tested)."""
    a = item.get("address", {}) or {}
    locality = next((a[k] for k in ("village", "hamlet", "town", "city", "suburb", "neighbourhood", "municipality") if a.get(k)), "")
    district = a.get("state_district") or a.get("county") or ""
    state = a.get("state", "")
    parts = [p for p in (locality, district, state) if p]
    seen: list[str] = []
    for p in parts:
        if p not in seen:
            seen.append(p)
    label = ", ".join(seen) or item.get("display_name", "Unknown place")
    return {"lat": float(item["lat"]), "lon": float(item["lon"]), "label": label, "display_name": item.get("display_name", label),
            "locality": locality, "district": district, "state": state, "country": a.get("country", ""),
            "region": _REGION_BY_LOWER.get(state.lower())}


def _get(path: str, params: dict):
    key = (path, tuple(sorted(params.items())))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < TTL:
        return hit[1]
    try:
        r = httpx.get(f"{BASE}{path}", params={**params, "format": "jsonv2", "addressdetails": 1}, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        log.warning("Nominatim failure: %s", e)
        raise GeocodeError("Place lookup is unavailable right now.") from e
    _cache[key] = (time.time(), data)
    return data


def reverse(lat: float, lon: float) -> dict:
    data = _get("/reverse", {"lat": round(lat, 5), "lon": round(lon, 5), "zoom": 16})
    if not isinstance(data, dict) or "error" in data:
        raise GeocodeError("No place found at this point.")
    data.setdefault("lat", lat)
    data.setdefault("lon", lon)
    return parse_place(data)


def search(q: str) -> list[dict]:
    data = _get("/search", {"q": q.strip(), "limit": 6, "countrycodes": "in"})
    return [parse_place(i) for i in data] if isinstance(data, list) else []