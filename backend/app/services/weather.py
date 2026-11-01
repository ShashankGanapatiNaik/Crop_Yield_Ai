"""Open-Meteo client (free, no API key). Builds auto-fill values for the Predict form."""
import logging
import time

import httpx

from ..core.config import get_settings

log = logging.getLogger("weather")
URL = "https://api.open-meteo.com/v1/forecast"
_cache: dict[tuple, tuple[float, dict]] = {}


class WeatherError(Exception):
    pass


def summarise(data: dict, lat: float, lon: float) -> dict:
    """Pure function (unit-testable): turn an Open-Meteo payload into model-ready inputs."""
    d = data.get("daily", {})
    times = d.get("time", [])
    n_past = max(0, len(times) - 7)  # last 7 entries are the forecast
    past = slice(0, n_past)
    avg = lambda xs: round(sum(v for v in xs if v is not None) / max(1, len([v for v in xs if v is not None])), 1)
    rain_past = sum(v for v in d.get("precipitation_sum", [])[past] if v is not None)
    cur = data.get("current", {})
    return {"latitude": lat, "longitude": lon, "period_days": n_past,
            "temperature_c": avg(d.get("temperature_2m_mean", [])[past]) if n_past else cur.get("temperature_2m"),
            "humidity_pct": avg(d.get("relative_humidity_2m_mean", [])[past]) if n_past else cur.get("relative_humidity_2m"),
            "rainfall_mm": round(rain_past, 1),
            "current_temperature_c": cur.get("temperature_2m"), "current_humidity_pct": cur.get("relative_humidity_2m"),
            "next_7d_rainfall_mm": round(sum(v for v in d.get("precipitation_sum", [])[n_past:] if v is not None), 1),
            "source": "open-meteo.com",
            "note": f"Averages and rainfall total cover the previous {n_past} days. Seasonal rainfall may differ - adjust if you know better."}


def fetch_weather(lat: float, lon: float) -> dict:
    ttl = get_settings().weather_cache_seconds
    key = (round(lat, 2), round(lon, 2))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    params = {"latitude": lat, "longitude": lon, "timezone": "auto", "past_days": 90, "forecast_days": 7,
              "current": "temperature_2m,relative_humidity_2m",
              "daily": "precipitation_sum,temperature_2m_mean,relative_humidity_2m_mean"}
    try:
        r = httpx.get(URL, params=params, timeout=12)
        r.raise_for_status()
        out = summarise(r.json(), lat, lon)
    except (httpx.HTTPError, ValueError) as e:
        log.warning("Open-Meteo failure: %s", e)
        raise WeatherError("Weather service is unavailable right now. Please enter values manually.") from e
    _cache[key] = (time.time(), out)
    return out
