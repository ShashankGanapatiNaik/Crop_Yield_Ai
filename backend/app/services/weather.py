"""Open-Meteo client (free, no API key). Builds auto-fill values for the Predict form.

The model needs *whole-season* rainfall and mean season temperature. A short recent window is a poor
proxy, so when a season is given we use the average of the last three COMPLETE seasons from the
Open-Meteo historical archive (ERA5 reanalysis); the 90-day window is only a fallback.
Caveat: gridded reanalysis (~25 km) tends to UNDER-estimate orographic rain on the Western Ghats and the
west coast, so a local rain-gauge or IMD figure should override it when known.
"""
import logging
import time
from datetime import date, timedelta

import httpx

from ..core.config import get_settings

log = logging.getLogger("weather")
URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
_cache: dict[tuple, tuple[float, dict]] = {}


class WeatherError(Exception):
    pass


def summarise(data: dict, lat: float, lon: float) -> dict:
    """Pure function (unit-testable): turn an Open-Meteo forecast payload into model-ready inputs."""
    d = data.get("daily", {})
    times = d.get("time", [])
    n_past = max(0, len(times) - 7)  # last 7 entries are the forecast
    past = slice(0, n_past)
    avg = lambda xs: round(sum(v for v in xs if v is not None) / max(1, len([v for v in xs if v is not None])), 1)  # noqa: E731
    rain_past = sum(v for v in d.get("precipitation_sum", [])[past] if v is not None)
    cur = data.get("current", {})
    return {"latitude": lat, "longitude": lon, "period_days": n_past,
            "temperature_c": avg(d.get("temperature_2m_mean", [])[past]) if n_past else cur.get("temperature_2m"),
            "humidity_pct": avg(d.get("relative_humidity_2m_mean", [])[past]) if n_past else cur.get("relative_humidity_2m"),
            "rainfall_mm": round(rain_past, 1), "rainfall_last_90d_mm": round(rain_past, 1),
            "current_temperature_c": cur.get("temperature_2m"), "current_humidity_pct": cur.get("relative_humidity_2m"),
            "next_7d_rainfall_mm": round(sum(v for v in d.get("precipitation_sum", [])[n_past:] if v is not None), 1),
            "season": None, "basis": f"total of the last {n_past} days only (not the whole season)", "seasons_used": [],
            "source": "open-meteo.com",
            "note": "Seasonal rainfall can differ a lot from recent rain. Enter your local rain-gauge or IMD value if you know it."}


def season_windows(season: str, today: date, n: int = 3, lag_days: int = 7) -> list[tuple[date, date]]:
    """The last n COMPLETE windows of a season (newest first). Kharif Jun-Oct, Rabi Nov-Feb, Summer Mar-May."""
    cutoff = today - timedelta(days=lag_days)  # archive data lags a few days
    out: list[tuple[date, date]] = []
    for y in range(today.year, today.year - 10, -1):
        if season == "Kharif":
            s, e = date(y, 6, 1), date(y, 10, 31)
        elif season == "Rabi":
            s, e = date(y, 11, 1), date(y + 1, 3, 1) - timedelta(days=1)
        elif season == "Summer":
            s, e = date(y, 3, 1), date(y, 5, 31)
        else:
            return []
        if e <= cutoff:
            out.append((s, e))
        if len(out) == n:
            break
    return out


def aggregate_seasons(times: list[str], precip: list, temp: list, wins: list[tuple[date, date]], min_cover: float = 0.9) -> list[dict]:
    """Per-window rainfall total (mm) and mean temperature. Windows with <90% of days present are skipped."""
    res = []
    for s, e in wins:
        lo, hi = s.isoformat(), e.isoformat()
        idx = [i for i, t in enumerate(times) if lo <= t <= hi]
        days = (e - s).days + 1
        valid = [i for i in idx if i < len(precip) and precip[i] is not None]
        if len(valid) < min_cover * days:
            continue
        tv = [temp[i] for i in idx if i < len(temp) and temp[i] is not None]
        res.append({"year": s.year, "rainfall_mm": round(sum(precip[i] for i in valid), 1), "temperature_c": round(sum(tv) / len(tv), 1) if tv else None})
    return res


def _seasonal(lat: float, lon: float, season: str, today: date) -> list[dict]:
    wins = season_windows(season, today)
    if not wins:
        return []
    params = {"latitude": lat, "longitude": lon, "timezone": "auto", "daily": "precipitation_sum,temperature_2m_mean",
              "start_date": min(w[0] for w in wins).isoformat(), "end_date": max(w[1] for w in wins).isoformat()}
    r = httpx.get(ARCHIVE, params=params, timeout=20)
    r.raise_for_status()
    d = r.json().get("daily", {})
    return aggregate_seasons(d.get("time", []), d.get("precipitation_sum", []), d.get("temperature_2m_mean", []), wins)


def fetch_weather(lat: float, lon: float, season: str | None = None) -> dict:
    ttl = get_settings().weather_cache_seconds
    key = (round(lat, 2), round(lon, 2), season)
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

    if season:
        try:
            seasons = _seasonal(lat, lon, season, date.today())
        except (httpx.HTTPError, ValueError) as e:
            log.warning("Open-Meteo archive failure: %s", e)
            seasons = []
        if seasons:
            rain = sum(s["rainfall_mm"] for s in seasons) / len(seasons)
            temps = [s["temperature_c"] for s in seasons if s["temperature_c"] is not None]
            out.update(season=season, seasons_used=seasons, rainfall_mm=round(rain, 1),
                       temperature_c=round(sum(temps) / len(temps), 1) if temps else out["temperature_c"],
                       basis=f"average {season} season rainfall of {', '.join(str(s['year']) for s in seasons)} (ERA5 reanalysis)")
    _cache[key] = (time.time(), out)
    return out