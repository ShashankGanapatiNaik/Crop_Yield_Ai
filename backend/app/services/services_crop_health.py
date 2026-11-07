"""Orchestration for crop-health requests: cache in the database, call the satellite provider only when needed,
interpret the result. No Earth Engine code here (that is earth_engine.py) and no HTTP code (that is api/crop_health.py)."""
import logging
from datetime import UTC, date, datetime, timedelta

from sqlalchemy.orm import Session

from ..core.config import Settings
from ..models.orm import FarmField, NdviObservation
from . import satellite as sat
from .earth_engine import COLLECTION
from .satellite import Observation, SatelliteError, SatelliteProvider, Thresholds

log = logging.getLogger("crop_health")

SOURCE = {"satellite": "Sentinel-2 (ESA Copernicus)", "dataset": COLLECTION, "index": "NDVI = (B8 - B4) / (B8 + B4)", "resolution_m": 10,
          "cloud_mask": "Scene Classification Layer (cloud, shadow, cirrus, snow removed)", "revisit": "about every 5 days, when not cloudy",
          "attribution": "Contains modified Copernicus Sentinel data"}


def thresholds_from_settings(s: Settings) -> Thresholds:
    return Thresholds(low=s.ndvi_low, high=s.ndvi_high)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _aware(dt: datetime | None) -> datetime | None:
    return dt if dt is None or dt.tzinfo else dt.replace(tzinfo=UTC)  # SQLite returns naive datetimes


def needs_sync(field: FarmField, days: int, refresh: bool, cache_hours: float, now: datetime | None = None) -> bool:
    if refresh or field.last_synced_at is None or (field.last_sync_days or 0) < days:
        return True
    return (now or _utcnow()) - _aware(field.last_synced_at) > timedelta(hours=cache_hours)


def load_observations(db: Session, field: FarmField, since: date) -> list[Observation]:
    rows = (db.query(NdviObservation).filter(NdviObservation.field_id == field.id, NdviObservation.obs_date >= since).order_by(NdviObservation.obs_date).all())
    return [Observation(r.obs_date, r.ndvi_mean, r.valid_fraction, r.scenes) for r in rows]


def sync_field(db: Session, field: FarmField, provider: SatelliteProvider, days: int, s: Settings, today: date | None = None) -> list[Observation]:
    """Fetch scene statistics for the window, keep only trustworthy observations, replace the cached rows of that window."""
    today = today or _utcnow().date()
    start = today - timedelta(days=days)
    rows = provider.scene_stats(field.geometry, start, today)          # raises SatelliteError: nothing is stored on failure
    obs = sat.aggregate_scene_rows(rows, s.satellite_min_valid_fraction)
    db.query(NdviObservation).filter(NdviObservation.field_id == field.id, NdviObservation.obs_date >= start).delete()
    now = _utcnow()
    for o in obs:
        db.add(NdviObservation(field_id=field.id, obs_date=o.date, ndvi_mean=o.ndvi_mean, valid_fraction=o.valid_fraction, scenes=o.scenes, fetched_at=now))
    field.last_synced_at, field.last_sync_days, field.last_sync_status = now, days, ("ok" if obs else "no_data")
    db.commit()
    log.info("field %s synced: %d scenes -> %d usable observations", field.id, len(rows), len(obs))
    return obs


def _ensure_synced(db: Session, field: FarmField, provider: SatelliteProvider, days: int, refresh: bool, s: Settings) -> dict:
    """Returns {"from_cache": bool, "stale": bool, "warning": str|None}. If the provider fails but cached data exists, serve it flagged as stale."""
    start = _utcnow().date() - timedelta(days=days)
    if not needs_sync(field, days, refresh, s.satellite_cache_hours):
        return {"from_cache": True, "stale": False, "warning": None}
    try:
        sync_field(db, field, provider, days, s)
        return {"from_cache": False, "stale": False, "warning": None}
    except SatelliteError as e:
        if load_observations(db, field, start):
            log.warning("serving cached observations for field %s after provider error: %s", field.id, e)
            return {"from_cache": True, "stale": True, "warning": f"Could not refresh from the satellite service ({e}). Showing the last saved observations."}
        raise


def _obs_out(o: Observation, th: Thresholds) -> dict:
    cat = sat.classify(o.ndvi_mean, th)
    return {**o.to_dict(), "category": cat, "color": sat.COLORS[cat], "label": next(x["label"] for x in sat.legend(th) if x["category"] == cat)}


def _base(field: FarmField, days: int, th: Thresholds, meta: dict) -> dict:
    today = _utcnow().date()
    return {"field": {"id": field.id, "name": field.name, "area_ha": field.area_ha}, "window": {"start": (today - timedelta(days=days)).isoformat(), "end": today.isoformat(), "days": days},
            "thresholds": {"low": th.low, "high": th.high, "note": "Approximate class boundaries for colouring; they vary by crop and growth stage."},
            "legend": sat.legend(th), "last_synced_at": field.last_synced_at.isoformat() if field.last_synced_at else None, "source": SOURCE, **meta}


def no_data_reason(days: int, min_valid: float) -> str:
    return (f"No cloud-free Sentinel-2 observation covering at least {min_valid:.0%} of this field was found in the last {days} days. "
            "This is common in cloudy seasons (monsoon). Try a longer period, or check again in a few days. No values are estimated.")


def latest_result(db: Session, field: FarmField, provider: SatelliteProvider, days: int, refresh: bool, style: str, s: Settings) -> dict:
    th = thresholds_from_settings(s)
    meta = _ensure_synced(db, field, provider, days, refresh, s)
    obs = load_observations(db, field, _utcnow().date() - timedelta(days=days))
    out = _base(field, days, th, meta)
    if not obs:
        return {**out, "status": "no_data", "reason": no_data_reason(days, s.satellite_min_valid_fraction), "observation": None, "previous": None, "interpretation": None,
                "layer": None, "layer_error": None}
    latest, (prev, _chg) = obs[-1], sat.trend_change(obs)
    layer, layer_error = None, None
    try:
        layer = {**provider.map_layer(field.geometry, latest.date, th, style), "bounds": _bounds(field.geometry)}
    except SatelliteError as e:       # the number and chart are still useful without the picture
        layer_error = str(e)
    return {**out, "status": "ok", "reason": None, "observation": _obs_out(latest, th), "previous": _obs_out(prev, th) if prev else None,
            "interpretation": sat.interpret(latest, prev, th), "layer": layer, "layer_error": layer_error}


def history_result(db: Session, field: FarmField, provider: SatelliteProvider, days: int, refresh: bool, s: Settings) -> dict:
    th = thresholds_from_settings(s)
    meta = _ensure_synced(db, field, provider, days, refresh, s)
    obs = load_observations(db, field, _utcnow().date() - timedelta(days=days))
    out = _base(field, days, th, meta)
    return {**out, "status": "ok" if obs else "no_data", "reason": None if obs else no_data_reason(days, s.satellite_min_valid_fraction),
            "count": len(obs), "observations": [_obs_out(o, th) for o in obs]}


def _bounds(geometry: dict) -> list[list[float]]:
    ring = geometry["coordinates"][0]
    lons, lats = [p[0] for p in ring], [p[1] for p in ring]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]   # Leaflet order: [[south, west], [north, east]]