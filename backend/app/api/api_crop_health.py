"""Satellite crop-health endpoints (Sentinel-2 NDVI via Google Earth Engine)."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.limiter import limiter
from ..core.security import get_current_user
from ..db import get_db
from ..models.orm import FarmField, User
from ..schemas import FieldCreate, FieldOut
from ..services import crop_health as svc
from ..services import satellite as sat
from ..services.earth_engine import COLLECTION, get_provider
from ..services.model_service import InputError
from ..services.satellite import GeometryError, SatelliteError, SatelliteProvider

log = logging.getLogger("api.crop_health")
router = APIRouter(prefix="/crop-health", tags=["crop-health"])
RATE = get_settings().rate_limit_satellite


def get_satellite_provider() -> SatelliteProvider:
    """Overridable in tests (app.dependency_overrides). The real provider initialises Earth Engine lazily on first use."""
    return get_provider()


def _owned(db: Session, user: User, field_id: int) -> FarmField:
    f = db.get(FarmField, field_id)
    if not f or f.user_id != user.id:
        raise HTTPException(404, "Field not found")
    return f


def _days(days: int | None) -> int:
    s = get_settings()
    d = days if days is not None else s.satellite_default_days
    if not 7 <= d <= s.satellite_max_days:
        raise InputError([{"field": "days", "message": f"days must be between 7 and {s.satellite_max_days}"}])
    return d


def _satellite_http(e: SatelliteError) -> HTTPException:
    return HTTPException(e.status_code, str(e))


@router.get("/status")
def status():
    """Whether satellite monitoring is configured on this server (no network call, no secrets)."""
    s = get_settings()
    th = svc.thresholds_from_settings(s)
    return {"configured": s.gee_configured, "provider": "Google Earth Engine", "dataset": COLLECTION, "thresholds": {"low": th.low, "high": th.high},
            "limits": {"min_field_ha": s.field_min_ha, "max_field_ha": s.field_max_ha, "max_fields_per_user": s.max_fields_per_user,
                       "default_days": s.satellite_default_days, "max_days": s.satellite_max_days}, "legend": sat.legend(th)}


@router.post("/fields", response_model=FieldOut, status_code=201)
@limiter.limit(RATE)
def create_field(request: Request, body: FieldCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    s = get_settings()
    if db.scalar(select(func.count()).select_from(FarmField).where(FarmField.user_id == user.id)) >= s.max_fields_per_user:
        raise InputError([{"field": "name", "message": f"field limit reached ({s.max_fields_per_user}); delete one first"}])
    try:
        if body.geometry is not None:
            v, source, radius = sat.validate_polygon(body.geometry, s.field_min_ha, s.field_max_ha), "polygon", None
        else:
            lat, lon = sat.validate_coordinate(body.latitude, body.longitude)
            v, source, radius = sat.validate_polygon(sat.circle_polygon(lat, lon, body.radius_m), s.field_min_ha, s.field_max_ha), "point_radius", body.radius_m
    except GeometryError as e:
        raise InputError([{"field": e.field, "message": str(e)}]) from e
    f = FarmField(user_id=user.id, name=body.name.strip(), geometry=v["geometry"], centroid_lat=v["centroid"][0], centroid_lon=v["centroid"][1], area_ha=v["area_ha"],
                  source=source, radius_m=radius)
    db.add(f); db.commit(); db.refresh(f)
    return f


@router.get("/fields", response_model=list[FieldOut])
def list_fields(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return db.scalars(select(FarmField).where(FarmField.user_id == user.id).order_by(FarmField.created_at.desc())).all()


@router.get("/fields/{field_id}", response_model=FieldOut)
def get_field(field_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return _owned(db, user, field_id)


@router.delete("/fields/{field_id}", status_code=204)
def delete_field(field_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    db.delete(_owned(db, user, field_id)); db.commit()
    return Response(status_code=204)


@router.get("/fields/{field_id}/latest")
@limiter.limit(RATE)
def latest(request: Request, field_id: int, days: int | None = Query(default=None, description="Search window in days"),
           refresh: bool = Query(default=False, description="Ignore the cache and ask the satellite service again"),
           style: str = Query(default="classes", pattern="^(classes|gradient)$"),
           db: Session = Depends(get_db), user: User = Depends(get_current_user), provider: SatelliteProvider = Depends(get_satellite_provider)):
    f = _owned(db, user, field_id)
    try:
        return svc.latest_result(db, f, provider, _days(days), refresh, style, get_settings())
    except SatelliteError as e:
        raise _satellite_http(e) from e


@router.get("/fields/{field_id}/history")
@limiter.limit(RATE)
def history(request: Request, field_id: int, days: int | None = Query(default=None), refresh: bool = Query(default=False),
            db: Session = Depends(get_db), user: User = Depends(get_current_user), provider: SatelliteProvider = Depends(get_satellite_provider)):
    f = _owned(db, user, field_id)
    try:
        return svc.history_result(db, f, provider, _days(days), refresh, get_settings())
    except SatelliteError as e:
        raise _satellite_http(e) from e