import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.src.config import (
    CROP_PROFILE,
    ECONOMICS,
    IRRIGATION,
    REGIONS,
    SEASONS,
    SOILS,
)

from ..core.config import get_settings
from ..core.limiter import limiter
from ..core.security import get_current_user
from ..db import get_db
from ..models.orm import Prediction, User
from ..schemas import CompareIn, PredictIn, PredictionOut, WhatIfIn
from ..services import geocode, pdf_report, weather
from ..services.model_service import ModelService, get_model_service


def get_optional_service() -> ModelService | None:
    """For public reference endpoints: fall back to static lists when no model is trained yet."""
    try:
        return get_model_service()
    except FileNotFoundError:
        return None


log = logging.getLogger("api")
router = APIRouter(tags=["prediction"])
PRED = get_settings().rate_limit_predict


def _row(p: Prediction) -> dict:
    return {"id": p.id, "crop": p.crop, "predicted_yield_t_ha": p.predicted_yield, "lower_t_ha": p.lower, "upper_t_ha": p.upper,
            "total_production_t": p.total_production, "profit": p.profit, "area_ha": p.inputs.get("area_ha"),
            "created_at": p.created_at, "model_version": p.model_version}


@router.post("/predict", response_model=PredictionOut)
@limiter.limit(PRED)
def predict(request: Request, body: PredictIn, db: Session = Depends(get_db), user: User = Depends(get_current_user),
            svc: ModelService = Depends(get_model_service)):
    data = body.model_dump()
    crop = data.pop("crop")
    out = svc.full_prediction(data, crop)
    rec = Prediction(user_id=user.id, crop=crop, inputs=out["inputs"], predicted_yield=out["predicted_yield_t_ha"], lower=out["lower_t_ha"],
                     upper=out["upper_t_ha"], total_production=out["total_production_t"], profit=out["economics"]["profit"],
                     result=out, model_version=svc.version)
    db.add(rec); db.commit(); db.refresh(rec)
    out.update(id=rec.id, created_at=rec.created_at)
    return out


@router.post("/compare-crops")
@limiter.limit(PRED)
def compare_crops(request: Request, body: CompareIn, user: User = Depends(get_current_user), svc: ModelService = Depends(get_model_service)):
    data = body.model_dump()
    crops = data.pop("crops")
    return {**svc.compare(data, crops), "inputs": data, "model_version": svc.version}


@router.post("/what-if")
@limiter.limit("120/minute")
def what_if(request: Request, body: WhatIfIn, user: User = Depends(get_current_user), svc: ModelService = Depends(get_model_service)):
    base = body.base.model_dump()
    crop = base.pop("crop")
    bad = [k for k in body.changes if k not in base]
    if bad:
        raise HTTPException(422, f"Unknown parameter(s) to change: {', '.join(bad)}")
    res = svc.what_if(base, crop, body.changes, body.sweep.model_dump() if body.sweep else None)
    sc = res["scenario"]["inputs"]
    # re-validate the modified scenario with the same rules as /predict
    try:
        PredictIn(**{**sc, "crop": crop})
    except Exception as e:  # pydantic ValidationError
        raise HTTPException(422, f"Scenario values out of range: {e}") from e
    return {"crop": crop, **res, "model_version": svc.version}


@router.get("/weather")
@limiter.limit("30/minute")
def get_weather(request: Request, lat: float = Query(ge=-90, le=90), lon: float = Query(ge=-180, le=180),
                season: str | None = Query(default=None, description="Kharif | Rabi | Summer: use the typical whole-season rainfall")):
    if season is not None:
        season = next((x for x in SEASONS if x.lower() == season.lower()), None)
        if season is None:
            raise HTTPException(422, f"season must be one of {', '.join(SEASONS)}")
    try:
        return weather.fetch_weather(lat, lon, season)
    except weather.WeatherError as e:
        raise HTTPException(502, str(e)) from e


def _match_region(place: dict) -> dict:
    """Map the place's state onto the regions the ACTIVE model knows (synthetic: 10 states; real: all states in the data)."""
    svc = get_optional_service()
    if svc and place.get("state"):
        place = {**place, "region": svc.canonical("region", place["state"]) or place.get("region") if svc.data_mode == "synthetic" else svc.canonical("region", place["state"])}
    return place


@router.get("/geocode/reverse")
@limiter.limit("60/minute")
def geocode_reverse(request: Request, lat: float = Query(ge=-90, le=90), lon: float = Query(ge=-180, le=180)):
    """Exact place name for a map point."""
    try:
        return _match_region(geocode.reverse(lat, lon))
    except geocode.GeocodeError as e:
        raise HTTPException(502, str(e)) from e


@router.get("/geocode/search")
@limiter.limit("60/minute")
def geocode_search(request: Request, q: str = Query(min_length=2, max_length=120)):
    """Search a place by name (India)."""
    try:
        return {"results": [_match_region(p) for p in geocode.search(q)]}
    except geocode.GeocodeError as e:
        raise HTTPException(502, str(e)) from e


@router.get("/history")
def history(skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    q = select(Prediction).where(Prediction.user_id == user.id).order_by(Prediction.created_at.desc())
    rows = db.scalars(q.offset(skip).limit(limit)).all()
    total = db.query(Prediction).filter(Prediction.user_id == user.id).count()
    return {"total": total, "items": [_row(r) for r in rows]}


def _own(db: Session, user: User, pid: int) -> Prediction:
    p = db.get(Prediction, pid)
    if not p or (p.user_id != user.id and user.role != "admin"):
        raise HTTPException(404, "Prediction not found")
    return p


@router.get("/history/{pid}", response_model=PredictionOut)
def history_item(pid: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    p = _own(db, user, pid)
    return {**p.result, "id": p.id, "created_at": p.created_at}


@router.delete("/history/{pid}", status_code=204)
def delete_history(pid: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    db.delete(_own(db, user, pid)); db.commit()
    return Response(status_code=204)


@router.get("/report/{pid}")
def report(pid: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    p = _own(db, user, pid)
    pdf = pdf_report.build_pdf(p, p.user.name)
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="yield-report-{p.id}.pdf"'})


@router.get("/crops")
def crops(svc: ModelService | None = Depends(get_optional_service)):
    """Reference lists for the form. Reflects the ACTIVE model: a district-level model has no soil-type or irrigation inputs."""
    names = svc.categories("crop") if svc else list(CROP_PROFILE)
    out = {"crops": [{"name": c, "seasons": CROP_PROFILE[c]["seasons"], "optimum_ph": CROP_PROFILE[c]["ph"], "optimum_temp_c": CROP_PROFILE[c]["temp"][0],
                      "optimum_water_mm": CROP_PROFILE[c]["rain"][0], **ECONOMICS[c]} for c in names if c in CROP_PROFILE],
           "regions": svc.categories("region") if svc else REGIONS, "seasons": svc.categories("season") if svc else SEASONS,
           "soil_types": svc.categories("soil_type") if svc else SOILS, "irrigation": svc.categories("irrigation") if svc else IRRIGATION}
    if svc:
        out.update(svc.describe())
    else:
        out.update(data_mode="synthetic", model_name=None, input_schema=None, provenance=None, districts=None)
    return out


@router.get("/model-info")
def model_info(svc: ModelService = Depends(get_model_service)):
    m = svc.meta
    keys = ["version", "trained_at", "data_source", "best_model", "selection_criterion", "explainer_model", "n_train", "n_test", "cv_folds",
            "optuna_trials", "metrics_test", "metrics_cv", "comparison", "per_crop_metrics", "uncertainty", "global_importance", "residual_summary", "cleaning",
            "data_mode", "name", "provenance", "input_schema", "split", "metrics_val", "per_region_metrics", "synthetic_reference", "final_fit"]
    return {k: m[k] for k in keys if k in m}


PLOTS = {"model_comparison", "residual_analysis", "shap_summary_beeswarm", "shap_global_importance", "shap_local_example", "prediction_intervals"}


@router.get("/model-plots/{name}")
def model_plot(name: str, svc: ModelService | None = Depends(get_optional_service)):
    """Serve training diagnostics of the ACTIVE model (whitelisted names only, so no path traversal is possible)."""
    root = Path(get_settings().reports_dir)
    folder = root / "real" / svc.name if svc and svc.data_mode == "real" else root / "eval"
    f = folder / f"{name}.png"
    if name not in PLOTS or not f.exists():
        raise HTTPException(404, "Plot not found")
    return FileResponse(f, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})