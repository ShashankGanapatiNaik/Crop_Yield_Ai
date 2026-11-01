import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.src.config import (
    CROP_PROFILE,
    ECONOMICS,
    IRRIGATION,
    REGIONS,
    REPORT_DIR,
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


@router.get("/geocode/reverse")
@limiter.limit("60/minute")
def geocode_reverse(request: Request, lat: float = Query(ge=-90, le=90), lon: float = Query(ge=-180, le=180)):
    """Exact place name for a map point."""
    try:
        return geocode.reverse(lat, lon)
    except geocode.GeocodeError as e:
        raise HTTPException(502, str(e)) from e


@router.get("/geocode/search")
@limiter.limit("60/minute")
def geocode_search(request: Request, q: str = Query(min_length=2, max_length=120)):
    """Search a place by name (India)."""
    try:
        return {"results": geocode.search(q)}
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
def crops():
    return {"crops": [{"name": c, "seasons": p["seasons"], "optimum_ph": p["ph"], "optimum_temp_c": p["temp"][0], "optimum_water_mm": p["rain"][0],
                       **ECONOMICS[c]} for c, p in CROP_PROFILE.items()],
            "regions": REGIONS, "seasons": SEASONS, "soil_types": SOILS, "irrigation": IRRIGATION}


@router.get("/model-info")
def model_info(svc: ModelService = Depends(get_model_service)):
    m = svc.meta
    keys = ["version", "trained_at", "data_source", "best_model", "selection_criterion", "explainer_model", "n_train", "n_test", "cv_folds",
            "optuna_trials", "metrics_test", "metrics_cv", "comparison", "per_crop_metrics", "uncertainty", "global_importance", "residual_summary", "cleaning"]
    return {k: m[k] for k in keys if k in m}


PLOTS = {"model_comparison", "residual_analysis", "shap_summary_beeswarm", "shap_global_importance", "shap_local_example", "prediction_intervals"}


@router.get("/model-plots/{name}")
def model_plot(name: str):
    """Serve training diagnostics (whitelisted names only, so no path traversal is possible)."""
    f = REPORT_DIR / "eval" / f"{name}.png"
    if name not in PLOTS or not f.exists():
        raise HTTPException(404, "Plot not found")
    return FileResponse(f, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})