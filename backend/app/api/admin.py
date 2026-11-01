from collections import Counter
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.security import require_admin
from ..db import get_db
from ..models.orm import ApiUsage, Prediction, User
from ..services.model_service import ModelService, get_model_service

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    since = datetime.now(timezone.utc) - timedelta(days=14)
    preds = db.scalars(select(Prediction).where(Prediction.created_at >= since)).all()
    per_day = Counter(p.created_at.strftime("%Y-%m-%d") for p in preds)
    days = [(datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(13, -1, -1)]
    by_crop = db.execute(select(Prediction.crop, func.count(), func.avg(Prediction.predicted_yield)).group_by(Prediction.crop).order_by(func.count().desc())).all()
    ep = db.execute(select(ApiUsage.path, func.count(), func.avg(ApiUsage.latency_ms)).group_by(ApiUsage.path).order_by(func.count().desc()).limit(10)).all()
    errors = db.scalar(select(func.count()).select_from(ApiUsage).where(ApiUsage.status >= 500)) or 0
    total_req = db.scalar(select(func.count()).select_from(ApiUsage)) or 0
    return {"users": db.scalar(select(func.count()).select_from(User)), "predictions": db.scalar(select(func.count()).select_from(Prediction)),
            "requests": total_req, "server_errors": errors,
            "predictions_per_day": [{"date": d, "count": per_day.get(d, 0)} for d in days],
            "by_crop": [{"crop": c, "count": n, "avg_yield": round(float(a), 2)} for c, n, a in by_crop],
            "endpoints": [{"path": p, "count": n, "avg_latency_ms": round(float(a), 1)} for p, n, a in ep]}


@router.get("/model-metrics")
def model_metrics(svc: ModelService = Depends(get_model_service)):
    m = svc.meta
    return {"version": m["version"], "best_model": m["best_model"], "metrics_test": m["metrics_test"], "metrics_cv": m["metrics_cv"],
            "comparison": m["comparison"], "per_crop_metrics": m["per_crop_metrics"], "uncertainty": m["uncertainty"],
            "best_params": m["best_params"], "cleaning": m["cleaning"]}


@router.get("/users")
def users(db: Session = Depends(get_db)):
    rows = db.execute(select(User, func.count(Prediction.id)).outerjoin(Prediction).group_by(User.id).order_by(User.id)).all()
    return [{"id": u.id, "name": u.name, "email": u.email, "role": u.role, "created_at": u.created_at, "predictions": n} for u, n in rows]
