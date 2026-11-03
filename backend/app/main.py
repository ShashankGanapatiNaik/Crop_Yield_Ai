import logging
import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import select

from .api import admin, auth, predict
from .core.config import get_settings
from .core.limiter import limiter
from .core.logging import setup_logging
from .core.security import hash_password
from .db import Base, SessionLocal, engine
from .models.orm import ApiUsage, User
from .services.model_service import InputError, get_model_service

setup_logging()
log = logging.getLogger("app")
settings = get_settings()


def seed_admin():
    with SessionLocal() as db:
        if not db.scalar(select(User).where(User.email == settings.admin_email)):
            db.add(User(name="Administrator", email=settings.admin_email, hashed_password=hash_password(settings.admin_password), role="admin"))
            db.commit()
            log.info("Seeded admin user %s", settings.admin_email)


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version="1.0.0", description="AI-based crop yield prediction and decision support API")
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_list, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def usage_logger(request: Request, call_next):
        t0 = time.perf_counter()
        try:
            resp = await call_next(request)
            status = resp.status_code
        except Exception:
            log.exception("Unhandled error on %s %s", request.method, request.url.path)
            resp, status = JSONResponse({"detail": "Internal server error"}, status_code=500), 500
        ms = (time.perf_counter() - t0) * 1000
        log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, status, ms)
        if not request.url.path.startswith(("/docs", "/openapi", "/health")):
            path = "/history/{id}" if request.url.path.startswith("/history/") else "/report/{id}" if request.url.path.startswith("/report/") else request.url.path
            with SessionLocal() as db:
                db.add(ApiUsage(path=path[:120], method=request.method, status=status, latency_ms=ms)); db.commit()
        return resp

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        errs = [{"field": ".".join(str(x) for x in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": "Validation failed", "errors": errs})

    @app.exception_handler(InputError)
    async def input_error_handler(request: Request, exc: InputError):
        return JSONResponse(status_code=422, content={"detail": "Validation failed", "errors": exc.errors})

    @app.on_event("startup")
    def startup():
        Base.metadata.create_all(engine)
        seed_admin()
        try:
            get_model_service()
        except FileNotFoundError as e:
            log.error("%s", e)

    @app.get("/health", tags=["meta"])
    def health():
        try:
            svc = get_model_service()
            return {"status": "ok", "model_version": svc.version}
        except FileNotFoundError:
            return JSONResponse({"status": "degraded", "detail": "model not trained"}, status_code=503)

    for r in (auth.router, predict.router, admin.router):
        app.include_router(r)
    return app


app = create_app()