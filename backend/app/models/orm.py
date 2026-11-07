from datetime import UTC, date, datetime

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db import Base


def now():
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="farmer")  # farmer | admin
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    crop: Mapped[str] = mapped_column(String(50), index=True)
    inputs: Mapped[dict] = mapped_column(JSON)
    predicted_yield: Mapped[float] = mapped_column(Float)
    lower: Mapped[float] = mapped_column(Float)
    upper: Mapped[float] = mapped_column(Float)
    total_production: Mapped[float] = mapped_column(Float)
    profit: Mapped[float] = mapped_column(Float)
    result: Mapped[dict] = mapped_column(JSON)  # full response payload (SHAP factors, recommendations, economics)
    model_version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    user: Mapped[User] = relationship(back_populates="predictions")


class ApiUsage(Base):
    __tablename__ = "api_usage"
    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String(120), index=True)
    method: Mapped[str] = mapped_column(String(10))
    status: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)


class FarmField(Base):
    """A farmer's field boundary used for satellite crop-health monitoring."""
    __tablename__ = "farm_fields"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    geometry: Mapped[dict] = mapped_column(JSON)          # GeoJSON Polygon, [longitude, latitude]
    centroid_lat: Mapped[float] = mapped_column(Float)
    centroid_lon: Mapped[float] = mapped_column(Float)
    area_ha: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(20), default="polygon")   # polygon | point_radius
    radius_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_sync_status: Mapped[str | None] = mapped_column(String(20), nullable=True)   # ok | no_data
    last_sync_days: Mapped[int | None] = mapped_column(Integer, nullable=True)        # window length of the last successful sync
    user: Mapped[User] = relationship()
    observations: Mapped[list["NdviObservation"]] = relationship(back_populates="field", cascade="all, delete-orphan", order_by="NdviObservation.obs_date")


class NdviObservation(Base):
    """One cloud-screened NDVI value per field and satellite date (cache of Earth Engine results)."""
    __tablename__ = "ndvi_observations"
    __table_args__ = (UniqueConstraint("field_id", "obs_date", name="uq_field_date"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    field_id: Mapped[int] = mapped_column(ForeignKey("farm_fields.id", ondelete="CASCADE"), index=True)
    obs_date: Mapped[date] = mapped_column(Date, index=True)
    ndvi_mean: Mapped[float] = mapped_column(Float)
    valid_fraction: Mapped[float] = mapped_column(Float)
    scenes: Mapped[int] = mapped_column(Integer, default=1)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    field: Mapped[FarmField] = relationship(back_populates="observations")