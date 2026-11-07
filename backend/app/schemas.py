from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    email: EmailStr
    role: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class FieldInputs(BaseModel):
    """Growing conditions. Numeric fields are range-validated here; which fields are REQUIRED and which categories are
    allowed depends on the active model (synthetic farm-level vs real district-level) and is enforced by the model service."""
    region: str
    season: str
    soil_type: str | None = None
    irrigation: str | None = None
    district: str | None = Field(default=None, max_length=80, description="District (real-data model): selects district soil and location estimates")
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    ph: float | None = Field(default=None, ge=3.5, le=10)
    nitrogen: float | None = Field(default=None, ge=0, le=600, description="Soil available N, kg/ha")
    phosphorus: float | None = Field(default=None, ge=0, le=250, description="Soil available P, kg/ha")
    potassium: float | None = Field(default=None, ge=0, le=800, description="Soil available K, kg/ha")
    rainfall_mm: float | None = Field(default=None, ge=0, le=6000, description="Seasonal rainfall, mm")
    temperature_c: float | None = Field(default=None, ge=-5, le=50, description="Mean season temperature, deg C")
    humidity_pct: float | None = Field(default=None, ge=0, le=100)
    area_ha: float | None = Field(default=None, gt=0, le=5000)
    fertilizer_kg_ha: float | None = Field(default=None, ge=0, le=1000)
    year: int = Field(default_factory=lambda: datetime.now().year, ge=1990, le=2100)


class PredictIn(FieldInputs):
    crop: str


class CompareIn(FieldInputs):
    crops: list[str] | None = Field(default=None, description="Defaults to all crops of the active model")


class Sweep(BaseModel):
    param: Literal["rainfall_mm", "temperature_c", "ph", "fertilizer_kg_ha", "nitrogen", "phosphorus", "potassium", "humidity_pct", "area_ha"]
    min: float
    max: float
    steps: int = Field(default=15, ge=3, le=40)


class WhatIfIn(BaseModel):
    base: PredictIn
    changes: dict[str, float | str | int] = Field(default_factory=dict)
    sweep: Sweep | None = None
    save: bool = False


class Factor(BaseModel):
    feature: str
    label: str
    value: float | str
    shap: float
    direction: Literal["increases", "decreases"]


class Recommendation(BaseModel):
    category: Literal["irrigation", "soil_ph", "nitrogen", "phosphorus", "potassium", "crop", "economics", "climate"]
    severity: Literal["critical", "warning", "info", "success"]
    title: str
    detail: str
    expected_gain_t_ha: float | None = None


class Economics(BaseModel):
    price_per_tonne: float
    revenue: float
    cost: float
    profit: float
    roi_pct: float


class PredictionOut(BaseModel):
    id: int | None = None
    crop: str
    inputs: dict
    predicted_yield_t_ha: float
    lower_t_ha: float
    upper_t_ha: float
    confidence_level: float = 0.8
    total_production_t: float
    economics: Economics
    base_value_t_ha: float
    top_factors: list[Factor]
    recommendations: list[Recommendation]
    model_version: int
    created_at: datetime | None = None
    model_name: str | None = None
    data_mode: str | None = None          # "synthetic" | "real"
    estimated_inputs: list[str] = Field(default_factory=list)   # inputs filled from district-level estimates
    data_notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- satellite crop health
class FieldCreate(BaseModel):
    """Define a field either by a GeoJSON polygon (longitude, latitude order) or by a centre point and a radius."""
    name: str = Field(min_length=1, max_length=120)
    geometry: dict | None = Field(default=None, description='GeoJSON Polygon: {"type": "Polygon", "coordinates": [[[lon, lat], ...]]}')
    latitude: float | None = None
    longitude: float | None = None
    radius_m: float = Field(default=100, description="Used with latitude/longitude: field approximated as a circle of this radius (10-3000 m)")

    @model_validator(mode="after")
    def _one_way_to_define_the_field(self):
        has_point = self.latitude is not None or self.longitude is not None
        if self.geometry is not None and has_point:
            raise ValueError("provide either `geometry` or `latitude` + `longitude`, not both")
        if self.geometry is None and (self.latitude is None or self.longitude is None):
            raise ValueError("provide `geometry` (polygon) or both `latitude` and `longitude`")
        return self


class FieldOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    geometry: dict
    centroid_lat: float
    centroid_lon: float
    area_ha: float
    source: str
    radius_m: float | None = None
    created_at: datetime
    last_synced_at: datetime | None = None
    last_sync_status: str | None = None