from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from ml.src.config import CROPS, IRRIGATION, REGIONS, SEASONS, SOILS


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
    """Growing conditions. Every numeric field is range-validated to agronomically plausible values."""
    region: str
    season: str
    soil_type: str
    irrigation: str
    ph: float = Field(ge=3.5, le=10)
    nitrogen: float = Field(ge=0, le=600, description="Soil available N, kg/ha")
    phosphorus: float = Field(ge=0, le=250, description="Soil available P, kg/ha")
    potassium: float = Field(ge=0, le=800, description="Soil available K, kg/ha")
    rainfall_mm: float = Field(ge=0, le=6000, description="Seasonal rainfall, mm")
    temperature_c: float = Field(ge=-5, le=50, description="Mean season temperature, deg C")
    humidity_pct: float = Field(ge=0, le=100)
    area_ha: float = Field(gt=0, le=5000)
    fertilizer_kg_ha: float = Field(ge=0, le=1000)
    year: int = Field(default_factory=lambda: datetime.now().year, ge=1990, le=2100)

    @field_validator("region")
    @classmethod
    def _region(cls, v):
        return _in(v, REGIONS, "region")

    @field_validator("season")
    @classmethod
    def _season(cls, v):
        return _in(v, SEASONS, "season")

    @field_validator("soil_type")
    @classmethod
    def _soil(cls, v):
        return _in(v, SOILS, "soil_type")

    @field_validator("irrigation")
    @classmethod
    def _irr(cls, v):
        return _in(v, IRRIGATION, "irrigation")


def _in(v: str, allowed: list[str], name: str) -> str:
    for a in allowed:
        if a.lower() == str(v).lower():
            return a
    raise ValueError(f"unknown {name} '{v}'. Allowed: {', '.join(allowed)}")


class PredictIn(FieldInputs):
    crop: str

    @field_validator("crop")
    @classmethod
    def _crop(cls, v):
        return _in(v, CROPS, "crop")


class CompareIn(FieldInputs):
    crops: list[str] | None = Field(default=None, description="Defaults to all crops")

    @field_validator("crops")
    @classmethod
    def _crops(cls, v):
        return None if v is None else [_in(c, CROPS, "crop") for c in v]


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
