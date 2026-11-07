from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Crop Yield AI"
    database_url: str = "sqlite:///./cropai.db"  # PostgreSQL in production: postgresql+psycopg2://user:pass@db:5432/cropai
    secret_key: str = "change-me-in-production-please-use-a-long-random-string"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 12
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    model_dir: str = str(ROOT / "ml" / "artifacts")
    model_version: str = "latest"  # "latest" | N (synthetic model_vN) | "real_latest" | "real_district_vN"
    reports_dir: str = str(ROOT / "ml" / "reports")
    rate_limit_enabled: bool = True
    rate_limit_default: str = "120/minute"
    rate_limit_predict: str = "30/minute"
    rate_limit_auth: str = "10/minute"
    admin_email: str = "admin@cropai.app"
    admin_password: str = "Admin@12345"
    weather_cache_seconds: int = 1800
    log_level: str = "INFO"

    # ---- satellite crop health (Google Earth Engine). Credentials are server-side only and never sent to the browser.
    gee_project: str = ""                          # Google Cloud project registered for Earth Engine (required)
    gee_service_account_email: str = ""            # optional when the JSON key contains it
    gee_service_account_key_file: str = ""         # path to the service-account JSON key file ...
    gee_service_account_json: SecretStr = SecretStr("")  # ... or the JSON key contents themselves (preferred for hosting secrets)
    ndvi_low: float = 0.2                          # approximate class boundaries for colouring (see docs/crop-health.md)
    ndvi_high: float = 0.5
    satellite_default_days: int = 90
    satellite_max_days: int = 730
    satellite_scene_max_cloud_pct: float = 80.0    # pre-filter whole scenes; per-pixel masking does the real work
    satellite_min_valid_fraction: float = 0.5      # a scene must have cloud-free pixels over this share of the field
    satellite_cache_hours: float = 12.0
    field_min_ha: float = 0.05
    field_max_ha: float = 1000.0
    max_fields_per_user: int = 20
    rate_limit_satellite: str = "20/minute"

    @property
    def gee_configured(self) -> bool:
        return bool(self.gee_project.strip())

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()