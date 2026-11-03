from functools import lru_cache
from pathlib import Path

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

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()