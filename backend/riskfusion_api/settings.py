from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISKFUSION_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://riskfusion:riskfusion@localhost:5432/riskfusion"
    storage_root: Path = Path("storage")
    data_root: Path = Path("data")
    max_upload_mb: int = 1024
    max_image_mb: int = 8
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    consent_version: str = "consent-v1.0"
    auto_analyse: bool = True  # analyse a recording (all 11 channels + risk) as soon as it is uploaded
    retention_days: int = 180  # raw media retention ceiling; project-end purge also applies (ETH-3)

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
