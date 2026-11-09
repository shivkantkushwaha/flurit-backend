"""Typed Application Configuration for Flurit.

Provides environment variable parsing, type validation, and application settings
using Pydantic / Pydantic Settings.

"""

import json
import os
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class _BaseConfig(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )

except ImportError:
    # Graceful fallback when pydantic-settings is not installed in the environment
    from pydantic import BaseModel

    class _BaseConfig(BaseModel):  # type: ignore[no-redef]
        def __init__(self, **values: Any):
            # Load environment variables into values if not explicitly provided
            env_values: dict[str, Any] = {}
            # Optional: check if .env file exists and parse simple key-values
            if os.path.exists(".env"):
                try:
                    with open(".env", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                env_values[k.strip().lower()] = v.strip().strip("\"'")
                except Exception:
                    pass
            for k, v in os.environ.items():
                env_values[k.lower()] = v

            for field_name in self.__class__.model_fields:
                if field_name not in values and field_name.lower() in env_values:
                    values[field_name] = env_values[field_name.lower()]

            super().__init__(**values)


class Settings(_BaseConfig):
    """Core application settings with strict validation."""

    # Application identity
    PROJECT_NAME: str = "Flurit"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: Literal["development", "staging", "production", "testing"] = "development"
    DEPLOYMENT_TIER: Literal["enterprise", "simplified"] = "enterprise"

    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_JSON_FORMAT: bool = False

    # Database: Dedicated Flurit PostgreSQL instance
    DATABASE_URL: str = Field(
        default="postgresql+psycopg2://flurit_user:flurit_password@localhost:5432/flurit_db",
        description="PostgreSQL connection string for Flurit application database.",
    )
    DB_POOL_SIZE: int = Field(
        default=10,
        ge=1,
        le=100,
        description="SQLAlchemy connection pool base size.",
    )
    DB_MAX_OVERFLOW: int = Field(
        default=5,
        ge=0,
        le=50,
        description="Maximum temporary connections above pool size.",
    )
    DB_POOL_TIMEOUT_SECONDS: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Seconds to wait before timing out on acquiring a pool connection.",
    )
    DB_POOL_RECYCLE_SECONDS: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description="Seconds after which active connections are recycled to prevent stale drops.",
    )

    # Supabase Authentication
    SUPABASE_URL: str = Field(
        default="https://jhankemazkagrybaoitu.supabase.co",
        description="Public URL for Supabase project instance.",
    )
    SUPABASE_JWKS_URL: str | None = Field(
        default=None,
        description="Optional override for Supabase JWKS endpoint. Defaults to {SUPABASE_URL}/auth/v1/.well-known/jwks.json.",
    )
    SUPABASE_JWT_ALGORITHM: str = Field(
        default="ES256",
        description="Active JWT signature algorithm (e.g. ES256, HS256).",
    )
    SUPABASE_ALLOW_HS256: bool = Field(
        default=False,
        description="Whether to permit legacy HS256 tokens. Defaults strictly to False.",
    )
    SUPABASE_JWT_SECRET: str | None = Field(
        default=None,
        description="Secret key used to verify legacy HS256 JWT signatures. Optional when using ES256.",
    )
    SUPABASE_JWT_AUDIENCE: str | None = Field(
        default="authenticated",
        description="Expected JWT audience claim (aud). Set to None to disable audience verification.",
    )
    SUPABASE_JWT_ISSUER: str | None = Field(
        default="https://jhankemazkagrybaoitu.supabase.co/auth/v1",
        description="Expected JWT issuer claim (iss). Strict validation enforced.",
    )
    SUPABASE_STRICT_SUB_UUID: bool = Field(
        default=False,
        description="Whether to strictly enforce UUID format for token 'sub' even in non-production. Always enforced in production.",
    )

    # CORS origins for browser client access
    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # ML Pipeline defaults
    PREDICTION_WINDOW_DAYS: int = Field(
        default=60,
        description="Candidate prediction horizon in days for future churn labeling.",
    )

    # Dataset Onboarding & Ingestion limits
    ONBOARDING_MAX_FILE_SIZE_BYTES: int = Field(
        default=50 * 1024 * 1024,  # 50 MB
        description="Maximum allowed uploaded CSV file size in bytes.",
    )
    ONBOARDING_MAX_ROW_COUNT: int = Field(
        default=500_000,
        description="Maximum allowed row count for uploaded CSV files.",
    )
    ONBOARDING_MAX_COLUMNS: int = Field(
        default=200,
        description="Maximum allowed column count for uploaded CSV files.",
    )
    ONBOARDING_MIN_OBSERVATION_DAYS: int = Field(
        default=60,
        description="Minimum date span required between earliest and latest dates for 60d churn labeling.",
    )
    ONBOARDING_RECOMMENDED_HISTORY_DAYS: int = Field(
        default=150,
        description="Recommended historical span in days (90d lookback + 60d prediction horizon).",
    )

    # Redis & Queue Configuration
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for ARQ queue broker.",
    )
    QUEUE_MODE: Literal["arq", "memory", "sync", "disabled"] = Field(
        default="memory",
        description="Queue broker mode: arq (Redis), memory (async in-process for dev/testing), sync, or disabled.",
    )
    QUEUE_JOB_TIMEOUT_SECONDS: int = Field(
        default=600,
        description="Maximum execution timeout for asynchronous processing jobs in seconds.",
    )
    QUEUE_MAX_RETRIES: int = Field(
        default=3,
        description="Maximum retry attempts on transient worker failures.",
    )

    # Storage Configuration
    STORAGE_BACKEND: Literal["local", "memory", "s3"] = Field(
        default="local",
        description="Storage backend type for uploaded files awaiting processing.",
    )
    STORAGE_LOCAL_DIR: str = Field(
        default="data/uploads",
        description="Local directory for temporary file storage.",
    )
    STORAGE_CLEANUP_AFTER_PROCESSING: bool = Field(
        default=True,
        description="Whether to automatically delete temporary upload files once job finishes.",
    )

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_isolation(cls, v: str) -> str:
        """Enforce strict isolation from Bynalt infrastructure."""
        if "bynalt" in v.lower():
            raise ValueError(
                "CRITICAL SECURITY VIOLATION: Flurit must remain completely isolated from Bynalt infrastructure. "
                "Database URL cannot reference Bynalt."
            )
        return v

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        """Parse CORS origins from a list, JSON string, or comma-separated string."""
        if isinstance(v, list):
            return [str(origin).strip() for origin in v]
        if isinstance(v, str):
            v_stripped = v.strip()
            if v_stripped.startswith("[") and v_stripped.endswith("]"):
                try:
                    parsed = json.loads(v_stripped)
                    if isinstance(parsed, list):
                        return [str(origin).strip() for origin in parsed]
                except json.JSONDecodeError:
                    pass
            # Comma-separated fallback
            return [origin.strip() for origin in v_stripped.split(",") if origin.strip()]
        return ["http://localhost:3000", "http://127.0.0.1:3000"]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.ENVIRONMENT == "testing"


def validate_production_settings(settings: Settings, raise_on_error: bool = True) -> list[str]:
    """Validate all required production settings and return configuration errors.

    Ensures that when running in production, mandatory secrets are configured,
    insecure defaults are rejected, and required infrastructure (ARQ, Redis,
    PostgreSQL, dedicated storage) is enforced.

    Args:
        settings: Application Settings instance.
        raise_on_error: Whether to raise ValueError if errors are found.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    # 1. Supabase JWT Authentication
    alg = settings.SUPABASE_JWT_ALGORITHM.upper()
    if alg == "ES256":
        if not settings.SUPABASE_URL or not settings.SUPABASE_URL.strip():
            errors.append("SUPABASE_URL is mandatory in production but is not set.")
        elif not settings.SUPABASE_URL.startswith("https://"):
            errors.append("SUPABASE_URL must start with https:// in production.")
        elif "localhost" in settings.SUPABASE_URL.lower():
            errors.append(
                "SUPABASE_URL points to localhost; provide hosted Supabase URL for production."
            )

        if not settings.SUPABASE_JWT_ISSUER or not settings.SUPABASE_JWT_ISSUER.strip():
            errors.append(
                "SUPABASE_JWT_ISSUER is mandatory for ES256 in production but is not set."
            )
        elif not settings.SUPABASE_JWT_ISSUER.startswith("https://"):
            errors.append("SUPABASE_JWT_ISSUER must start with https:// in production.")

        if settings.SUPABASE_ALLOW_HS256:
            if not settings.SUPABASE_JWT_SECRET:
                errors.append(
                    "SUPABASE_JWT_SECRET is mandatory when SUPABASE_ALLOW_HS256 is enabled in production."
                )
            elif len(settings.SUPABASE_JWT_SECRET.strip()) < 32:
                errors.append("SUPABASE_JWT_SECRET must be at least 32 characters in production.")
            elif "test-jwt-secret" in settings.SUPABASE_JWT_SECRET.lower():
                errors.append(
                    "SUPABASE_JWT_SECRET contains test secret string; replace with production secret."
                )
    elif alg == "HS256":
        if not settings.SUPABASE_JWT_SECRET:
            errors.append("SUPABASE_JWT_SECRET is mandatory in production but is not set.")
        elif len(settings.SUPABASE_JWT_SECRET.strip()) < 32:
            errors.append("SUPABASE_JWT_SECRET must be at least 32 characters in production.")
        elif "test-jwt-secret" in settings.SUPABASE_JWT_SECRET.lower():
            errors.append(
                "SUPABASE_JWT_SECRET contains test secret string; replace with production secret."
            )
    else:
        errors.append(
            f"SUPABASE_JWT_ALGORITHM '{settings.SUPABASE_JWT_ALGORITHM}' is unsupported in production."
        )

    # 2. Database URL
    db_url_low = settings.DATABASE_URL.lower()
    if "sqlite" in db_url_low:
        errors.append("SQLite cannot be used as application database in production.")
    if "bynalt" in db_url_low:
        errors.append("CRITICAL: DATABASE_URL references Bynalt infrastructure.")
    if "flurit_password" in db_url_low or ":password@" in db_url_low:
        errors.append(
            "DATABASE_URL contains default/insecure password; configure strong credentials."
        )
    if "@localhost" in db_url_low or "@127.0.0.1" in db_url_low:
        errors.append(
            "DATABASE_URL points to localhost; provide hosted PostgreSQL URL for production."
        )

    # 3. Queue & Background Worker
    if settings.DEPLOYMENT_TIER == "enterprise":
        if settings.QUEUE_MODE != "arq":
            errors.append(
                f"QUEUE_MODE is set to '{settings.QUEUE_MODE}'. Production requires QUEUE_MODE='arq' backed by Redis."
            )
        redis_url_low = settings.REDIS_URL.lower()
        if not (redis_url_low.startswith("redis://") or redis_url_low.startswith("rediss://")):
            errors.append("REDIS_URL must begin with redis:// or rediss:// in production.")
        elif (
            "@localhost" in redis_url_low
            or "@127.0.0.1" in redis_url_low
            or "://localhost" in redis_url_low
        ):
            errors.append("REDIS_URL points to localhost; provide hosted Redis URL for production.")

        # 4. Storage Backend (Enterprise multi-dyno requires non-memory)
        if settings.STORAGE_BACKEND == "memory":
            errors.append("STORAGE_BACKEND cannot be 'memory' in production.")
    else:
        # Simplified deployment tier: single API process + PostgreSQL
        if settings.QUEUE_MODE == "arq":
            redis_url_low = settings.REDIS_URL.lower()
            if not (redis_url_low.startswith("redis://") or redis_url_low.startswith("rediss://")):
                errors.append(
                    "REDIS_URL must begin with redis:// or rediss:// when QUEUE_MODE='arq'."
                )
            elif (
                "@localhost" in redis_url_low
                or "@127.0.0.1" in redis_url_low
                or "://localhost" in redis_url_low
            ):
                errors.append(
                    "REDIS_URL points to localhost; provide hosted Redis URL for production."
                )

        # Storage Backend in simplified tier (local filesystem is valid for single dyno)
        if settings.STORAGE_BACKEND == "memory":
            errors.append("STORAGE_BACKEND cannot be 'memory' in production.")

    # 5. CORS Origins
    if "*" in settings.CORS_ORIGINS:
        errors.append("CORS_ORIGINS cannot contain wildcard '*' when allow_credentials=True.")
    default_dev_origins = {"http://localhost:3000", "http://127.0.0.1:3000"}
    if set(settings.CORS_ORIGINS) == default_dev_origins:
        errors.append(
            "CORS_ORIGINS only contains localhost dev origins; configure production domain(s)."
        )

    if errors and raise_on_error:
        error_list = "\n  - ".join(errors)
        raise ValueError(
            f"CRITICAL PRODUCTION CONFIGURATION ERRORS DETECTED:\n  - {error_list}\n"
            "Backend startup aborted to prevent insecure deployment."
        )

    return errors


@lru_cache
def get_settings() -> Settings:
    """Retrieve cached application settings instance (singleton)."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear cached settings instance (useful in automated tests)."""
    get_settings.cache_clear()
