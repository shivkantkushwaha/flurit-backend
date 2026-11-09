"""Test Configuration and Fixtures for Flurit."""

import os

import pytest

# Ensure legacy test suites using mock HS256 tokens continue to function
os.environ.setdefault("SUPABASE_ALLOW_HS256", "true")
os.environ.setdefault("SUPABASE_JWT_SECRET", "flurit-secure-test-jwt-secret-key-32-chars-min!")

from app.core.auth import clear_jwks_client_cache
from app.core.config import Settings, clear_settings_cache


@pytest.fixture(autouse=True)
def reset_settings_env():
    """Ensure settings cache and JWKS client cache are cleared before and after each test."""
    clear_settings_cache()
    clear_jwks_client_cache()
    yield
    clear_settings_cache()
    clear_jwks_client_cache()


@pytest.fixture
def test_settings() -> Settings:
    """Provide isolated test settings instance."""
    return Settings(
        ENVIRONMENT="testing",
        PROJECT_NAME="Flurit-Test",
        LOG_LEVEL="DEBUG",
        DATABASE_URL="postgresql+psycopg2://test_user:test_pass@localhost:5432/test_db",
    )
