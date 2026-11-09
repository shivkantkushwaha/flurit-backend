"""Tests for typed application configuration in app/core/config.py."""

import unittest

from pydantic import ValidationError

from app.core.config import Settings, clear_settings_cache, get_settings


class TestSettings(unittest.TestCase):
    """Unit tests for Flurit application Settings."""

    def setUp(self) -> None:
        clear_settings_cache()

    def tearDown(self) -> None:
        clear_settings_cache()

    def test_default_settings(self) -> None:
        """Verify that default settings instantiate with expected values."""
        settings = Settings()
        self.assertEqual(settings.PROJECT_NAME, "Flurit")
        self.assertEqual(settings.VERSION, "0.1.0")
        self.assertEqual(settings.ENVIRONMENT, "development")
        self.assertEqual(settings.LOG_LEVEL, "INFO")
        self.assertFalse(settings.LOG_JSON_FORMAT)
        self.assertEqual(settings.PREDICTION_WINDOW_DAYS, 60)
        self.assertFalse(settings.is_production)
        self.assertFalse(settings.is_testing)
        self.assertIn("http://localhost:3000", settings.CORS_ORIGINS)
        self.assertIn("flurit_db", settings.DATABASE_URL)

    def test_environment_properties(self) -> None:
        """Verify is_production and is_testing properties."""
        prod_settings = Settings(ENVIRONMENT="production")
        self.assertTrue(prod_settings.is_production)
        self.assertFalse(prod_settings.is_testing)

        test_settings = Settings(ENVIRONMENT="testing")
        self.assertFalse(test_settings.is_production)
        self.assertTrue(test_settings.is_testing)

    def test_cors_origins_parsing(self) -> None:
        """Verify CORS origins string and list parsing."""
        # JSON string list
        s1 = Settings(CORS_ORIGINS='["http://example.com", "http://app.flurit.io"]')  # type: ignore[arg-type]
        self.assertEqual(len(s1.CORS_ORIGINS), 2)
        self.assertIn("http://example.com", s1.CORS_ORIGINS)
        self.assertIn("http://app.flurit.io", s1.CORS_ORIGINS)

        # Comma separated
        s2 = Settings(CORS_ORIGINS="http://a.com, http://b.com")  # type: ignore[arg-type]
        self.assertEqual(s2.CORS_ORIGINS, ["http://a.com", "http://b.com"])

        # Native list
        s3 = Settings(CORS_ORIGINS=["http://native.com"])
        self.assertEqual(s3.CORS_ORIGINS, ["http://native.com"])

    def test_bynalt_infrastructure_isolation_enforced(self) -> None:
        """Enforce that Flurit database configuration strictly rejects Bynalt infrastructure."""
        with self.assertRaises(ValueError) as ctx:
            Settings(DATABASE_URL="postgresql://user:pass@bynalt-prod.internal:5432/bynalt_db")
        self.assertIn("Bynalt", str(ctx.exception))

    def test_invalid_environment_rejected(self) -> None:
        """Ensure invalid environment names are rejected by type validation."""
        with self.assertRaises(ValidationError):
            Settings(ENVIRONMENT="invalid_env")  # type: ignore[arg-type]

    def test_singleton_get_settings_caching(self) -> None:
        """Ensure get_settings returns cached singleton and clear_settings_cache resets it."""
        s1 = get_settings()
        s2 = get_settings()
        self.assertIs(s1, s2)

        clear_settings_cache()
        s3 = get_settings()
        self.assertIsNot(s1, s3)


if __name__ == "__main__":
    unittest.main()
