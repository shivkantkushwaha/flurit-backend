"""Tests for minimal FastAPI application in app/main.py."""

import unittest

from starlette.testclient import TestClient

from app.main import app


class TestFastAPIApp(unittest.TestCase):
    """Integration tests for Flurit application endpoints."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_root_endpoint(self) -> None:
        """Verify root endpoint metadata response."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("Flurit", data["message"])
        self.assertEqual(data["version"], "0.1.0")

    def test_health_check_endpoint(self) -> None:
        """Verify /health endpoint returns HTTP 200 with ok status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["app"], "Flurit")
        self.assertEqual(data["version"], "0.1.0")

    def test_api_v1_health_check(self) -> None:
        """Verify /api/v1/health endpoint returns versioned health confirmation."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["api_version"], "v1")
        self.assertEqual(data["app"], "Flurit")

    def test_openapi_schema_generated(self) -> None:
        """Verify OpenAPI JSON schema is generated correctly."""
        response = self.client.get("/api/v1/openapi.json")
        self.assertEqual(response.status_code, 200)
        schema = response.json()
        self.assertEqual(schema["info"]["title"], "Flurit")
        self.assertEqual(schema["info"]["version"], "0.1.0")
        self.assertIn("/health", schema["paths"])


if __name__ == "__main__":
    unittest.main()
