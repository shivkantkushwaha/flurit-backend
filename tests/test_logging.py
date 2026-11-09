"""Tests for structured logging in app/core/logging.py."""

import json
import logging
import unittest

from app.core.logging import (
    JSONFormatter,
    StandardFormatter,
    get_logger,
    setup_logging,
)


class TestLogging(unittest.TestCase):
    """Unit tests for Flurit structured logging."""

    def test_json_formatter_valid_output(self) -> None:
        """Verify that JSONFormatter produces valid JSON with required metadata keys."""
        formatter = JSONFormatter(environment="testing")
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="Customer risk computed: %s",
            args=("A001",),
            exc_info=None,
        )
        formatted_str = formatter.format(record)
        data = json.loads(formatted_str)

        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["logger"], "test_logger")
        self.assertEqual(data["message"], "Customer risk computed: A001")
        self.assertEqual(data["environment"], "testing")
        self.assertEqual(data["line"], 42)
        self.assertIn("timestamp", data)

    def test_json_formatter_with_extra_fields(self) -> None:
        """Verify that extra kwargs attached to log records appear in the JSON output."""
        formatter = JSONFormatter(environment="testing")
        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname=__file__,
            lineno=10,
            msg="Elevated churn probability detected",
            args=(),
            exc_info=None,
        )
        record.__dict__["account_id"] = "A1042"
        record.__dict__["churn_prob"] = 0.85

        formatted_str = formatter.format(record)
        data = json.loads(formatted_str)

        self.assertIn("extra", data)
        self.assertEqual(data["extra"]["account_id"], "A1042")
        self.assertEqual(data["extra"]["churn_prob"], 0.85)

    def test_standard_formatter_output(self) -> None:
        """Verify that StandardFormatter formats clean human-readable log lines."""
        formatter = StandardFormatter()
        record = logging.LogRecord(
            name="flurit.core",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Application started",
            args=(),
            exc_info=None,
        )
        formatted_str = formatter.format(record)
        self.assertIn("[INFO    ]", formatted_str)
        self.assertIn("[flurit.core]", formatted_str)
        self.assertIn("Application started", formatted_str)

    def test_setup_logging_configures_root_logger(self) -> None:
        """Verify that setup_logging adjusts root logger level and handlers."""
        root = setup_logging(log_level="DEBUG", json_format=True, environment="testing")
        self.assertEqual(root.level, logging.DEBUG)
        self.assertTrue(len(root.handlers) > 0)
        self.assertIsInstance(root.handlers[0].formatter, JSONFormatter)

    def test_get_logger_retrieval(self) -> None:
        """Verify get_logger returns a named Logger instance."""
        logger = get_logger("flurit.test")
        self.assertEqual(logger.name, "flurit.test")


if __name__ == "__main__":
    unittest.main()
