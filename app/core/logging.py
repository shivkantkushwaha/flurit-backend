"""Structured Logging Configuration for Flurit.

Provides both human-readable standard console formatting and structured JSON logging
for production observability and reproducible debugging.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# Context variable to hold active request/job correlation ID
correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)

SENSITIVE_KEYS = {
    "password",
    "token",
    "secret",
    "authorization",
    "jwt",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "private_key",
}


def get_correlation_id() -> str | None:
    """Retrieve active correlation ID from context."""
    return correlation_id_ctx.get()


def set_correlation_id(correlation_id: str | None) -> None:
    """Set active correlation ID in context."""
    correlation_id_ctx.set(correlation_id)


def _redact_sensitive_data(data: Any) -> Any:
    """Recursively redact sensitive keys from logging dictionaries and structures."""
    if isinstance(data, dict):
        redacted: dict[str, Any] = {}
        for k, v in data.items():
            if any(s in str(k).lower() for s in SENSITIVE_KEYS):
                redacted[str(k)] = "[REDACTED]"
            else:
                redacted[str(k)] = _redact_sensitive_data(v)
        return redacted
    if isinstance(data, list):
        return [_redact_sensitive_data(item) for item in data]
    return data


class JSONFormatter(logging.Formatter):
    """Formats log records as structured JSON lines."""

    def __init__(self, environment: str = "development") -> None:
        super().__init__()
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            "environment": self.environment,
        }

        # Include request/job correlation ID if present in context
        cid = get_correlation_id()
        if cid:
            log_payload["request_id"] = cid

        # Include exception information if present
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        # Include custom extra fields if attached to the record
        standard_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
        }
        extra_fields = {
            k: v
            for k, v in record.__dict__.items()
            if k not in standard_attrs and not k.startswith("_")
        }
        if extra_fields:
            log_payload["extra"] = _redact_sensitive_data(extra_fields)

        return json.dumps(log_payload, default=str)


class StandardFormatter(logging.Formatter):
    """Human-readable console log formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        record_time = datetime.fromtimestamp(record.created, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
        cid = get_correlation_id()
        req_prefix = f" [req:{cid}]" if cid else ""
        base_msg = f"[{record_time}] [{record.levelname:<8}] [{record.name}]{req_prefix} {record.getMessage()}"
        if record.exc_info:
            base_msg += f"\n{self.formatException(record.exc_info)}"
        return base_msg


def setup_logging(
    log_level: str | None = None,
    json_format: bool | None = None,
    environment: str = "development",
) -> logging.Logger:
    """Configure structured logging for the Flurit backend.

    Args:
        log_level: Desired logging level (e.g. "INFO", "DEBUG").
        json_format: Whether to output structured JSON strings.
        environment: Application deployment environment name.

    Returns:
        The configured root logger.
    """
    level_str = (log_level or "INFO").upper()
    numeric_level = getattr(logging, level_str, logging.INFO)

    # Determine formatter
    formatter = JSONFormatter(environment=environment) if json_format else StandardFormatter()

    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # Reset and configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to prevent duplicate lines
    while root_logger.handlers:
        root_logger.handlers.pop()

    root_logger.addHandler(console_handler)

    # Align third-party loggers
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        lib_logger = logging.getLogger(logger_name)
        lib_logger.handlers = []
        lib_logger.propagate = True

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve a named logger instance."""
    return logging.getLogger(name)
