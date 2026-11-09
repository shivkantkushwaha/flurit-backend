"""Flurit Minimal FastAPI Application Entrypoint.

Establishes the foundation for serving API routes, liveness and readiness health checks,
request correlation tracking, and CORS middleware configuration.
"""

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_v1_router
from app.core.config import get_settings, validate_production_settings
from app.core.logging import get_logger, set_correlation_id, setup_logging
from app.db.session import check_db_readiness, dispose_db_engine

settings = get_settings()
logger = get_logger("flurit.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context for startup validation and graceful shutdown routines."""
    # 1. Initialize structured logging
    setup_logging(
        log_level=settings.LOG_LEVEL,
        json_format=settings.LOG_JSON_FORMAT,
        environment=settings.ENVIRONMENT,
    )

    # 2. Enforce strict production configuration validation on startup
    if settings.is_production:
        validate_production_settings(settings, raise_on_error=True)

    logger.info(
        "Flurit backend initialized",
        extra={
            "app": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
        },
    )

    # 3. Pre-warm ML inference service if artifacts exist
    try:
        from app.ml.inference.service import get_inference_service

        service = get_inference_service()
        logger.info(
            "ML inference service pre-warmed successfully",
            extra={
                "model_version": service._model_version,
                "is_loaded": service.is_loaded,
            },
        )
    except Exception as exc:
        logger.warning(
            "ML inference service deferred loading",
            extra={"reason": str(exc)},
        )

    yield

    # 4. Graceful shutdown routine: drain and dispose pooled database connections
    logger.info("Initiating Flurit backend graceful shutdown")
    try:
        dispose_db_engine()
    except Exception as exc:
        logger.warning("Error disposing database engine on shutdown", extra={"error": str(exc)})
    logger.info("Flurit backend shutdown complete")


def create_application() -> FastAPI:
    """Factory function to build and configure the FastAPI instance."""
    current_settings = get_settings()

    app = FastAPI(
        title=current_settings.PROJECT_NAME,
        version=current_settings.VERSION,
        description="Customer-retention intelligence platform for B2B SaaS (Predict -> Explain -> Prioritize)",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url=f"{current_settings.API_V1_STR}/openapi.json",
        lifespan=lifespan,
    )

    # Correlation ID middleware for end-to-end request tracing
    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next: Any) -> Response:
        req_id = (
            request.headers.get("X-Request-ID")
            or request.headers.get("X-Correlation-ID")
            or f"req_{uuid.uuid4().hex[:16]}"
        )
        set_correlation_id(req_id)
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=current_settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount versioned API routes
    app.include_router(api_v1_router, prefix=current_settings.API_V1_STR)

    # Root metadata endpoint
    @app.get("/", tags=["System"])
    async def root() -> dict[str, Any]:
        """Root metadata endpoint."""
        return {
            "message": "Welcome to Flurit Retention Intelligence API",
            "docs_url": "/docs",
            "version": current_settings.VERSION,
            "status": "online",
        }

    # --------------------------------------------------------------------------
    # Liveness Probes (independent of external dependencies / database)
    # --------------------------------------------------------------------------
    @app.get("/health", tags=["System"])
    @app.get("/healthz", tags=["System"])
    async def liveness_check() -> dict[str, Any]:
        """Liveness probe confirming the HTTP server process is running and responsive."""
        return {
            "status": "ok",
            "app": current_settings.PROJECT_NAME,
            "version": current_settings.VERSION,
            "environment": current_settings.ENVIRONMENT,
        }

    # Versioned API health check (exact backward-compatible schema)
    @app.get(f"{current_settings.API_V1_STR}/health", tags=["System"])
    async def api_v1_health() -> dict[str, Any]:
        """API v1 health probe."""
        return {
            "status": "ok",
            "api_version": "v1",
            "app": current_settings.PROJECT_NAME,
            "environment": current_settings.ENVIRONMENT,
        }

    # --------------------------------------------------------------------------
    # Readiness Probes (verifying database connectivity via lightweight SELECT 1)
    # --------------------------------------------------------------------------
    @app.get("/ready", tags=["System"])
    @app.get("/readyz", tags=["System"])
    async def readiness_check() -> JSONResponse:
        """Readiness probe verifying PostgreSQL connectivity before accepting live traffic.

        Returns HTTP 200 when database is responsive; returns HTTP 503 if unreachable.
        Safeguards credentials and hosts from being exposed in responses or logs.
        """
        is_ready = check_db_readiness()
        if is_ready:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": "ready",
                    "database": "connected",
                    "app": current_settings.PROJECT_NAME,
                    "version": current_settings.VERSION,
                    "environment": current_settings.ENVIRONMENT,
                },
            )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "database": "unavailable",
                "detail": "Database connection probe failed.",
            },
        )

    @app.get(f"{current_settings.API_V1_STR}/ready", tags=["System"])
    async def api_v1_readiness() -> JSONResponse:
        """Versioned API v1 readiness probe."""
        is_ready = check_db_readiness()
        if is_ready:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": "ready",
                    "api_version": "v1",
                    "database": "connected",
                    "app": current_settings.PROJECT_NAME,
                    "environment": current_settings.ENVIRONMENT,
                },
            )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "api_version": "v1",
                "database": "unavailable",
                "detail": "Database connection probe failed.",
            },
        )

    return app


app = create_application()
