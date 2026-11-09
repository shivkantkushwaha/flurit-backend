"""Database Engine and Session Management for Flurit.

CRITICAL ARCHITECTURAL ISOLATION:
Flurit uses its own dedicated database instance. Connection to or reference of
Bynalt infrastructure is strictly prohibited.
"""

from collections.abc import Generator

import sqlalchemy as sa
from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("flurit.db")

_shared_engine: Engine | None = None


def validate_database_url_isolation(url: str) -> None:
    """Ensure database connection string never touches Bynalt infrastructure."""
    if "bynalt" in url.lower():
        raise ValueError(
            "CRITICAL SECURITY VIOLATION: Flurit must remain completely isolated from Bynalt infrastructure. "
            "Database URL cannot reference Bynalt."
        )


def create_db_engine(database_url: str | None = None) -> Engine:
    """Create SQLAlchemy engine with isolation safeguards and bounded pooling."""
    settings = get_settings()
    target_url = database_url or settings.DATABASE_URL
    validate_database_url_isolation(target_url)

    # SQLite configuration for testing vs PostgreSQL for application
    if target_url.startswith("sqlite"):
        return create_engine(
            target_url,
            connect_args={"check_same_thread": False},
        )
    return create_engine(
        target_url,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT_SECONDS,
        pool_recycle=settings.DB_POOL_RECYCLE_SECONDS,
        pool_pre_ping=True,
    )


def get_shared_engine(database_url: str | None = None) -> Engine:
    """Retrieve or initialize the shared database engine singleton."""
    global _shared_engine
    if _shared_engine is None or database_url is not None:
        engine = create_db_engine(database_url)
        if database_url is None:
            _shared_engine = engine
        return engine
    return _shared_engine


def dispose_db_engine() -> None:
    """Safely dispose all pooled connections on application shutdown."""
    global _shared_engine
    if _shared_engine is not None:
        try:
            _shared_engine.dispose()
        except Exception as exc:
            logger.warning("Error disposing database engine", extra={"error": str(exc)})
        _shared_engine = None


def check_db_readiness(engine: Engine | None = None) -> bool:
    """Perform lightweight 'SELECT 1' database readiness check.

    Returns True if database is reachable and responsive, False otherwise.
    Safeguards credentials, hostnames, and passwords from leaking in logs or exceptions.
    """
    eng = engine or get_shared_engine()
    try:
        with eng.connect() as conn:
            val = conn.execute(sa.text("SELECT 1;")).scalar()
            return val == 1
    except Exception as exc:
        logger.warning(
            "Database readiness probe failed",
            extra={"error_type": exc.__class__.__name__},
        )
        return False


def get_db(engine: Engine | None = Depends(lambda: None)) -> Generator[Session, None, None]:
    """FastAPI database session dependency."""
    active_engine = engine or get_shared_engine()
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=active_engine)
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
