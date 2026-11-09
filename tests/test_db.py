"""Tests for database engine creation and Bynalt isolation validation."""

import unittest

from sqlalchemy.orm import Session

from app.db.session import (
    create_db_engine,
    get_db,
    validate_database_url_isolation,
)


class TestDatabaseSession(unittest.TestCase):
    """Unit tests for database session initialization and isolation safeguards."""

    def test_bynalt_infrastructure_isolation_rejected(self) -> None:
        """Verify that any database URL containing bynalt is immediately rejected."""
        with self.assertRaises(ValueError) as ctx:
            validate_database_url_isolation("postgresql://user:pass@bynalt-prod.internal:5432/db")
        self.assertIn("Bynalt", str(ctx.exception))

    def test_sqlite_engine_creation(self) -> None:
        """Verify in-memory SQLite engine creates cleanly."""
        engine = create_db_engine("sqlite:///:memory:")
        self.assertIsNotNone(engine)

    def test_get_db_session_yields_session(self) -> None:
        """Verify get_db generator yields active session and closes upon exit."""
        test_engine = create_db_engine("sqlite:///:memory:")
        db_generator = get_db(engine=test_engine)
        session = next(db_generator)
        self.assertIsInstance(session, Session)
        # Complete generator to close session
        try:
            next(db_generator)
        except StopIteration:
            pass


if __name__ == "__main__":
    unittest.main()
