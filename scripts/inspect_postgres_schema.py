"""Inspect and verify all tables, columns, constraints, and indexes in flurit_db."""

import sqlalchemy as sa

from app.core.config import get_settings


def inspect_schema():
    settings = get_settings()
    engine = sa.create_engine(settings.DATABASE_URL)

    with engine.connect() as conn:
        print("=" * 70)
        print("DATABASE CONNECTION IDENTITY:")
        row = conn.execute(
            sa.text("SELECT current_database(), current_user, session_user;")
        ).fetchone()
        print(f"  Database:     {row[0]}")
        print(f"  Current User: {row[1]}")
        print(f"  Session User: {row[2]}")
        assert row[0] == "flurit_db", f"Unexpected database: {row[0]}"
        assert row[1] == "flurit_app", f"Unexpected user: {row[1]}"
        print("=" * 70)

        print("\nPUBLIC TABLES & ROW COUNTS:")
        tables_res = conn.execute(
            sa.text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;"
            )
        ).fetchall()
        tables = [r[0] for r in tables_res]
        for t in tables:
            cnt = conn.execute(sa.text(f"SELECT count(*) FROM {t};")).scalar()
            print(f"  - {t:<26}: {cnt} rows")

        print("\nTABLE CONSTRAINTS (PK, FK, UNIQUE):")
        constraints_res = conn.execute(
            sa.text("""
            SELECT
                r.relname AS table_name,
                c.conname AS constraint_name,
                c.contype AS constraint_type,
                pg_get_constraintdef(c.oid) AS definition
            FROM pg_constraint c
            JOIN pg_class r ON c.conrelid = r.oid
            JOIN pg_namespace n ON r.relnamespace = n.oid
            WHERE n.nspname = 'public'
            ORDER BY r.relname, c.contype DESC, c.conname;
        """)
        ).fetchall()

        type_labels = {"p": "PRIMARY KEY", "f": "FOREIGN KEY", "u": "UNIQUE", "c": "CHECK"}
        for c in constraints_res:
            label = type_labels.get(c[2], c[2])
            print(f"  [{c[0]:<20}] {label:<12} {c[1]:<32} -> {c[3]}")

        print("\nINDEXES:")
        indexes_res = conn.execute(
            sa.text("""
            SELECT tablename, indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
            ORDER BY tablename, indexname;
        """)
        ).fetchall()
        for idx in indexes_res:
            print(f"  [{idx[0]:<20}] {idx[1]:<35}")

        print("=" * 70)
        print("SCHEMA INSPECTION COMPLETED SUCCESSFULLY")
        print("=" * 70)


if __name__ == "__main__":
    inspect_schema()
