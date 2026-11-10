"""Store uploaded country names as well as country codes."""

import sqlalchemy as sa
from alembic import op

revision = "0004_account_country_names"
down_revision = "0003_processing_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("accounts", "country", existing_type=sa.String(8), type_=sa.Text())


def downgrade() -> None:
    # PostgreSQL must reject the downgrade if any stored country exceeds eight characters.
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM accounts WHERE char_length(country) > 8) THEN
                RAISE EXCEPTION 'Cannot downgrade: accounts contain country names longer than eight characters';
            END IF;
        END
        $$;
    """)
    op.alter_column("accounts", "country", existing_type=sa.Text(), type_=sa.String(8))
