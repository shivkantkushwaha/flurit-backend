"""Optional local smoke test against the supplied five-table demo and frozen model."""

import asyncio
import os
import time
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.auth import create_jwt_token
from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.main import create_application


@pytest.mark.parametrize("full_history", [False, True], ids=["latest", "history"])
def test_demo_upload_to_persisted_predictions(full_history):
    directory = os.getenv("FLURIT_E2E_CSV_DIR")
    if not directory:
        pytest.skip("Set FLURIT_E2E_CSV_DIR to the demo CSV directory")

    mapping = {
        "accounts": {"account_id": "account_id", "signup_date": "signup_date", "industry": "industry", "country": "country", "referral_source": "referral_source"},
        "subscriptions": {"subscription_id": "subscription_id", "account_id": "account_id", "start_date": "start_date", "end_date": "end_date", "mrr": "mrr_amount", "seats": "total_seats", "billing_frequency": "billing_interval"},
        "feature_usage": {"usage_id": "usage_id", "subscription_id": "subscription_id", "event_time": "usage_date", "feature_name": "feature_name", "duration_secs": "usage_time_sec", "is_error": "error_count", "is_beta": "is_beta", "usage_count": "usage_count"},
        "support_tickets": {"ticket_id": "ticket_id", "account_id": "account_id", "created_at": "created_at", "priority": "priority", "satisfaction_score": "satisfaction"},
        "churn_events": {"churn_event_id": "churn_id", "account_id": "account_id", "churn_date": "churn_date"},
    }
    pg_url = os.getenv("FLURIT_E2E_POSTGRES_URL")
    if pg_url:
        url = make_url(pg_url)
        assert url.get_backend_name() == "postgresql" and url.host in {"127.0.0.1", "::1", "localhost"}
        engine = create_engine(url, hide_parameters=True)
    else:
        engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def enable_foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(engine)
    settings = Settings(_env_file=None, ENVIRONMENT="testing", DATABASE_URL="sqlite://",
                        SUPABASE_ALLOW_HS256=True, SUPABASE_JWT_SECRET="local-integration-only-secret-value",
                        SUPABASE_JWT_ISSUER=None)
    app = create_application()
    app.dependency_overrides[get_settings] = lambda: settings

    def database():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = database
    files = [("files", (f"{name}.csv", (Path(directory) / f"{name}.csv").read_bytes(), "text/csv")) for name in mapping]

    def auth(subject):
        token = create_jwt_token({"sub": subject, "exp": int(time.time()) + 900}, settings.SUPABASE_JWT_SECRET)
        return {"Authorization": f"Bearer {token}"}

    async def run():
        import json

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test/api/v1/") as client:
            headers = auth("demo-owner")
            assert (await client.get("customers")).status_code == 401
            start = time.perf_counter()
            response = await client.post("onboarding/profile", headers=headers, files=files)
            assert response.status_code == 200, response.text
            print(f"profile: {time.perf_counter() - start:.2f}s", flush=True)
            mapping["dataset_profiles"] = response.json()["datasets"]
            response = await client.post("onboarding/map", headers=headers, json=mapping)
            assert response.status_code == 200, response.text
            assert response.json()["feature_coverage"]["covered_count"] == 45
            form = {"mapping": json.dumps(mapping)}
            response = await client.post("onboarding/preview", headers=headers, files=files, data=form)
            assert response.status_code == 200, response.text
            assert response.json()["existing_model_scoring_ready"]
            if not full_history:
                form["options"] = json.dumps({"observation_dates": [response.json()["potential_observation_dates"][-1]]})
            start = time.perf_counter()
            response = await client.post("onboarding/process", headers=headers, files=files, data=form)
            assert response.status_code == 200, response.text
            result = response.json()
            print(f"process: {time.perf_counter() - start:.2f}s, {result['persisted_summary']}", flush=True)
            assert result["processing_status"] == "SUCCESS", result
            assert result["sentinel_scoring_used"] is False
            assert result["scoring_summary"]["model_version"] == "1.0.0"
            assert result["persisted_summary"]["persisted_predictions"] > 0
            saved = await client.get("onboarding/analysis/latest", headers=headers)
            assert saved.status_code == 200
            assert saved.json()["result_summary"] == result
            assert (await client.get("onboarding/analysis/latest", headers=auth("other-tenant"))).json() is None
            roster = (await client.get("customers", headers=headers)).json()
            assert roster["total"] == 500
            second_page = (await client.get("customers", headers=headers, params={"page": 2})).json()
            assert second_page["total"] == roster["total"]
            assert not ({row["account_id"] for row in roster["items"]} & {row["account_id"] for row in second_page["items"]})
            account = next(item for item in roster["items"] if item["risk_score"] is not None)
            filtered = (await client.get("customers", headers=headers, params={
                "risk_tier": account["risk_tier"], "search": account["account_id"],
            })).json()
            assert [row["account_id"] for row in filtered["items"]] == [account["account_id"]]
            path = f"customers/{account['account_id']}"
            detail = (await client.get(path, headers=headers)).json()
            assert detail["risk_score"] == account["risk_score"]
            assert detail["top_risk_factors"]
            assert (await client.get(f"{path}/history", headers=headers)).json()["items"]
            assert (await client.get(path, headers=auth("other-tenant"))).status_code == 404

    try:
        asyncio.run(run())
    finally:
        engine.dispose()
