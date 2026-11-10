# F8 Completion Report — Backend Persistence and Customer Intelligence

## Delivered

- The synchronous `POST /api/v1/onboarding/process` path now receives the authorized SQLAlchemy session and invokes one atomic persistence service after transformation and inference.
- Canonical account, subscription, usage, ticket, and churn source records are persisted tenant-scoped, along with customer snapshots and serialized canonical feature vectors.
- Successful batch outputs are linked to persisted snapshot IDs and saved idempotently as model runs, predictions, explanations, and segments. The prediction write receives the real workspace ID.
- The processing response now includes feature coverage, explicit sentinel-scoring disclosure, and persisted-record counts/model-run ID.
- Processing returns `INFERENCE_FAILED` when transformation succeeded but the model cannot complete; it does not fabricate a score. Datasets that created usable snapshots but are incompatible with the frozen model remain `PARTIAL_SUCCESS`; runs without usable labels are `NOT_READY`.
- New tenant-scoped retrieval APIs:
  - `GET /api/v1/customers`
  - `GET /api/v1/customers/{account_id}`
  - `GET /api/v1/customers/{account_id}/history`
- The frontend customer roster and detail view now use those persisted APIs. Empty and failure views remain explicit; no demo customer rows are produced.

## Data-integrity notes

- Source data is transformed before persistence and the transaction is committed only after all requested artifacts are written. A persistence failure is rolled back by the onboarding service.
- Snapshot, feature, and prediction uniqueness follows the existing database keys. Reprocessing updates matching records instead of duplicating prediction/explanation/segment records.
- Account display names and plan tiers are not part of the current onboarding mapping contract. The backend stores the account ID as the required database name fallback and exposes it as `null` to the UI; plan tier is stored as `unknown`, not invented as a commercial tier.
- The model score is always labeled as an uncalibrated risk ranking, not a churn probability.

## Validation performed

- `python -m compileall -q app` — passed.
- `ruff check` on changed backend modules — passed.
- `pytest -q tests/test_persistence.py --maxfail=1` — 7 passed.
- `pytest -q tests/test_processing.py --maxfail=1` advanced through five tests before the execution host stopped the process without a session handle.
- `pytest -q tests/test_auth_es256.py::TestSupabaseES256Authentication::test_expired_token_rejected` — passed.
- `npm run typecheck` in `C:\flurit-frontend` — passed.

## Remaining environment limitation

This Windows host uses Python 3.14. Starlette `TestClient` currently blocks before dispatching any request because AnyIO cannot create its background event-loop self-pipe (`socket._fallback_socketpair` blocks in `accept`). This is reproducible for `/health`, before application routing, authentication, database, or model code runs. It prevents the TestClient-based backend suites from completing in this host environment. It is a test-runtime compatibility issue, not a reason to weaken authentication or bypass tenant checks. Run those suites with a supported Python/AnyIO Windows combination (for example a project virtual environment based on Python 3.12/3.13) to obtain the full suite result.
