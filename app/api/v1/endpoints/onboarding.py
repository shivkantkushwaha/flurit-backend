import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.queue import QueueBroker, get_queue_broker
from app.core.storage import StorageBackend, get_storage_backend
from app.core.workspace import WorkspaceContext, get_workspace_context
from app.db.session import get_db
from app.models.job import ProcessingJob
from app.onboarding.schemas import (
    DatasetProfileResponse,
    EndToEndAnalysisResponse,
    JobCancelResponse,
    JobListResponse,
    JobStatusResponse,
    JobSubmissionResponse,
    ProcessingOptions,
    ProcessingPreviewResponse,
    ReadinessAssessmentResponse,
    SchemaMappingRequest,
)
from app.onboarding.service import OnboardingService, get_onboarding_service

logger = get_logger("flurit.api.onboarding")

router = APIRouter()


@router.post(
    "/onboarding/profile",
    response_model=DatasetProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and profile customer CSV datasets",
    tags=["Dataset Onboarding"],
)
async def profile_datasets(
    files: list[UploadFile] = File(..., description="One or more CSV datasets to profile"),
    workspace_ctx: WorkspaceContext = Depends(get_workspace_context),
    service: OnboardingService = Depends(get_onboarding_service),
) -> DatasetProfileResponse:
    """Upload one or more CSV files, parse them safely, and generate structural & statistical profiles.

    Inspects column data types, missing value percentages, duplicate rows, candidate
    primary/foreign keys, date ranges, and candidate churn targets. Requires authorized workspace.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one CSV file must be provided for profiling.",
        )

    try:
        response = await service.profile_uploaded_files(files=files)
        logger.info(
            "Completed dataset profiling for upload",
            extra={
                "workspace_id": workspace_ctx.workspace_id,
                "requested_by": workspace_ctx.user_id,
                "file_count": len(files),
                "dataset_names": [d.file_name for d in response.datasets],
            },
        )
        return response
    except ValueError as exc:
        err_msg = str(exc)
        logger.warning(
            "Dataset upload validation error",
            extra={
                "error": err_msg,
                "workspace_id": workspace_ctx.workspace_id,
                "requested_by": workspace_ctx.user_id,
            },
        )
        if "exceeds maximum allowed upload size" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=err_msg,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg,
        ) from exc
    except Exception as exc:
        logger.error(
            "Unexpected error during dataset profiling",
            extra={
                "error": str(exc),
                "workspace_id": workspace_ctx.workspace_id,
                "requested_by": workspace_ctx.user_id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while profiling the uploaded datasets.",
        ) from exc


@router.post(
    "/onboarding/map",
    response_model=ReadinessAssessmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate concept schema mapping and assess dataset readiness",
    tags=["Dataset Onboarding"],
)
async def map_schema_and_assess_readiness(
    request: SchemaMappingRequest,
    workspace_ctx: WorkspaceContext = Depends(get_workspace_context),
    service: OnboardingService = Depends(get_onboarding_service),
) -> ReadinessAssessmentResponse:
    """Validate user-defined column mappings against Flurit's core SaaS entities.

    Evaluates:
      1. Mandatory relational keys, timestamps, and column existence.
      2. Temporal observation span for 60-day churn target construction.
      3. Coverage across all 45 canonical features and compatibility with the frozen model.

    Returns mapping preview, validation issues, recommendations, and overall readiness
    status (READY, PARTIALLY_READY, or UNSUPPORTED). Requires authorized workspace.
    """
    try:
        response = service.assess_readiness(request)
        logger.info(
            "Assessed dataset onboarding readiness",
            extra={
                "workspace_id": workspace_ctx.workspace_id,
                "requested_by": workspace_ctx.user_id,
                "overall_status": response.overall_status.value,
                "target_readiness": response.target_readiness.value,
                "scoring_readiness": response.scoring_readiness.value,
                "feature_coverage_pct": response.feature_coverage.coverage_pct,
            },
        )
        return response
    except Exception as exc:
        logger.error(
            "Unexpected error during schema mapping assessment",
            extra={
                "error": str(exc),
                "workspace_id": workspace_ctx.workspace_id,
                "requested_by": workspace_ctx.user_id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while assessing schema mapping readiness.",
        ) from exc


@router.post(
    "/onboarding/preview",
    response_model=ProcessingPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Preview custom dataset transformation and execution readiness",
    tags=["Dataset Processing & Execution"],
)
async def preview_custom_dataset(
    files: list[UploadFile] = File(..., description="Uploaded CSV dataset files"),
    mapping: str = Form(..., description="JSON-serialized SchemaMappingRequest"),
    options: str | None = Form(
        default=None, description="Optional JSON-serialized ProcessingOptions"
    ),
    workspace_ctx: WorkspaceContext = Depends(get_workspace_context),
    service: OnboardingService = Depends(get_onboarding_service),
) -> ProcessingPreviewResponse:
    """Validate schema mapping and preview execution without running heavy ML scoring.

    Performs referential integrity validation, checks temporal spans, and estimates
    point-in-time snapshot eligibility and readiness flags. Scoped to authorized workspace.
    """
    try:
        mapping_request = SchemaMappingRequest.model_validate_json(mapping)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Malformed schema mapping JSON: {exc}",
        ) from exc

    opts: ProcessingOptions | None = None
    if options:
        try:
            opts = ProcessingOptions.model_validate_json(options)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Malformed processing options JSON: {exc}",
            ) from exc

    try:
        preview_resp = await service.preview_processing(
            files=files,
            mapping=mapping_request,
            options=opts,
            workspace_id=workspace_ctx.workspace_id,
        )
        logger.info(
            "Completed custom dataset processing preview",
            extra={
                "workspace_id": workspace_ctx.workspace_id,
                "requested_by": workspace_ctx.user_id,
                "analysis_ready": preview_resp.analysis_ready,
                "label_ready": preview_resp.label_construction_ready,
                "scoring_ready": preview_resp.existing_model_scoring_ready,
            },
        )
        return preview_resp
    except ValueError as exc:
        err_msg = str(exc)
        logger.warning(
            "Dataset processing preview validation error",
            extra={
                "error": err_msg,
                "workspace_id": workspace_ctx.workspace_id,
                "requested_by": workspace_ctx.user_id,
            },
        )
        if "exceeds maximum allowed upload size" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=err_msg,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg,
        ) from exc
    except Exception as exc:
        logger.error(
            "Unexpected error during dataset processing preview",
            extra={
                "error": str(exc),
                "workspace_id": workspace_ctx.workspace_id,
                "requested_by": workspace_ctx.user_id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while previewing dataset processing.",
        ) from exc


@router.post(
    "/onboarding/process",
    response_model=EndToEndAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute end-to-end custom dataset processing, snapshotting, and ML analysis (Synchronous)",
    tags=["Dataset Processing & Execution"],
)
async def process_custom_dataset(
    response: Response,
    files: list[UploadFile] = File(..., description="Uploaded CSV dataset files"),
    mapping: str = Form(..., description="JSON-serialized SchemaMappingRequest"),
    options: str | None = Form(
        default=None, description="Optional JSON-serialized ProcessingOptions"
    ),
    workspace_ctx: WorkspaceContext = Depends(get_workspace_context),
    db: Session = Depends(get_db),
    service: OnboardingService = Depends(get_onboarding_service),
) -> EndToEndAnalysisResponse:
    """Execute complete end-to-end processing pipeline on uploaded customer datasets.

    Performs:
      1. Relational validation & canonical intermediate transformation.
      2. Point-in-time snapshot generation & leak-free 60-day target labeling.
      3. Canonical 45-feature extraction across mapped domains.
      4. Scoring, SHAP factor attribution, K-Means segmentation, and Value x Risk prioritization
         (for compatible datasets).
    Scoped to authorized tenant workspace.

    Note:
      This synchronous endpoint is retained for backwards compatibility. For production datasets,
      prefer the asynchronous POST /onboarding/process/async endpoint.
    """
    cfg = get_settings()
    if cfg.DEPLOYMENT_TIER == "simplified":
        endpoint_headers = {
            "X-Processing-Mode": "synchronous-primary",
            "X-Deployment-Tier": "simplified",
        }
    else:
        endpoint_headers = {
            "Warning": (
                '299 - "Synchronous dataset processing is deprecated and may time out on large uploads. '
                'Use POST /onboarding/process/async for background processing."'
            ),
            "Deprecation": "true",
        }
    response.headers.update(endpoint_headers)

    try:
        mapping_request = SchemaMappingRequest.model_validate_json(mapping)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Malformed schema mapping JSON: {exc}",
            headers=endpoint_headers,
        ) from exc

    opts: ProcessingOptions | None = None
    if options:
        try:
            opts = ProcessingOptions.model_validate_json(options)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Malformed processing options JSON: {exc}",
                headers=endpoint_headers,
            ) from exc

    try:
        analysis_result = await service.process_dataset(
            files=files,
            mapping=mapping_request,
            options=opts,
            workspace_id=workspace_ctx.workspace_id,
            session=db,
            user_id=workspace_ctx.user_id,
        )
        logger.info(
            "Completed end-to-end custom dataset processing",
            extra={
                "workspace_id": workspace_ctx.workspace_id,
                "requested_by": workspace_ctx.user_id,
                "status": analysis_result.processing_status,
                "analysis_ready": analysis_result.analysis_ready,
                "label_ready": analysis_result.label_construction_ready,
                "scoring_ready": analysis_result.existing_model_scoring_ready,
            },
        )
        return analysis_result
    except ValueError as exc:
        err_msg = str(exc)
        logger.warning(
            "Dataset processing validation error",
            extra={
                "error": err_msg,
                "workspace_id": workspace_ctx.workspace_id,
                "requested_by": workspace_ctx.user_id,
            },
        )
        if "exceeds maximum allowed upload size" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=err_msg,
                headers=endpoint_headers,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg,
            headers=endpoint_headers,
        ) from exc
    except DBAPIError as exc:
        sqlstate = getattr(exc.orig, "pgcode", None)
        logger.error("Could not save dataset results (SQLSTATE %s)", sqlstate,
                     extra={"workspace_id": workspace_ctx.workspace_id})
        if sqlstate in {"22001", "42P01", "42703"}:
            detail = "The database schema could not store the analysis. Ask your administrator to check the database migrations."
        else:
            detail = "The database could not save the analysis. Please try again once it is available."
        raise HTTPException(status_code=503, detail=detail, headers=endpoint_headers) from exc
    except Exception as exc:
        logger.error(
            "Unexpected error during end-to-end dataset processing (%s)",
            type(exc).__name__,
            extra={
                "workspace_id": workspace_ctx.workspace_id,
                "requested_by": workspace_ctx.user_id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during end-to-end dataset processing.",
            headers=endpoint_headers,
        ) from exc


@router.get("/onboarding/analysis/latest", response_model=JobStatusResponse | None)
def get_latest_analysis(
    workspace_ctx: WorkspaceContext = Depends(get_workspace_context),
    db: Session = Depends(get_db),
) -> JobStatusResponse | None:
    job = db.scalars(
        select(ProcessingJob)
        .where(
            ProcessingJob.workspace_id == workspace_ctx.workspace_id,
            ProcessingJob.status == "succeeded",
            ProcessingJob.result_summary.is_not(None),
        )
        .order_by(ProcessingJob.completed_at.desc().nulls_last(), ProcessingJob.created_at.desc())
        .limit(1)
    ).first()
    if job is None:
        return None
    return JobStatusResponse(
        job_id=job.job_id, workspace_id=job.workspace_id, dataset_name=job.dataset_name,
        status=job.status, progress_pct=job.progress_pct, created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        result_summary=job.result_summary,
    )


@router.post(
    "/onboarding/process/async",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue asynchronous custom dataset processing and ML analysis",
    tags=["Dataset Processing & Execution"],
)
@router.post(
    "/onboarding/jobs",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue asynchronous custom dataset processing job (alias)",
    tags=["Dataset Processing & Execution"],
)
async def submit_processing_job_async(
    response: Response,
    files: list[UploadFile] = File(..., description="Uploaded CSV dataset files"),
    mapping: str = Form(..., description="JSON-serialized SchemaMappingRequest"),
    options: str | None = Form(
        default=None, description="Optional JSON-serialized ProcessingOptions"
    ),
    dataset_name: str | None = Form(
        default=None, description="Optional dataset identifier or friendly name"
    ),
    idempotency_key: str | None = Header(
        default=None,
        alias="X-Idempotency-Key",
        description="Optional client-provided idempotency token",
    ),
    idempotency_key_form: str | None = Form(
        default=None,
        alias="idempotency_key",
        description="Optional form field idempotency token",
    ),
    workspace_ctx: WorkspaceContext = Depends(get_workspace_context),
    db: Session = Depends(get_db),
    storage: StorageBackend = Depends(get_storage_backend),
    queue: QueueBroker = Depends(get_queue_broker),
) -> JobSubmissionResponse:
    """Submit CSV datasets and schema mapping for background asynchronous processing.

    Accepts datasets, persists them safely in partitioned temporary storage, registers
    a tracking record in `processing_jobs`, and dispatches execution to the background queue.
    Returns HTTP 202 Accepted with a unique job ID immediately. Scoped to authorized workspace.
    """
    settings = get_settings()
    if settings.QUEUE_MODE == "disabled":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Asynchronous background processing is disabled in this simplified deployment. "
                "Please use the synchronous processing endpoint POST /api/v1/onboarding/process, "
                "or configure QUEUE_MODE='arq' backed by Redis for background queue execution."
            ),
        )

    if settings.QUEUE_MODE == "memory":
        response.headers["X-Queue-Durability"] = "ephemeral-in-process"
        response.headers["Warning"] = (
            '299 - "In-process queue active. Jobs are ephemeral and will not survive process restarts."'
        )

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one CSV file must be provided for processing.",
        )

    try:
        mapping_request = SchemaMappingRequest.model_validate_json(mapping)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Malformed schema mapping JSON: {exc}",
        ) from exc

    opts: ProcessingOptions | None = None
    if options:
        try:
            opts = ProcessingOptions.model_validate_json(options)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Malformed processing options JSON: {exc}",
            ) from exc

    effective_key = idempotency_key or idempotency_key_form
    if effective_key:
        existing_job = db.execute(
            select(ProcessingJob).where(
                ProcessingJob.workspace_id == workspace_ctx.workspace_id,
                ProcessingJob.idempotency_key == effective_key,
            )
        ).scalar_one_or_none()
        if existing_job:
            logger.info(
                "Found existing processing job for idempotency key",
                extra={
                    "job_id": existing_job.job_id,
                    "workspace_id": workspace_ctx.workspace_id,
                    "idempotency_key": effective_key,
                },
            )
            return JobSubmissionResponse(
                job_id=existing_job.job_id,
                status=existing_job.status,
                workspace_id=existing_job.workspace_id,
                created_at=existing_job.created_at.isoformat(),
                message="Existing job returned for provided idempotency key.",
            )

    settings = get_settings()
    file_payloads: list[tuple[str, bytes]] = []
    for file in files:
        filename = Path(file.filename or "data.csv").name
        content = await file.read()
        if len(content) > settings.ONBOARDING_MAX_FILE_SIZE_BYTES:
            max_mb = settings.ONBOARDING_MAX_FILE_SIZE_BYTES / (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File '{filename}' exceeds maximum allowed upload size ({max_mb:.0f} MB).",
            )
        file_payloads.append((filename, content))

    job_id = f"job_{uuid.uuid4().hex[:16]}"
    file_keys: list[str] = []
    try:
        for filename, content in file_payloads:
            fkey = await storage.save_file(
                workspace_id=workspace_ctx.workspace_id,
                job_id=job_id,
                filename=filename,
                content=content,
            )
            file_keys.append(fkey)
    except Exception as exc:
        logger.error(
            "Failed saving uploaded dataset files to storage",
            extra={
                "job_id": job_id,
                "workspace_id": workspace_ctx.workspace_id,
                "error": str(exc),
            },
        )
        await storage.delete_job_files(workspace_id=workspace_ctx.workspace_id, job_id=job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stage uploaded files for asynchronous processing.",
        ) from exc

    now_dt = datetime.now(UTC)
    job = ProcessingJob(
        job_id=job_id,
        workspace_id=workspace_ctx.workspace_id,
        user_id=workspace_ctx.user_id,
        dataset_name=dataset_name or (file_payloads[0][0] if file_payloads else None),
        status="queued",
        progress_pct=0,
        stage_message="Job queued for background processing",
        file_keys=file_keys,
        mapping_payload=mapping_request.model_dump(mode="json"),
        options_payload=opts.model_dump(mode="json") if opts else None,
        idempotency_key=effective_key,
        created_at=now_dt,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        await queue.enqueue_processing_job(
            job_id=job.job_id,
            workspace_id=workspace_ctx.workspace_id,
        )
    except Exception as exc:
        logger.exception(
            "Failed dispatching job to background queue broker",
            extra={"job_id": job.job_id, "workspace_id": workspace_ctx.workspace_id},
        )
        job.status = "failed"
        job.stage_message = "Failed to dispatch job to queue broker"
        job.error_message = "Background queue broker dispatch failed."
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Background queue broker is temporarily unavailable.",
        ) from exc

    logger.info(
        "Submitted asynchronous dataset processing job",
        extra={
            "job_id": job.job_id,
            "workspace_id": workspace_ctx.workspace_id,
            "user_id": workspace_ctx.user_id,
        },
    )
    submission_msg = (
        "Dataset processing job queued for in-process execution (ephemeral queue mode; non-durable across process restarts)."
        if settings.QUEUE_MODE == "memory"
        else "Dataset processing job queued successfully."
    )
    return JobSubmissionResponse(
        job_id=job.job_id,
        status=job.status,
        workspace_id=job.workspace_id,
        created_at=job.created_at.isoformat(),
        message=submission_msg,
    )


@router.get(
    "/onboarding/jobs/{job_id}",
    response_model=JobStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get processing job status, progress, and results",
    tags=["Dataset Processing & Execution"],
)
async def get_job_status(
    job_id: str,
    workspace_ctx: WorkspaceContext = Depends(get_workspace_context),
    db: Session = Depends(get_db),
) -> JobStatusResponse:
    """Retrieve execution status, percentage progress, stage notes, and results for a job.

    Enforces strict workspace tenant isolation: foreign workspace job lookups return HTTP 403.
    """
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Processing job '{job_id}' not found.",
        )

    if job.workspace_id != workspace_ctx.workspace_id:
        logger.warning(
            "Unauthorized cross-workspace job access attempt",
            extra={
                "job_id": job_id,
                "requested_by_workspace": workspace_ctx.workspace_id,
                "target_workspace": job.workspace_id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You do not have permission to access jobs in another workspace.",
        )

    return JobStatusResponse(
        job_id=job.job_id,
        workspace_id=job.workspace_id,
        dataset_name=job.dataset_name,
        status=job.status,
        progress_pct=job.progress_pct,
        stage_message=job.stage_message,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error_message=job.error_message,
        result_summary=job.result_summary,
    )


@router.get(
    "/onboarding/jobs",
    response_model=JobListResponse,
    status_code=status.HTTP_200_OK,
    summary="List dataset processing jobs for authorized workspace",
    tags=["Dataset Processing & Execution"],
)
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100, description="Max jobs to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    workspace_ctx: WorkspaceContext = Depends(get_workspace_context),
    db: Session = Depends(get_db),
) -> JobListResponse:
    """List recent dataset processing jobs scoped strictly to caller's authorized workspace."""
    total = (
        db.execute(
            select(func.count())
            .select_from(ProcessingJob)
            .where(ProcessingJob.workspace_id == workspace_ctx.workspace_id)
        ).scalar()
        or 0
    )

    jobs = (
        db.execute(
            select(ProcessingJob)
            .where(ProcessingJob.workspace_id == workspace_ctx.workspace_id)
            .order_by(ProcessingJob.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        .scalars()
        .all()
    )

    items = [
        JobStatusResponse(
            job_id=j.job_id,
            workspace_id=j.workspace_id,
            dataset_name=j.dataset_name,
            status=j.status,
            progress_pct=j.progress_pct,
            stage_message=j.stage_message,
            created_at=j.created_at.isoformat(),
            started_at=j.started_at.isoformat() if j.started_at else None,
            completed_at=j.completed_at.isoformat() if j.completed_at else None,
            error_message=j.error_message,
            result_summary=j.result_summary,
        )
        for j in jobs
    ]

    return JobListResponse(total=total, jobs=items)


@router.post(
    "/onboarding/jobs/{job_id}/cancel",
    response_model=JobCancelResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel a queued or running processing job",
    tags=["Dataset Processing & Execution"],
)
async def cancel_job(
    job_id: str,
    workspace_ctx: WorkspaceContext = Depends(get_workspace_context),
    db: Session = Depends(get_db),
    storage: StorageBackend = Depends(get_storage_backend),
) -> JobCancelResponse:
    """Cancel an active processing job and release staged files. Scoped to authorized workspace."""
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Processing job '{job_id}' not found.",
        )

    if job.workspace_id != workspace_ctx.workspace_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You do not have permission to modify jobs in another workspace.",
        )

    if job.status in ("succeeded", "failed", "cancelled"):
        return JobCancelResponse(
            job_id=job.job_id,
            status=job.status,
            message=f"Job is already completed with status: {job.status}.",
        )

    job.status = "cancelled"
    job.stage_message = "Cancelled by user request"
    job.completed_at = datetime.now(UTC)
    db.commit()

    await storage.delete_job_files(workspace_id=workspace_ctx.workspace_id, job_id=job_id)

    logger.info(
        "Cancelled processing job",
        extra={"job_id": job.job_id, "workspace_id": workspace_ctx.workspace_id},
    )
    return JobCancelResponse(
        job_id=job.job_id,
        status="cancelled",
        message="Processing job was successfully cancelled.",
    )
