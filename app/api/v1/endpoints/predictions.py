"""FastAPI Endpoints for Model Inference, Explainability, and Portfolio Prioritization."""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.core.workspace import WorkspaceContext, get_workspace_context
from app.db.session import get_db
from app.ml.inference.schemas import (
    CustomerRiskRequest,
    CustomerRiskResponse,
    PersistedCustomerDetailResponse,
    PersistedCustomerHistoryItem,
    PersistedCustomerHistoryResponse,
    PersistedCustomerListResponse,
    PersistedCustomerSummary,
    PrioritizationQuadrantsRequest,
    PrioritizationQuadrantsResponse,
    ScorePredictionRequest,
    ScorePredictionResponse,
)
from app.ml.inference.service import (
    SCORE_INTERPRETATION_NOTICE,
    ModelInferenceService,
    get_inference_service,
)
from app.models.account import Account
from app.models.ml import (
    CustomerExplanation,
    CustomerFeature,
    CustomerPrediction,
    CustomerSegment,
    ModelRun,
)
from app.models.subscription import Subscription

logger = get_logger("flurit.api.predictions")

router = APIRouter()


def _iso(value: object | None) -> str | None:
    return value.isoformat() if isinstance(value, (date,)) else None


def _latest_prediction(session: Session, workspace_id: str, account_id: str) -> CustomerPrediction | None:
    return session.scalars(
        select(CustomerPrediction)
        .where(
            CustomerPrediction.workspace_id == workspace_id,
            CustomerPrediction.account_id == account_id,
        )
        .order_by(desc(CustomerPrediction.predicted_at), desc(CustomerPrediction.prediction_id))
    ).first()


def _latest_feature(session: Session, workspace_id: str, account_id: str) -> CustomerFeature | None:
    return session.scalars(
        select(CustomerFeature)
        .where(CustomerFeature.workspace_id == workspace_id, CustomerFeature.account_id == account_id)
        .order_by(desc(CustomerFeature.snapshot_date), desc(CustomerFeature.feature_id))
    ).first()


@router.get(
    "/customers",
    response_model=PersistedCustomerListResponse,
    summary="List persisted customer intelligence for the authorized workspace",
    tags=["Customer Intelligence"],
)
async def list_customers(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    search: str | None = Query(default=None, max_length=255),
    risk_tier: str | None = Query(default=None, max_length=32),
    segment: str | None = Query(default=None, max_length=100),
    workspace_ctx: WorkspaceContext = Depends(get_workspace_context),
    db: Session = Depends(get_db),
) -> PersistedCustomerListResponse:
    """Return only persisted tenant customers; no synthetic roster rows are ever generated."""
    statement = select(Account).where(Account.workspace_id == workspace_ctx.workspace_id)
    if search and search.strip():
        term = f"%{search.strip()}%"
        statement = statement.where(
            (Account.account_id.ilike(term)) | (Account.account_name.ilike(term))
        )
    accounts = list(db.scalars(statement.order_by(Account.account_id)).all())
    items: list[PersistedCustomerSummary] = []
    for account in accounts:
        prediction = _latest_prediction(db, workspace_ctx.workspace_id, account.account_id)
        if risk_tier and (prediction is None or prediction.risk_level != risk_tier):
            continue
        segment_row = (
            db.scalars(
                select(CustomerSegment)
                .where(
                    CustomerSegment.workspace_id == workspace_ctx.workspace_id,
                    CustomerSegment.account_id == account.account_id,
                    CustomerSegment.model_run_id == prediction.model_run_id if prediction else False,
                )
                .order_by(desc(CustomerSegment.created_at))
            ).first()
            if prediction
            else None
        )
        if segment and (segment_row is None or segment_row.cluster_name != segment):
            continue
        feature = _latest_feature(db, workspace_ctx.workspace_id, account.account_id)
        items.append(PersistedCustomerSummary(
            account_id=account.account_id,
            account_name=None if account.account_name == account.account_id else account.account_name,
            industry=account.industry,
            country=account.country,
            current_mrr=feature.current_mrr if feature else None,
            risk_score=prediction.churn_probability if prediction else None,
            risk_tier=prediction.risk_level if prediction else None,
            segment_name=segment_row.cluster_name if segment_row else None,
            assessed_at=_iso(prediction.predicted_at) if prediction else None,
            snapshot_date=_iso(prediction.snapshot.snapshot_date) if prediction and prediction.snapshot else None,
            score_interpretation=SCORE_INTERPRETATION_NOTICE if prediction else None,
        ))
    total = len(items)
    start = (page - 1) * page_size
    return PersistedCustomerListResponse(items=items[start:start + page_size], total=total, page=page, page_size=page_size)


@router.get(
    "/customers/{account_id}",
    response_model=PersistedCustomerDetailResponse,
    summary="Retrieve a persisted customer and latest ML assessment",
    tags=["Customer Intelligence"],
)
async def get_customer(
    account_id: str = Path(..., min_length=1, max_length=64),
    workspace_ctx: WorkspaceContext = Depends(get_workspace_context),
    db: Session = Depends(get_db),
) -> PersistedCustomerDetailResponse:
    account = db.get(Account, {"workspace_id": workspace_ctx.workspace_id, "account_id": account_id})
    if account is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Customer was not found in this workspace.")
    prediction = _latest_prediction(db, workspace_ctx.workspace_id, account_id)
    feature = _latest_feature(db, workspace_ctx.workspace_id, account_id)
    explanation = (
        db.scalars(select(CustomerExplanation).where(CustomerExplanation.prediction_id == prediction.prediction_id)).first()
        if prediction else None
    )
    segment_row = (
        db.scalars(select(CustomerSegment).where(
            CustomerSegment.workspace_id == workspace_ctx.workspace_id,
            CustomerSegment.account_id == account_id,
            CustomerSegment.model_run_id == prediction.model_run_id,
        ).order_by(desc(CustomerSegment.created_at))).first()
        if prediction else None
    )
    subscriptions = list(db.scalars(select(Subscription).where(
        Subscription.workspace_id == workspace_ctx.workspace_id,
        Subscription.account_id == account_id,
    ).order_by(desc(Subscription.start_date))).all())
    return PersistedCustomerDetailResponse(
        account_id=account.account_id,
        account_name=None if account.account_name == account.account_id else account.account_name,
        industry=account.industry,
        country=account.country,
        signup_date=account.signup_date.isoformat(),
        plan_tier=None if account.plan_tier == "unknown" else account.plan_tier,
        current_mrr=feature.current_mrr if feature else None,
        risk_score=prediction.churn_probability if prediction else None,
        risk_tier=prediction.risk_level if prediction else None,
        segment_name=segment_row.cluster_name if segment_row else None,
        snapshot_date=_iso(prediction.snapshot.snapshot_date) if prediction and prediction.snapshot else None,
        assessed_at=_iso(prediction.predicted_at) if prediction else None,
        score_interpretation=SCORE_INTERPRETATION_NOTICE if prediction else None,
        explanation_summary=explanation.summary_text if explanation else None,
        top_risk_factors=explanation.top_risk_factors if explanation else [],
        subscriptions=[{
            "subscription_id": sub.subscription_id, "start_date": sub.start_date.isoformat(),
            "end_date": _iso(sub.end_date), "mrr_amount": sub.mrr_amount,
            "arr_amount": sub.arr_amount, "billing_frequency": sub.billing_frequency,
        } for sub in subscriptions],
    )


@router.get(
    "/customers/{account_id}/history",
    response_model=PersistedCustomerHistoryResponse,
    summary="Retrieve persisted model-run history for one customer",
    tags=["Customer Intelligence"],
)
async def get_customer_history(
    account_id: str = Path(..., min_length=1, max_length=64),
    workspace_ctx: WorkspaceContext = Depends(get_workspace_context),
    db: Session = Depends(get_db),
) -> PersistedCustomerHistoryResponse:
    account = db.get(Account, {"workspace_id": workspace_ctx.workspace_id, "account_id": account_id})
    if account is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Customer was not found in this workspace.")
    predictions = list(db.scalars(select(CustomerPrediction).where(
        CustomerPrediction.workspace_id == workspace_ctx.workspace_id,
        CustomerPrediction.account_id == account_id,
    ).order_by(desc(CustomerPrediction.predicted_at), desc(CustomerPrediction.prediction_id))).all())
    items: list[PersistedCustomerHistoryItem] = []
    for prediction in predictions:
        model = db.get(ModelRun, prediction.model_run_id)
        segment_row = db.scalars(select(CustomerSegment).where(
            CustomerSegment.workspace_id == workspace_ctx.workspace_id,
            CustomerSegment.account_id == account_id,
            CustomerSegment.model_run_id == prediction.model_run_id,
        ).order_by(desc(CustomerSegment.created_at))).first()
        items.append(PersistedCustomerHistoryItem(
            prediction_id=prediction.prediction_id,
            snapshot_date=_iso(prediction.snapshot.snapshot_date) if prediction.snapshot else None,
            assessed_at=_iso(prediction.predicted_at) or "",
            model_name=model.model_name if model else "unavailable",
            model_version=model.version if model else "unavailable",
            risk_score=prediction.churn_probability,
            risk_tier=prediction.risk_level,
            segment_name=segment_row.cluster_name if segment_row else None,
            score_interpretation=SCORE_INTERPRETATION_NOTICE,
        ))
    return PersistedCustomerHistoryResponse(account_id=account_id, items=items)


@router.post(
    "/predictions/score",
    response_model=ScorePredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Score canonical feature vector for customer churn risk",
    tags=["Predictions & Inference"],
)
async def score_prediction(
    request: ScorePredictionRequest,
    workspace_ctx: WorkspaceContext = Depends(get_workspace_context),
    service: ModelInferenceService = Depends(get_inference_service),
) -> ScorePredictionResponse:
    """Score a point-in-time canonical customer feature payload.

    Computes uncalibrated model churn score and assigns risk tier using the
    frozen Logistic Regression champion pipeline. Scoped to authorized tenant workspace.
    """
    try:
        response = service.score(request)
        logger.info(
            "Computed churn score prediction",
            extra={
                "account_id": request.account_id or "anonymous",
                "model_score": response.model_score,
                "risk_tier": response.risk_tier,
                "workspace_id": workspace_ctx.workspace_id,
                "requested_by": workspace_ctx.user_id,
            },
        )
        return response
    except ValueError as exc:
        logger.warning("Feature validation error during scoring", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Feature validation error: {exc}",
        ) from exc
    except RuntimeError as exc:
        logger.error("Inference service runtime failure", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model inference service is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during scoring", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction scoring.",
        ) from exc


@router.post(
    "/customers/{account_id}/risk",
    response_model=CustomerRiskResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict customer churn score, SHAP explanation factors, and segment assignment",
    tags=["Predictions & Inference"],
)
async def customer_risk(
    request: CustomerRiskRequest,
    account_id: str = Path(..., description="Customer unique account identifier"),
    workspace_ctx: WorkspaceContext = Depends(get_workspace_context),
    db: Session = Depends(get_db),
    service: ModelInferenceService = Depends(get_inference_service),
) -> CustomerRiskResponse:
    """Generate comprehensive customer risk assessment scoped to authorized workspace.

    Includes uncalibrated churn score, local additive SHAP factor contributions (raising
    vs mitigating risk), and unsupervised K-Means behavioral segment assignment.
    Rejects cross-workspace account queries with HTTP 403 Forbidden.
    """
    account_id_clean = account_id.strip()
    if not account_id_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account ID must not be empty.",
        )

    # Enforce workspace scoping: check if this account exists in another workspace
    other_workspace_account = db.scalars(
        select(Account.account_id).where(
            Account.account_id == account_id_clean,
            Account.workspace_id != workspace_ctx.workspace_id,
        )
    ).first()
    my_workspace_account = db.scalars(
        select(Account.account_id).where(
            Account.account_id == account_id_clean,
            Account.workspace_id == workspace_ctx.workspace_id,
        )
    ).first()

    if other_workspace_account is not None and my_workspace_account is None:
        logger.warning(
            "Cross-workspace account risk access blocked",
            extra={
                "account_id": account_id_clean,
                "caller_workspace_id": workspace_ctx.workspace_id,
                "requested_by": workspace_ctx.user_id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: Account '{account_id_clean}' does not belong to authorized workspace.",
        )

    try:
        response = service.explain_and_segment(account_id=account_id_clean, request=request)
        logger.info(
            "Computed customer risk assessment",
            extra={
                "account_id": account_id_clean,
                "model_score": response.model_score,
                "risk_tier": response.risk_tier,
                "segment_id": response.segment_id,
                "workspace_id": workspace_ctx.workspace_id,
                "requested_by": workspace_ctx.user_id,
            },
        )
        return response
    except ValueError as exc:
        logger.warning("Validation error in customer risk endpoint", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {exc}",
        ) from exc
    except RuntimeError as exc:
        logger.error("Inference service unavailable", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model inference service is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error in customer risk endpoint", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating customer risk assessment.",
        ) from exc


@router.post(
    "/prioritization/quadrants",
    response_model=PrioritizationQuadrantsResponse,
    status_code=status.HTTP_200_OK,
    summary="Group scored customer accounts into Value x Risk priority quadrants",
    tags=["Predictions & Inference"],
)
async def prioritize_quadrants(
    request: PrioritizationQuadrantsRequest,
    workspace_ctx: WorkspaceContext = Depends(get_workspace_context),
    service: ModelInferenceService = Depends(get_inference_service),
) -> PrioritizationQuadrantsResponse:
    """Prioritize a portfolio of scored customer accounts into Value x Risk quadrants.

    Calculates Estimated Revenue at Risk (MRR * score) and aggregates accounts into 5
    actionable priority tiers. Note: Revenue at risk is an operational prioritization
    estimate, NOT a guaranteed financial loss or calibrated probability.
    """
    try:
        response = service.prioritize_quadrants(request)
        logger.info(
            "Evaluated portfolio prioritization quadrants",
            extra={
                "record_count": response.total_accounts,
                "total_mrr": response.total_mrr,
                "total_at_risk": response.total_estimated_revenue_at_risk_mrr,
                "workspace_id": workspace_ctx.workspace_id,
                "requested_by": workspace_ctx.user_id,
            },
        )
        return response
    except ValueError as exc:
        logger.warning("Validation error in prioritization", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prioritization validation error: {exc}",
        ) from exc
    except RuntimeError as exc:
        logger.error("Inference service unavailable for prioritization", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model inference service is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during portfolio prioritization", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while computing portfolio prioritization.",
        ) from exc
