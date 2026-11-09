"""FastAPI Endpoints for Model Inference, Explainability, and Portfolio Prioritization."""

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.core.workspace import WorkspaceContext, get_workspace_context
from app.db.session import get_db
from app.ml.inference.schemas import (
    CustomerRiskRequest,
    CustomerRiskResponse,
    PrioritizationQuadrantsRequest,
    PrioritizationQuadrantsResponse,
    ScorePredictionRequest,
    ScorePredictionResponse,
)
from app.ml.inference.service import ModelInferenceService, get_inference_service
from app.models.account import Account

logger = get_logger("flurit.api.predictions")

router = APIRouter()


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
