"""Flurit API v1 Router Aggregator."""

from fastapi import APIRouter

from app.api.v1.endpoints import onboarding, predictions

api_v1_router = APIRouter()

# Mount predictions and risk endpoints
api_v1_router.include_router(predictions.router)

# Mount dataset onboarding and schema mapping endpoints
api_v1_router.include_router(onboarding.router)
