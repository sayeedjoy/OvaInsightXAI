"""Prediction and health endpoints."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.model import predictor
from app.schemas.input_schema import HealthResponse, PredictionRequest, PredictionResponse
from app.services.preprocessing import request_to_features
from app.utils.config import FeatureOrderError

# Add backend root to path to import test_case_generator
backend_root = Path(__file__).resolve().parent.parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from test_case_generator import generate_negative_cases, generate_positive_cases

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(payload: PredictionRequest) -> PredictionResponse:
    """Validate input, run inference, and return the model output."""
    try:
        features = request_to_features(payload)
    except FeatureOrderError as exc:
        logger.warning("Invalid feature payload: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    try:
        result = predictor.predict(features)
    except FileNotFoundError as exc:
        logger.error("Model file missing: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Model not loaded: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    except ValueError as exc:
        logger.error("Model inference failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return PredictionResponse(prediction=result.prediction, confidence=result.confidence)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Lightweight readiness probe."""
    try:
        predictor.ensure_model_loaded()
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model file missing",
        )

    return HealthResponse(status="ok")


@router.get("/test-case/negative")
async def get_negative_test_case() -> dict:
    """
    Generate a single negative (normal/healthy) test case.
    
    Returns a JSON object with all 12 biomarker features in clinically
    realistic normal ranges. This can be used to auto-fill the prediction form
    for testing normal cases.
    """
    try:
        test_case = generate_negative_cases(n=1, random_state=None)
        return test_case
    except Exception as exc:
        logger.error("Failed to generate negative test case: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate test case: {str(exc)}",
        )


@router.get("/test-case/positive")
async def get_positive_test_case() -> dict:
    """
    Generate a single positive (ovarian cancer) test case.
    
    Returns a JSON object with all 12 biomarker features showing patterns
    characteristic of ovarian cancer (elevated CA125, HE4, etc.). This can be
    used to auto-fill the prediction form for testing cancer cases.
    """
    try:
        test_case = generate_positive_cases(n=1, random_state=None)
        return test_case
    except Exception as exc:
        logger.error("Failed to generate positive test case: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate test case: {str(exc)}",
        )


