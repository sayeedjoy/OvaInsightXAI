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

logger = logging.getLogger(__name__)

# Add backend root to path to import test_case_generator
backend_root = Path(__file__).resolve().parent.parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

# Try to import test_case_generator, but make it optional
try:
    from test_case_generator import generate_negative_cases, generate_positive_cases
    TEST_CASE_GENERATOR_AVAILABLE = True
except ImportError:
    logger.warning("test_case_generator not found. Test case endpoints will be disabled.")
    TEST_CASE_GENERATOR_AVAILABLE = False
    # Create dummy functions to avoid errors
    def generate_negative_cases(*args, **kwargs):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Test case generator not available"
        )
    def generate_positive_cases(*args, **kwargs):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Test case generator not available"
        )

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(payload: PredictionRequest) -> PredictionResponse:
    """Validate input, run inference, and return the model output."""
    try:
        # Convert request to features
        try:
            features = request_to_features(payload)
        except FeatureOrderError as exc:
            logger.warning("Invalid feature payload: %s", exc, exc_info=True)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        except Exception as exc:
            logger.error("Unexpected error during feature preprocessing: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Feature preprocessing failed: {str(exc)}"
            )

        # Run prediction
        try:
            result = predictor.predict(features)
        except FileNotFoundError as exc:
            logger.error("Model file missing: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model file not found: {str(exc)}"
            )
        except RuntimeError as exc:
            logger.error("Model not loaded: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model not loaded: {str(exc)}"
            )
        except ValueError as exc:
            logger.error("Model inference failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model inference error: {str(exc)}"
            )
        except Exception as exc:
            logger.error("Unexpected error during prediction: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(exc)}"
            )

        # Return response
        try:
            return PredictionResponse(prediction=result.prediction, confidence=result.confidence)
        except Exception as exc:
            logger.error("Error creating response: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create response: {str(exc)}"
            )
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as exc:
        # Catch-all for any other unexpected exceptions
        logger.error("Unexpected error in predict endpoint: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(exc)}"
        )


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
        logger.error("Failed to generate negative test case: %s", exc, exc_info=True)
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
        logger.error("Failed to generate positive test case: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate test case: {str(exc)}",
        )


