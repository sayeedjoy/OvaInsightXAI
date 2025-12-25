"""Prediction and health endpoints."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.model import predictor
from app.model.registry import MODEL_REGISTRY
from app.schemas.input_schema import (
    HealthResponse,
    HepatitisBRequest,
    PcosRequest,
    PredictionRequest,
    PredictionResponse,
)
from app.services.preprocessing import request_to_features
from app.services.xai import compute_all_xai_explanations
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


def _predict_with_model(payload, model_key: str, feature_order: list[str], include_xai: bool = True) -> PredictionResponse:
    """Shared prediction handler across models."""
    try:
        features = request_to_features(payload, feature_order)
    except FeatureOrderError as exc:
        logger.warning("Invalid feature payload for %s: %s", model_key, exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error during feature preprocessing for %s: %s", model_key, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature preprocessing failed: {str(exc)}"
        )

    try:
        predictor.ensure_model_loaded(model_key)
    except FileNotFoundError as exc:
        logger.error("Model file missing for %s: %s", model_key, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model file not found: {str(exc)}"
        )
    except Exception as exc:
        logger.error("Error loading model %s: %s", model_key, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to load model: {str(exc)}"
        )

    try:
        result = predictor.predict(features, model_key=model_key)
    except FileNotFoundError as exc:
        logger.error("Model file missing during prediction for %s: %s", model_key, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model file not found: {str(exc)}"
        )
    except RuntimeError as exc:
        logger.error("Model not loaded for %s: %s", model_key, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model not loaded: {str(exc)}"
        )
    except ValueError as exc:
        logger.error("Model inference failed for %s: %s", model_key, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference error: {str(exc)}"
        )
    except Exception as exc:
        logger.error("Unexpected error during prediction for %s: %s", model_key, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(exc)}"
        )

    # Compute XAI explanations if requested
    xai_explanations = None
    if include_xai:
        try:
            logger.info("Computing XAI explanations for %s", model_key)
            xai_explanations = compute_all_xai_explanations(model_key, features)
            # Convert to dict for JSON serialization
            xai_explanations = {
                "shap": xai_explanations["shap"],
                "lime": xai_explanations["lime"],
                "pdp_1d": xai_explanations["pdp_1d"],
                "ice_1d": xai_explanations["ice_1d"],
                "ale_1d": xai_explanations["ale_1d"],
            }
        except Exception as exc:
            logger.warning("Failed to compute XAI explanations for %s: %s", model_key, exc, exc_info=True)
            # Don't fail the prediction if XAI fails, just log it
            xai_explanations = {"error": f"XAI computation failed: {str(exc)}"}

    try:
        return PredictionResponse(
            prediction=result.prediction,
            confidence=result.confidence,
            xai=xai_explanations
        )
    except Exception as exc:
        logger.error("Error creating response for %s: %s", model_key, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create response: {str(exc)}"
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    payload: PredictionRequest,
    include_xai: bool = True
) -> PredictionResponse:
    """Validate input, run inference, and return the ovarian model output."""
    config = MODEL_REGISTRY["ovarian"]
    try:
        data = payload.model_dump()
        return _predict_with_model(data, "ovarian", config.feature_order, include_xai=include_xai)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected error in predict endpoint: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(exc)}"
        )


@router.post("/predict/ovarian", response_model=PredictionResponse)
async def predict_ovarian(
    payload: PredictionRequest,
    include_xai: bool = True
) -> PredictionResponse:
    """Prediction endpoint for ovarian model (alias of /predict)."""
    config = MODEL_REGISTRY["ovarian"]
    return _predict_with_model(payload.model_dump(), "ovarian", config.feature_order, include_xai=include_xai)


@router.post("/predict/hepatitis_b", response_model=PredictionResponse)
async def predict_hepatitis_b(
    payload: HepatitisBRequest,
    include_xai: bool = True
) -> PredictionResponse:
    """Prediction endpoint for hepatitis B model."""
    config = MODEL_REGISTRY["hepatitis_b"]
    # Use by_alias=True so payload keys match the CSV headers
    return _predict_with_model(payload.model_dump(by_alias=True), "hepatitis_b", config.feature_order, include_xai=include_xai)


@router.post("/predict/pcos", response_model=PredictionResponse)
async def predict_pcos(
    payload: PcosRequest,
    include_xai: bool = True
) -> PredictionResponse:
    """Prediction endpoint for PCOS model."""
    config = MODEL_REGISTRY["pcos"]
    # Use by_alias=True so the payload keys match the original CSV headers
    return _predict_with_model(payload.model_dump(by_alias=True), "pcos", config.feature_order, include_xai=include_xai)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Lightweight readiness probe."""
    try:
        predictor.ensure_model_loaded("ovarian")
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model file missing",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not ready: {str(exc)}",
        )

    return HealthResponse(status="ok")


@router.get("/model-info")
async def model_info(model: str | None = None) -> dict:
    """
    Operational debug endpoint: shows which model path is used and feature counts.
    """
    model_key = model or "ovarian"
    if model_key not in MODEL_REGISTRY:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown model: {model_key}")
    try:
        predictor.ensure_model_loaded(model_key)
    except Exception:
        # Still return useful info even if model didn't load
        return predictor.get_model_info(model_key)
    return predictor.get_model_info(model_key)


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


