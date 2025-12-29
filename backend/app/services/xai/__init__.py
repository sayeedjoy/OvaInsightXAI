"""XAI (Explainable AI) service package for computing model explanations."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from app.services.xai.ale_explainer import compute_ale_1d
from app.services.xai.ice_explainer import compute_ice_1d
from app.services.xai.lime_explainer import compute_lime_explanation
from app.services.xai.pdp_explainer import compute_pdp_1d
from app.services.xai.shap_explainer import compute_shap_explanation
from app.services.xai.image_shap_explainer import compute_image_shap_explanation
from app.services.xai.image_lime_explainer import compute_image_lime_explanation
from app.services.xai.image_pdp_explainer import compute_image_pdp_explanation
from app.services.xai.image_ice_explainer import compute_image_ice_explanation
from app.services.xai.image_ale_explainer import compute_image_ale_explanation

logger = logging.getLogger(__name__)

__all__ = [
    "compute_shap_explanation",
    "compute_lime_explanation",
    "compute_pdp_1d",
    "compute_ice_1d",
    "compute_ale_1d",
    "compute_all_xai_explanations",
    "compute_image_shap_explanation",
    "compute_image_lime_explanation",
    "compute_image_pdp_explanation",
    "compute_image_ice_explanation",
    "compute_image_ale_explanation",
    "compute_all_image_xai_explanations",
]

# Environment-aware configuration
# In deployment, we use longer timeouts for image XAI which requires more computation
XAI_TIMEOUT_SECONDS = int(os.getenv("XAI_TIMEOUT_SECONDS", "60"))
XAI_PARALLEL = os.getenv("XAI_PARALLEL", "true").lower() == "true"


def _safe_compute(func, *args, **kwargs) -> dict[str, Any]:
    """Safely execute an XAI computation and catch any exceptions."""
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        logger.error("XAI computation failed for %s: %s", func.__name__, exc, exc_info=True)
        return {"error": str(exc), "error_type": type(exc).__name__}


def compute_all_xai_explanations(
    model_key: str,
    instance_features: list[float]
) -> dict[str, Any]:
    """Compute all XAI explanations for a prediction instance.
    
    Uses parallel computation for faster results. All 5 XAI methods
    run concurrently using ThreadPoolExecutor.
    """
    if XAI_PARALLEL:
        return _compute_parallel(model_key, instance_features)
    else:
        return _compute_sequential(model_key, instance_features)


def _compute_parallel(model_key: str, instance_features: list[float]) -> dict[str, Any]:
    """Compute all XAI explanations in parallel using ThreadPoolExecutor."""
    results = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        futures = {
            "shap": executor.submit(_safe_compute, compute_shap_explanation, model_key, instance_features),
            "lime": executor.submit(_safe_compute, compute_lime_explanation, model_key, instance_features),
            "pdp_1d": executor.submit(_safe_compute, compute_pdp_1d, model_key),
            "ice_1d": executor.submit(_safe_compute, compute_ice_1d, model_key),
            "ale_1d": executor.submit(_safe_compute, compute_ale_1d, model_key),
        }
        
        # Collect results with timeout
        for key, future in futures.items():
            try:
                results[key] = future.result(timeout=XAI_TIMEOUT_SECONDS)
            except FuturesTimeoutError:
                logger.warning("XAI computation timed out for %s after %ds", key, XAI_TIMEOUT_SECONDS)
                results[key] = {"error": f"Computation timed out after {XAI_TIMEOUT_SECONDS}s", "error_type": "TimeoutError"}
            except Exception as exc:
                logger.error("Unexpected error getting result for %s: %s", key, exc, exc_info=True)
                results[key] = {"error": str(exc), "error_type": type(exc).__name__}
    
    return results


def _compute_sequential(model_key: str, instance_features: list[float]) -> dict[str, Any]:
    """Compute all XAI explanations sequentially (fallback mode)."""
    return {
        "shap": _safe_compute(compute_shap_explanation, model_key, instance_features),
        "lime": _safe_compute(compute_lime_explanation, model_key, instance_features),
        "pdp_1d": _safe_compute(compute_pdp_1d, model_key),
        "ice_1d": _safe_compute(compute_ice_1d, model_key),
        "ale_1d": _safe_compute(compute_ale_1d, model_key),
    }


def compute_all_image_xai_explanations(
    model_key: str,
    image_tensor: Any
) -> dict[str, Any]:
    """Compute all XAI explanations for an image prediction.
    
    For image models, we compute SHAP, LIME, PDP, ICE, and ALE using
    patch-based approaches where image regions are treated as features.
    
    Args:
        model_key: The model key (e.g., "brain_tumor")
        image_tensor: Preprocessed image tensor (PyTorch tensor)
    
    Returns:
        Dictionary with all 5 XAI explanations (SHAP, LIME, PDP, ICE, ALE)
    """
    if XAI_PARALLEL:
        return _compute_image_xai_parallel(model_key, image_tensor)
    else:
        return _compute_image_xai_sequential(model_key, image_tensor)


def _compute_image_xai_parallel(model_key: str, image_tensor: Any) -> dict[str, Any]:
    """Compute image XAI explanations in parallel."""
    results = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all 5 image XAI tasks
        futures = {
            "shap": executor.submit(_safe_compute, compute_image_shap_explanation, model_key, image_tensor),
            "lime": executor.submit(_safe_compute, compute_image_lime_explanation, model_key, image_tensor),
            "pdp_1d": executor.submit(_safe_compute, compute_image_pdp_explanation, model_key, image_tensor),
            "ice_1d": executor.submit(_safe_compute, compute_image_ice_explanation, model_key, image_tensor),
            "ale_1d": executor.submit(_safe_compute, compute_image_ale_explanation, model_key, image_tensor),
        }
        
        # Collect results with timeout
        for key, future in futures.items():
            try:
                results[key] = future.result(timeout=XAI_TIMEOUT_SECONDS)
            except FuturesTimeoutError:
                logger.warning("Image XAI computation timed out for %s after %ds", key, XAI_TIMEOUT_SECONDS)
                results[key] = {"error": f"Computation timed out after {XAI_TIMEOUT_SECONDS}s", "error_type": "TimeoutError"}
            except Exception as exc:
                logger.error("Unexpected error getting result for %s: %s", key, exc, exc_info=True)
                results[key] = {"error": str(exc), "error_type": type(exc).__name__}
    
    return results


def _compute_image_xai_sequential(model_key: str, image_tensor: Any) -> dict[str, Any]:
    """Compute image XAI explanations sequentially (fallback mode)."""
    results = {
        "shap": _safe_compute(compute_image_shap_explanation, model_key, image_tensor),
        "lime": _safe_compute(compute_image_lime_explanation, model_key, image_tensor),
        "pdp_1d": _safe_compute(compute_image_pdp_explanation, model_key, image_tensor),
        "ice_1d": _safe_compute(compute_image_ice_explanation, model_key, image_tensor),
        "ale_1d": _safe_compute(compute_image_ale_explanation, model_key, image_tensor),
    }
    
    return results
