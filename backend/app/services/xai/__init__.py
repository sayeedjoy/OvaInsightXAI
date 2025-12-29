"""XAI (Explainable AI) service package for computing model explanations.

OPTIMIZED FOR LOW CPU USAGE:
- Sequential execution by default (XAI_PARALLEL=false)
- Essential-only mode (SHAP + LIME only, skip PDP/ICE/ALE)
- Ultra-light parameter defaults
- Lazy/on-demand method loading
"""

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
# PERFORMANCE OPTIMIZATIONS:
# - XAI_PARALLEL=false (default) to avoid CPU contention
# - XAI_ESSENTIAL_ONLY=true to compute only SHAP+LIME
# - XAI_ENABLED=false to completely disable XAI
XAI_TIMEOUT_SECONDS = int(os.getenv("XAI_TIMEOUT_SECONDS", "60"))
XAI_PARALLEL = os.getenv("XAI_PARALLEL", "false").lower() == "true"  # Default to sequential
XAI_ESSENTIAL_ONLY = os.getenv("XAI_ESSENTIAL_ONLY", "true").lower() == "true"  # Only SHAP + LIME
XAI_ENABLED = os.getenv("XAI_ENABLED", "true").lower() == "true"  # Master switch


def _safe_compute(func, *args, **kwargs) -> dict[str, Any]:
    """Safely execute an XAI computation and catch any exceptions."""
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        logger.error("XAI computation failed for %s: %s", func.__name__, exc, exc_info=True)
        return {"error": str(exc), "error_type": type(exc).__name__}


def _placeholder_result(method_name: str) -> dict[str, Any]:
    """Return a placeholder result indicating method was skipped for performance."""
    return {
        "skipped": True,
        "reason": f"{method_name} computation skipped for performance (XAI_ESSENTIAL_ONLY=true)",
        "hint": "Set XAI_ESSENTIAL_ONLY=false to enable all XAI methods"
    }


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
    """Compute XAI explanations for an image prediction.
    
    OPTIMIZED FOR LOW CPU:
    - Default: Sequential execution (not parallel)
    - Default: Essential only (SHAP + LIME)
    - Can be disabled entirely via XAI_ENABLED=false
    
    Args:
        model_key: The model key (e.g., "brain_tumor")
        image_tensor: Preprocessed image tensor (PyTorch tensor)
    
    Returns:
        Dictionary with XAI explanations
    """
    # Master switch to disable all XAI
    if not XAI_ENABLED:
        logger.info("XAI disabled via XAI_ENABLED=false")
        return {
            "shap": {"skipped": True, "reason": "XAI disabled via XAI_ENABLED=false"},
            "lime": {"skipped": True, "reason": "XAI disabled via XAI_ENABLED=false"},
            "pdp_1d": {"skipped": True, "reason": "XAI disabled via XAI_ENABLED=false"},
            "ice_1d": {"skipped": True, "reason": "XAI disabled via XAI_ENABLED=false"},
            "ale_1d": {"skipped": True, "reason": "XAI disabled via XAI_ENABLED=false"},
        }
    
    # Always run sequentially for image XAI to avoid CPU spikes
    # (parallel processing 5 PyTorch model inferences causes 100% CPU)
    return _compute_image_xai_sequential_optimized(model_key, image_tensor)


def _compute_image_xai_parallel(model_key: str, image_tensor: Any) -> dict[str, Any]:
    """Compute image XAI explanations in parallel.
    
    WARNING: This can cause 100% CPU usage! Use sequential mode instead.
    """
    results = {}
    
    # Limit workers to 2 to reduce CPU contention
    max_workers = 2 if XAI_ESSENTIAL_ONLY else 3
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit essential tasks
        futures = {
            "shap": executor.submit(_safe_compute, compute_image_shap_explanation, model_key, image_tensor),
            "lime": executor.submit(_safe_compute, compute_image_lime_explanation, model_key, image_tensor),
        }
        
        # Submit non-essential tasks only if not in essential-only mode
        if not XAI_ESSENTIAL_ONLY:
            futures["pdp_1d"] = executor.submit(_safe_compute, compute_image_pdp_explanation, model_key, image_tensor)
            futures["ice_1d"] = executor.submit(_safe_compute, compute_image_ice_explanation, model_key, image_tensor)
            futures["ale_1d"] = executor.submit(_safe_compute, compute_image_ale_explanation, model_key, image_tensor)
        
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
    
    # Add placeholders for skipped methods
    if XAI_ESSENTIAL_ONLY:
        results["pdp_1d"] = _placeholder_result("PDP")
        results["ice_1d"] = _placeholder_result("ICE")
        results["ale_1d"] = _placeholder_result("ALE")
    
    return results


def _compute_image_xai_sequential_optimized(model_key: str, image_tensor: Any) -> dict[str, Any]:
    """Compute image XAI explanations sequentially with optimizations.
    
    This is the recommended mode for cloud deployments to prevent CPU spikes.
    - Runs one XAI method at a time
    - In essential-only mode, computes only SHAP and LIME
    - Other methods return placeholder results
    """
    logger.info("Computing image XAI sequentially (CPU-optimized mode)")
    
    results = {}
    
    # Essential methods: always compute SHAP and LIME
    logger.debug("Computing SHAP explanation...")
    results["shap"] = _safe_compute(compute_image_shap_explanation, model_key, image_tensor)
    
    logger.debug("Computing LIME explanation...")
    results["lime"] = _safe_compute(compute_image_lime_explanation, model_key, image_tensor)
    
    # Non-essential methods: compute only if XAI_ESSENTIAL_ONLY=false
    if XAI_ESSENTIAL_ONLY:
        logger.info("Skipping PDP/ICE/ALE (XAI_ESSENTIAL_ONLY=true)")
        results["pdp_1d"] = _placeholder_result("PDP")
        results["ice_1d"] = _placeholder_result("ICE")
        results["ale_1d"] = _placeholder_result("ALE")
    else:
        logger.debug("Computing PDP explanation...")
        results["pdp_1d"] = _safe_compute(compute_image_pdp_explanation, model_key, image_tensor)
        
        logger.debug("Computing ICE explanation...")
        results["ice_1d"] = _safe_compute(compute_image_ice_explanation, model_key, image_tensor)
        
        logger.debug("Computing ALE explanation...")
        results["ale_1d"] = _safe_compute(compute_image_ale_explanation, model_key, image_tensor)
    
    return results


def _compute_image_xai_sequential(model_key: str, image_tensor: Any) -> dict[str, Any]:
    """Compute image XAI explanations sequentially (legacy mode - computes all 5)."""
    results = {
        "shap": _safe_compute(compute_image_shap_explanation, model_key, image_tensor),
        "lime": _safe_compute(compute_image_lime_explanation, model_key, image_tensor),
        "pdp_1d": _safe_compute(compute_image_pdp_explanation, model_key, image_tensor),
        "ice_1d": _safe_compute(compute_image_ice_explanation, model_key, image_tensor),
        "ale_1d": _safe_compute(compute_image_ale_explanation, model_key, image_tensor),
    }
    
    return results
