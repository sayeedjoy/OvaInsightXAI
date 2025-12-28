"""
Load the serialized model once and expose simple prediction helpers.

This module serves as the backward-compatible facade for the refactored model loading
and prediction system. All public functions are re-exported from submodules.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Sequence

# Re-export from base module
from app.model.base import (
    DEBUG_LOG_PATH,
    MODEL_DATA_VERSION,
    PredictionResult,
    _DEFAULT_MODEL_KEY,
    _LAST_LOAD_ERRORS,
    _LOADED_MODEL_PATHS,
    _MODEL_ARTIFACTS,
    _MODELS,
    _SUPPORTS_PROBA,
    candidate_model_paths,
    expected_feature_count,
    get_config,
    get_model_feature_count,
    get_model_feature_names,
    resolve_primary_path,
)
from app.model.registry import MODEL_REGISTRY, ModelConfig

# Import loader functions
from app.model.loaders import (
    extract_estimator,
    load_pytorch_model,
    load_sklearn_model,
    retrain_model_if_needed,
    validate_model_feature_count,
)

# Import predictor functions
from app.model.predictors import predict_image, predict_tabular

logger = logging.getLogger(__name__)


# ============================================================================
# Backward-compatible aliases (private functions with underscore prefix)
# ============================================================================

_get_config = get_config
_resolve_primary_path = resolve_primary_path
_expected_feature_count = expected_feature_count
_get_model_feature_count = get_model_feature_count
_get_model_feature_names = get_model_feature_names
_candidate_model_paths = candidate_model_paths
_validate_model_feature_count = validate_model_feature_count
_load_model = load_sklearn_model
_load_pytorch_model = load_pytorch_model
_extract_estimator = extract_estimator
_retrain_model_if_needed = retrain_model_if_needed


def _load_and_extract(path: Path, *, config: ModelConfig) -> tuple[Any, Any]:
    """Load artifact from path and extract estimator/pipeline."""
    # Check if this is a PyTorch model
    if path.suffix == ".pth" or config.key == "brain_tumor":
        model = load_pytorch_model(path)
        return model, model
    
    # sklearn model loading
    if config.key == "ovarian":
        retrain_model_if_needed(path)
    artifact = load_sklearn_model(path)
    model = extract_estimator(artifact)
    # Validate feature-count for models where schema must align with API expectations
    if config.key in ("ovarian", "hepatitis_b"):
        validate_model_feature_count(model, model_path=path, config=config)
    return artifact, model


def get_model_info(model_key: str = _DEFAULT_MODEL_KEY) -> dict[str, Any]:
    """
    Return debug information about the model selection and feature expectations.

    Intended for operational debugging (e.g., verifying VPS loads the correct model).
    """
    config = get_config(model_key)
    primary_path = resolve_primary_path(config)
    expected = expected_feature_count(config)
    
    artifact = _MODEL_ARTIFACTS.get(model_key)
    model = _MODELS.get(model_key)
    loaded_path = _LOADED_MODEL_PATHS.get(model_key)
    actual = get_model_feature_count(model) if model is not None else None

    # Get model data version from artifact if available
    model_data_version = None
    model_sklearn_version = None
    if isinstance(artifact, Mapping):
        model_data_version = artifact.get("model_data_version", "unknown")
        model_sklearn_version = artifact.get("sklearn_version", "unknown")
    
    info: dict[str, Any] = {
        "model_key": model_key,
        "configured_model_path": str(primary_path),
        "configured_model_path_exists": primary_path.exists(),
        "expected_feature_count": expected,
        "expected_feature_order": config.feature_order,
        "loaded_model_path": str(loaded_path) if loaded_path else None,
        "loaded_model_type": type(model).__name__ if model is not None else None,
        "loaded_model_feature_count": actual,
        "feature_count_match": (actual == expected) if actual is not None else None,
        "last_load_error": _LAST_LOAD_ERRORS.get(model_key),
        "candidate_paths": [str(p) for p in candidate_model_paths(primary_path, config)],
        "current_model_data_version": MODEL_DATA_VERSION,
        "loaded_model_data_version": model_data_version,
        "loaded_model_sklearn_version": model_sklearn_version,
    }
    return info


def feature_self_check(model_key: str) -> dict[str, Any]:
    """
    Lightweight guardrail to verify a model can be loaded and its feature
    count matches the configured schema. Useful for operational diagnostics
    and quick unit/integration checks.
    """
    try:
        ensure_model_loaded(model_key)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "model_key": model_key,
        }

    info = get_model_info(model_key)
    return {
        "ok": info.get("feature_count_match") is True,
        "model_key": model_key,
        "expected_feature_count": info.get("expected_feature_count"),
        "loaded_model_feature_count": info.get("loaded_model_feature_count"),
        "last_load_error": info.get("last_load_error"),
    }


def ensure_model_loaded(model_key: str = _DEFAULT_MODEL_KEY) -> None:
    """Load the model into memory if it has not been loaded yet."""
    if model_key in _MODELS:
        return

    config = get_config(model_key)
    primary_path = resolve_primary_path(config)
    candidates = candidate_model_paths(primary_path, config)

    last_error: Exception | None = None
    for path in candidates:
        if not path.exists():
            logger.debug("Candidate model path does not exist: %s", path)
            continue
        try:
            artifact, model = _load_and_extract(path, config=config)
            _MODEL_ARTIFACTS[model_key] = artifact
            _MODELS[model_key] = model
            _SUPPORTS_PROBA[model_key] = callable(getattr(model, "predict_proba", None))
            _LOADED_MODEL_PATHS[model_key] = path
            _LAST_LOAD_ERRORS[model_key] = None
            break
        except Exception as exc:
            logger.warning("Failed to load model from %s: %s", path, exc)
            _LAST_LOAD_ERRORS[model_key] = str(exc)
            last_error = exc

    if model_key not in _MODELS:
        raise FileNotFoundError(
            f"Unable to load a compatible model for key={model_key}. Last error: {last_error}"
        ) from last_error
    
    logger.info(
        "Model '%s' loaded successfully (%s). Supports predict_proba=%s",
        model_key,
        type(_MODELS[model_key]).__name__,
        _SUPPORTS_PROBA[model_key],
    )


def get_model(model_key: str = _DEFAULT_MODEL_KEY) -> Any:
    """Get the loaded model for a given key."""
    if model_key not in _MODELS:
        raise RuntimeError("Model has not been loaded yet. Call ensure_model_loaded().")
    return _MODELS[model_key]


def predict(features: Sequence[float], *, model_key: str = _DEFAULT_MODEL_KEY) -> PredictionResult:
    """Run inference on a single vector of features."""
    return predict_tabular(features, model_key=model_key)


def warmup(model_keys: Iterable[str] | None = None) -> None:
    """Convenience hook to load the model(s) during app startup."""
    keys = list(model_keys) if model_keys is not None else list(MODEL_REGISTRY.keys())
    for key in keys:
        try:
            ensure_model_loaded(key)
        except FileNotFoundError as exc:
            logger.error("Unable to warm up model '%s': %s", key, exc)
            logger.warning("Model file missing for '%s'. Health check will report this issue.", key)
        except Exception as exc:
            logger.error("Error during model warmup for '%s': %s", key, exc, exc_info=True)
            logger.warning("Application will start but model '%s' may not be available", key)


# Re-export for backward compatibility
__all__ = [
    # Types
    "PredictionResult",
    "ModelConfig",
    # Public functions
    "ensure_model_loaded",
    "feature_self_check",
    "get_model",
    "get_model_info",
    "predict",
    "predict_image",
    "warmup",
    # Constants
    "MODEL_DATA_VERSION",
    "MODEL_REGISTRY",
]
