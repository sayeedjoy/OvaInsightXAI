"""Base types, caches, and utility functions for model loading and prediction."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple, Union

from app.model.registry import MODEL_REGISTRY, ModelConfig
from app.utils.config import MODEL_PATH

logger = logging.getLogger(__name__)

# Model data version - increment this to force retraining with new training data
# v2: Balanced classes with clear separation between positive/negative cases
MODEL_DATA_VERSION = "v2"

# Debug log path setup
_workspace_log = Path(r"c:\Users\User\Downloads\new\.cursor\debug.log")
_docker_log = Path("/app/.cursor/debug.log") if Path("/app").exists() else None
DEBUG_LOG_PATH = _workspace_log if _workspace_log.parent.exists() else (_docker_log if _docker_log and _docker_log.parent.exists() else Path(".cursor/debug.log"))
DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_DEFAULT_MODEL_KEY = "ovarian"

# Per-model caches (shared across all loaders and predictors)
_MODELS: dict[str, Any] = {}
_MODEL_ARTIFACTS: dict[str, Any] = {}
_SUPPORTS_PROBA: dict[str, bool] = {}
_LOADED_MODEL_PATHS: dict[str, Path] = {}
_LAST_LOAD_ERRORS: dict[str, str | None] = {}


class PredictionResult(NamedTuple):
    """Result from model prediction with optional confidence score."""
    prediction: Union[int, float]
    confidence: float | None


def get_config(model_key: str) -> ModelConfig:
    """Get model configuration from registry."""
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {model_key}")
    return MODEL_REGISTRY[model_key]


def resolve_primary_path(config: ModelConfig) -> Path:
    """Resolve the primary path for a model configuration."""
    # Preserve existing env override for ovarian model only
    if config.key == "ovarian":
        return MODEL_PATH
    return config.path


def expected_feature_count(config: ModelConfig) -> int:
    """Get expected feature count from model configuration."""
    return len(config.feature_order)


def get_model_feature_count(model: Any) -> int | None:
    """
    Try to determine the number of features the loaded estimator expects.

    Most sklearn estimators expose `n_features_in_` after fitting.
    """
    n = getattr(model, "n_features_in_", None)
    if isinstance(n, int):
        return n
    try:
        return int(n) if n is not None else None
    except Exception:
        return None


def get_model_feature_names(model: Any) -> list[str] | None:
    """
    Try to get the feature names the model expects.
    
    Models trained with pandas DataFrames store feature names in `feature_names_in_`.
    For Pipeline objects, we check the Pipeline first, then the final estimator.
    """
    try:
        from sklearn.pipeline import Pipeline
        
        # For Pipeline objects, check the Pipeline's feature_names_in_ first
        if isinstance(model, Pipeline):
            # Pipelines may have feature_names_in_ at the pipeline level
            pipeline_feature_names = getattr(model, "feature_names_in_", None)
            if pipeline_feature_names is not None:
                return list(pipeline_feature_names)
            
            # Otherwise, check the final estimator
            if hasattr(model, "steps") and len(model.steps) > 0:
                final_estimator = model.steps[-1][1]
                final_feature_names = getattr(final_estimator, "feature_names_in_", None)
                if final_feature_names is not None:
                    return list(final_feature_names)
        
        # For direct estimators, check feature_names_in_
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None:
            return list(feature_names)
    except Exception as exc:
        logger.debug("Error getting model feature names: %s", exc)
    
    return None


def candidate_model_paths(primary: Path, config: ModelConfig) -> list[Path]:
    """
    Return a list of paths to try, in order.

    This helps in Docker deployments where MODEL_PATH may point to a mounted file
    that is missing or outdated, while the image contains a baked model.
    """
    candidates: list[Path] = []

    def add(p: Path) -> None:
        if p not in candidates:
            candidates.append(p)

    add(primary)

    in_docker = Path("/app").exists()
    if in_docker:
        # Keep backwards-compatible docker fallbacks for ovarian only
        if config.key == "ovarian":
            add(Path("/app/app/model/model.pkl"))
            add(Path("/opt/models/model.pkl"))

    return candidates
