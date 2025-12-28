"""sklearn/tabular model prediction logic."""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd

from app.model.base import (
    DEBUG_LOG_PATH,
    PredictionResult,
    _MODELS,
    _SUPPORTS_PROBA,
    expected_feature_count,
    get_config,
    get_model_feature_names,
)
from app.utils.config import validate_feature_iterable

logger = logging.getLogger(__name__)


def predict_tabular(features: Sequence[float], *, model_key: str = "ovarian") -> PredictionResult:
    """Run inference on a single vector of features for tabular/sklearn models."""
    if model_key not in _MODELS:
        raise RuntimeError("Model has not been loaded yet. Call ensure_model_loaded().")

    config = get_config(model_key)
    vector = validate_feature_iterable(features)
    
    logger.debug("Input features (%s): %s", model_key, vector)
    logger.debug("Feature count: %d (expected: %d)", len(vector), expected_feature_count(config))
    
    # Convert to numpy array with proper shape: (n_samples, n_features)
    try:
        batch_array = np.array(vector).reshape(1, -1)
        logger.debug("Batch shape: %s, dtype: %s", batch_array.shape, batch_array.dtype)
    except Exception as exc:
        logger.error("Failed to create numpy array from features: %s", exc, exc_info=True)
        raise ValueError(f"Failed to prepare feature array: {str(exc)}") from exc
    
    # Convert to pandas DataFrame with feature names
    try:
        model = _MODELS[model_key]
        model_feature_names = get_model_feature_names(model)
        
        if model_feature_names is not None:
            if len(model_feature_names) != len(batch_array[0]):
                logger.warning(
                    "Model feature names count (%d) doesn't match feature vector length (%d). "
                    "Falling back to numpy array.",
                    len(model_feature_names), len(batch_array[0])
                )
                batch = batch_array
            else:
                feature_names_to_use = model_feature_names
                logger.debug("Using model's expected feature names: %s", feature_names_to_use[:5])
                batch = pd.DataFrame(batch_array, columns=feature_names_to_use)
                logger.debug("Converted to DataFrame with %d columns", len(batch.columns))
        else:
            logger.debug("Model doesn't expose feature_names_in_, trying registry feature names")
            batch = pd.DataFrame(batch_array, columns=config.feature_order)
            logger.debug("Converted to DataFrame with registry feature names: %s", list(batch.columns)[:5])
    except Exception as exc:
        logger.warning("Failed to convert to DataFrame with feature names: %s. Falling back to numpy array.", exc)
        batch = batch_array
    
    # Debug logging
    try:
        import sklearn
        runtime_sklearn = sklearn.__version__
    except Exception:
        runtime_sklearn = "unknown"
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            batch_shape = list(batch.shape) if hasattr(batch, "shape") else "unknown"
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "sklearn_predictor.py:predict_tabular:before_predict", "message": "Before prediction", "data": {"model_key": model_key, "model_type": str(type(_MODELS[model_key])), "runtime_sklearn": runtime_sklearn, "batch_shape": batch_shape, "supports_proba": _SUPPORTS_PROBA[model_key]}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
    except Exception:
        pass
    
    # Run prediction
    try:
        logger.debug("Calling model.predict() with batch shape: %s", batch.shape)
        raw_prediction = _MODELS[model_key].predict(batch)[0]
        logger.debug("Raw prediction result: %s (type: %s)", raw_prediction, type(raw_prediction))
    except (AttributeError, TypeError) as exc:
        if "_fill_dtype" in str(exc) or "has no attribute" in str(exc):
            logger.error("Model prediction failed due to sklearn version mismatch: %s", exc, exc_info=True)
            try:
                import sklearn
                sklearn_version = sklearn.__version__
            except ImportError:
                sklearn_version = "unknown"
            raise ValueError(
                f"Model prediction failed due to sklearn version incompatibility (version: {sklearn_version}). "
                "The model may need to be retrained with the current sklearn version."
            ) from exc
        logger.error("Model prediction failed with AttributeError/TypeError: %s", exc, exc_info=True)
        raise
    except Exception as exc:
        logger.error("Unexpected error during model prediction: %s", exc, exc_info=True)
        raise ValueError(f"Model prediction failed: {str(exc)}") from exc
    
    # Get confidence if available
    confidence = None
    if _SUPPORTS_PROBA.get(model_key):
        try:
            logger.debug("Calling model.predict_proba() with batch shape: %s", batch.shape)
            proba = _MODELS[model_key].predict_proba(batch)[0]
            confidence = float(max(proba))
            logger.debug("Prediction probabilities: %s, confidence: %s", proba, confidence)
        except (AttributeError, TypeError) as exc:
            logger.warning("predict_proba failed, continuing without confidence: %s", exc)
        except Exception as exc:
            logger.warning("Unexpected error during predict_proba: %s", exc, exc_info=True)
    
    # Cast numpy scalars to native Python types
    prediction_value: Union[int, float] = (
        raw_prediction.item() if hasattr(raw_prediction, "item") else raw_prediction
    )
    
    logger.info("Prediction completed for %s: prediction=%s, confidence=%s", model_key, prediction_value, confidence)

    return PredictionResult(prediction=prediction_value, confidence=confidence)
