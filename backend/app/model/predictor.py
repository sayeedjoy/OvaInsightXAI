"""Load the serialized model once and expose simple prediction helpers."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple, Sequence, Union

from app.utils.config import MODEL_PATH, validate_feature_iterable

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_ARTIFACT: Any = None
_SUPPORTS_PROBA = False


class PredictionResult(NamedTuple):
    prediction: Union[int, float]
    confidence: float | None


def _load_model(path: Path):
    """Internal helper to deserialize the pickle file."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")

    try:
        import joblib  # type: ignore
    except ImportError:  # pragma: no cover - joblib optional
        joblib = None

    if joblib:
        logger.info("Loading model via joblib from %s", path)
        return joblib.load(path)

    logger.info("Joblib unavailable, falling back to pickle for %s", path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def _extract_estimator(artifact: Any):
    """
    Return the actual estimator object from the loaded artifact.

    Some training scripts persist additional metadata (scalers, configs, etc.)
    in a dict. We only need the object that implements predict()/predict_proba.
    """

    if hasattr(artifact, "predict"):
        return artifact

    if isinstance(artifact, Mapping):
        candidate_keys = (
            "model",
            "estimator",
            "classifier",
            "meta_logreg",
        )
        for key in candidate_keys:
            value = artifact.get(key)
            if value is not None and hasattr(value, "predict"):
                logger.info("Extracted estimator from artifact key '%s'.", key)
                return value
        raise ValueError(
            "Loaded model artifact is a mapping but none of the expected keys "
            "('model', 'estimator', 'classifier', 'meta_logreg') contain a valid estimator."
        )

    raise TypeError(
        f"Loaded artifact of type {type(artifact)!r} does not expose predict(). "
        "Update predictor._extract_estimator to handle this format."
    )


def ensure_model_loaded() -> None:
    """Load the model into memory if it has not been loaded yet."""
    global _MODEL, _MODEL_ARTIFACT, _SUPPORTS_PROBA
    if _MODEL is not None:
        return

    _MODEL_ARTIFACT = _load_model(MODEL_PATH)
    _MODEL = _extract_estimator(_MODEL_ARTIFACT)
    _SUPPORTS_PROBA = hasattr(_MODEL, "predict_proba")
    logger.info(
        "Model loaded (%s). Supports predict_proba=%s",
        type(_MODEL).__name__,
        _SUPPORTS_PROBA,
    )


def predict(features: Sequence[float]) -> PredictionResult:
    """Run inference on a single vector of features."""
    if _MODEL is None:
        raise RuntimeError("Model has not been loaded yet. Call ensure_model_loaded().")

    vector = validate_feature_iterable(features)
    # Most sklearn-style estimators expect a 2D array: (n_samples, n_features)
    batch = [vector]
    raw_prediction = _MODEL.predict(batch)[0]
    confidence = None

    if _SUPPORTS_PROBA:
        proba = _MODEL.predict_proba(batch)[0]
        confidence = float(max(proba))

    # Cast numpy scalars to native Python types for JSON serialization.
    prediction_value: Union[int, float] = (
        raw_prediction.item() if hasattr(raw_prediction, "item") else raw_prediction
    )

    return PredictionResult(prediction=prediction_value, confidence=confidence)


def warmup() -> None:
    """Convenience hook to load the model during app startup."""
    try:
        ensure_model_loaded()
    except FileNotFoundError as exc:
        # Log loudly but allow the app to keep starting so health checks reveal the issue.
        logger.error("Unable to warm up model: %s", exc)
        logger.warning("Model file missing. Health check will report this issue.")
        # Don't raise - let the app start so we can see the error via health check
        # The app will start but prediction endpoints will fail
    except Exception as exc:
        # Log any other errors but don't crash the app
        logger.error("Error during model warmup: %s", exc, exc_info=True)
        logger.warning("Application will start but model may not be available")


