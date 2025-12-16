"""Centralized configuration and constants for the FastAPI backend."""

from pathlib import Path
from typing import Iterable, Mapping

# Paths
APP_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = APP_DIR / "model" / "model.pkl"

# Frontend origins allowed to access the API.
ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]

# Ordered list of feature keys used by the ML model.
FEATURE_ORDER: list[str] = [
    "age",
    "alb",
    "alp",
    "bun",
    "ca125",
    "eo_abs",
    "ggt",
    "he4",
    "mch",
    "mono_abs",
    "na",
    "pdw",
]


class FeatureOrderError(ValueError):
    """Raised when the incoming payload is missing required features."""


def ordered_feature_vector(payload: Mapping[str, float]) -> list[float]:
    """Return the model-ready feature vector following FEATURE_ORDER."""
    missing: list[str] = [key for key in FEATURE_ORDER if key not in payload]
    if missing:
        raise FeatureOrderError(
            f"Missing feature(s) required for inference: {', '.join(missing)}"
        )
    return [float(payload[key]) for key in FEATURE_ORDER]


def validate_feature_iterable(values: Iterable[float]) -> list[float]:
    """Ensure iterable can be consumed multiple times and cast to floats."""
    return [float(value) for value in values]


