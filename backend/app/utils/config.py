"""Centralized configuration and constants for the FastAPI backend."""

import os
from pathlib import Path
from typing import Iterable, Mapping

# Paths
APP_DIR = Path(__file__).resolve().parent.parent

# Model path: Read from environment variable, default to Docker-friendly path
# Falls back to relative path for local development if Docker path doesn't exist
_model_path_env = os.getenv("MODEL_PATH", "/app/app/model/model.pkl")
MODEL_PATH = Path(_model_path_env)
# If Docker path doesn't exist and we're not in Docker, use relative path
if not MODEL_PATH.exists() and not os.path.exists("/app"):
    MODEL_PATH = APP_DIR / "model" / "model.pkl"

# Frontend origins allowed to access the API.
# Read from environment variable (comma-separated), default to localhost for development
_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS: list[str] = [
    origin.strip() for origin in _origins_env.split(",") if origin.strip()
]

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


