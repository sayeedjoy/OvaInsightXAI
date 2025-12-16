"""Centralized configuration and constants for the FastAPI backend."""

import os
from pathlib import Path
from typing import Iterable, Mapping

# Paths
APP_DIR = Path(__file__).resolve().parent.parent

# Model path: Read from environment variable, default to Docker-friendly path
# Falls back to relative path for local development if Docker path doesn't exist
import logging
logger = logging.getLogger(__name__)

_model_path_env = os.getenv("MODEL_PATH", "/app/app/model/model.pkl")
MODEL_PATH = Path(_model_path_env)

# Check multiple locations in order:
# 1. MODEL_PATH environment variable (if set and exists)
# 2. /app/app/model/model.pkl (default Docker path)
# 3. /opt/models/model.pkl (fallback for manually uploaded models in Docker)
# 4. Relative path app/model/model.pkl (for local development)

if not MODEL_PATH.exists():
    # Check if we're in Docker environment
    in_docker = os.path.exists("/app")
    
    if in_docker:
        # In Docker: check /opt/models/model.pkl as fallback
        opt_model_path = Path("/opt/models/model.pkl")
        if opt_model_path.exists():
            MODEL_PATH = opt_model_path
            logger.info(f"Using model from /opt/models/model.pkl")
        else:
            # Keep the default Docker path even if it doesn't exist yet
            # (it might be downloaded during startup)
            logger.info(f"Model not found at {MODEL_PATH} or /opt/models/model.pkl, will attempt download")
    else:
        # Not in Docker: use relative path for local development
        MODEL_PATH = APP_DIR / "model" / "model.pkl"
        logger.info(f"Using local development path: {MODEL_PATH}")

logger.info(f"MODEL_PATH configured as: {MODEL_PATH}")
if MODEL_PATH.exists():
    logger.info(f"Model file exists at: {MODEL_PATH}")
else:
    logger.warning(f"Model file does NOT exist at: {MODEL_PATH}")

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


