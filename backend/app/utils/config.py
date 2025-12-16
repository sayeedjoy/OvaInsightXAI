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

_DEFAULT_DOCKER_MODEL_PATH = Path("/app/app/model/model.pkl")
_OPT_DOCKER_MODEL_PATH = Path("/opt/models/model.pkl")

_model_path_env = os.getenv("MODEL_PATH", str(_DEFAULT_DOCKER_MODEL_PATH))
_model_path_env_set = "MODEL_PATH" in os.environ
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
        # If MODEL_PATH env var was set but doesn't exist, try common Docker locations.
        if _model_path_env_set:
            if _DEFAULT_DOCKER_MODEL_PATH.exists():
                MODEL_PATH = _DEFAULT_DOCKER_MODEL_PATH
                logger.warning(
                    "MODEL_PATH was set to %s but file does not exist; falling back to baked model at %s",
                    _model_path_env,
                    MODEL_PATH,
                )
            elif _OPT_DOCKER_MODEL_PATH.exists():
                MODEL_PATH = _OPT_DOCKER_MODEL_PATH
                logger.warning(
                    "MODEL_PATH was set to %s but file does not exist; falling back to %s",
                    _model_path_env,
                    MODEL_PATH,
                )
            else:
                # Keep env-provided path; app startup may attempt download depending on config.
                logger.warning(
                    "MODEL_PATH was set to %s but no model file found at fallback locations (%s, %s).",
                    _model_path_env,
                    _DEFAULT_DOCKER_MODEL_PATH,
                    _OPT_DOCKER_MODEL_PATH,
                )
        else:
            # Env var not set: prefer baked image model, then /opt/models.
            if _DEFAULT_DOCKER_MODEL_PATH.exists():
                MODEL_PATH = _DEFAULT_DOCKER_MODEL_PATH
                logger.info("Using baked Docker model path: %s", MODEL_PATH)
            elif _OPT_DOCKER_MODEL_PATH.exists():
                MODEL_PATH = _OPT_DOCKER_MODEL_PATH
                logger.info("Using model from %s", MODEL_PATH)
            else:
                # Keep the default docker path even if it doesn't exist yet (it might be downloaded at startup).
                MODEL_PATH = _DEFAULT_DOCKER_MODEL_PATH
                logger.info("Model not found yet; will use %s (startup may attempt download)", MODEL_PATH)
    else:
        # Not in Docker: use relative path for local development
        MODEL_PATH = APP_DIR / "model" / "model.pkl"
        logger.info("Using local development path: %s", MODEL_PATH)

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


