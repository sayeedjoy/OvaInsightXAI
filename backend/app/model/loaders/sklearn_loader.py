"""sklearn/pickle model loading logic."""

from __future__ import annotations

import json
import logging
import pickle
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.model.base import (
    DEBUG_LOG_PATH,
    MODEL_DATA_VERSION,
    expected_feature_count,
    get_model_feature_count,
    logger,
)
from app.model.registry import ModelConfig

logger = logging.getLogger(__name__)


def validate_model_feature_count(model: Any, *, model_path: Path, config: ModelConfig) -> None:
    """Validate that model feature count matches expected count from config."""
    expected = expected_feature_count(config)
    actual = get_model_feature_count(model)
    if actual is None:
        logger.warning(
            "Loaded model does not expose n_features_in_; cannot validate feature count. path=%s type=%s",
            model_path,
            type(model).__name__,
        )
        return

    if actual != expected:
        raise ValueError(
            f"Loaded model expects {actual} features but API is configured for {expected} "
            f"following feature order: {config.feature_order}. "
            f"Fix the model artifact to match the API schema or update the configured feature order. "
            f"Loaded from: {model_path}"
        )


def load_sklearn_model(path: Path) -> Any:
    """Load sklearn model from pickle/joblib file."""
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "sklearn_loader.py:load_sklearn_model:entry", "message": "Loading model", "data": {"model_path": str(path), "path_exists": path.exists(), "path_absolute": str(path.resolve())}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
    except: pass
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")

    try:
        import joblib
    except ImportError:
        joblib = None
    
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = "unknown"

    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "sklearn_loader.py:load_sklearn_model:sklearn_version", "message": "Runtime sklearn version", "data": {"sklearn_version": sklearn_version, "joblib_available": joblib is not None}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
    except: pass

    logger.info("Loading model with scikit-learn version: %s", sklearn_version)

    if joblib:
        logger.info("Loading model via joblib from %s", path)

        # Compatibility shim for unpickling unknown DeployableModel
        class _ShimDeployableModel:
            """Compatibility shim for unpickling unknown DeployableModel."""
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        main_module = sys.modules.get("__main__")
        if main_module and not hasattr(main_module, "DeployableModel"):
            setattr(main_module, "DeployableModel", _ShimDeployableModel)

        try:
            artifact = joblib.load(path)
            try:
                with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "sklearn_loader.py:load_sklearn_model:after_load", "message": "Model artifact loaded", "data": {"artifact_type": str(type(artifact)), "is_mapping": isinstance(artifact, Mapping), "has_pipeline_key": isinstance(artifact, Mapping) and "pipeline" in artifact, "file_size": path.stat().st_size if path.exists() else 0}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            
            # Check if model has metadata and verify sklearn version
            if isinstance(artifact, Mapping) and "pipeline" in artifact:
                model_sklearn_version = artifact.get("sklearn_version", "unknown")
                logger.info("Model was trained with sklearn version: %s", model_sklearn_version)
                
                if model_sklearn_version != "unknown" and model_sklearn_version != sklearn_version:
                    logger.warning(
                        "Model sklearn version (%s) differs from current version (%s). "
                        "This may cause compatibility issues.",
                        model_sklearn_version,
                        sklearn_version
                    )
                return artifact
            else:
                logger.warning(
                    "Model does not contain version metadata. "
                    "It may have been trained with a different sklearn version."
                )
                return artifact
                
        except (ModuleNotFoundError, ImportError) as exc:
            missing_module = str(exc).split("'")[1] if "'" in str(exc) else "unknown module"
            logger.error(
                "Failed to load model: missing required module '%s'. "
                "Please install it (e.g., 'pip install %s') and restart the application.",
                missing_module,
                missing_module
            )
            raise ImportError(
                f"Model requires module '{missing_module}' which is not installed. "
                f"Please install it (e.g., 'pip install {missing_module}') and restart the application."
            ) from exc
        except (AttributeError, TypeError) as exc:
            if "_fill_dtype" in str(exc) or "has no attribute" in str(exc):
                logger.error(
                    "Model version mismatch detected. The model was trained with a different "
                    "scikit-learn version. Current version: %s. Error: %s",
                    sklearn_version,
                    exc
                )
                raise ValueError(
                    f"Model incompatible with current scikit-learn version ({sklearn_version}). "
                    "Please retrain the model with the current sklearn version."
                ) from exc
            raise

    logger.info("Joblib unavailable, falling back to pickle for %s", path)
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (ModuleNotFoundError, ImportError) as exc:
        missing_module = str(exc).split("'")[1] if "'" in str(exc) else "unknown module"
        logger.error(
            "Failed to load model via pickle: missing required module '%s'. "
            "Please install it (e.g., 'pip install %s') and restart the application.",
            missing_module,
            missing_module
        )
        raise ImportError(
            f"Model requires module '{missing_module}' which is not installed. "
            f"Please install it (e.g., 'pip install {missing_module}') and restart the application."
        ) from exc


def extract_estimator(artifact: Any) -> Any:
    """
    Return the actual estimator object from the loaded artifact.

    Some training scripts persist additional metadata (scalers, configs, etc.)
    in a dict. We only need the object that implements predict()/predict_proba.
    """
    if hasattr(artifact, "predict"):
        return artifact

    if isinstance(artifact, Mapping):
        available_keys = list(artifact.keys())
        logger.debug("Artifact is a mapping with keys: %s", available_keys)
        
        # First check for new format with "pipeline" key
        if "pipeline" in artifact:
            pipeline = artifact["pipeline"]
            if hasattr(pipeline, "predict"):
                logger.info("Extracted pipeline from artifact with sklearn version: %s", 
                          artifact.get("sklearn_version", "unknown"))
                return pipeline
        
        # Check for common sklearn pipeline/estimator keys
        candidate_keys = (
            "model",
            "estimator",
            "classifier",
            "meta_logreg",
            "pipeline",
            "sklearn_pipeline",
            "clf",
            "classifier_model",
        )
        for key in candidate_keys:
            value = artifact.get(key)
            if value is not None and hasattr(value, "predict"):
                logger.info("Extracted estimator from artifact key '%s'.", key)
                return value
        
        # Try to find any value in the dict that has a predict method
        for key, value in artifact.items():
            if hasattr(value, "predict") and not isinstance(value, (str, int, float, bool, type(None))):
                logger.info("Extracted estimator from artifact key '%s' (auto-detected).", key)
                return value
        
        raise ValueError(
            f"Loaded model artifact is a mapping but none of the expected keys "
            f"('pipeline', 'model', 'estimator', 'classifier', 'meta_logreg') contain a valid estimator. "
            f"Available keys in artifact: {available_keys}. "
            f"Please check the model file structure or update extract_estimator to handle this format."
        )

    raise TypeError(
        f"Loaded artifact of type {type(artifact)!r} does not expose predict(). "
        "Update extract_estimator to handle this format."
    )


def retrain_model_if_needed(model_path: Path) -> bool:
    """Retrain the model if sklearn version mismatch is detected. Returns True if retrained."""
    # Import the full retraining logic from the existing location
    # This is a placeholder - the actual implementation is complex and will be kept in a separate module
    try:
        import subprocess
        import sklearn
        sklearn_version = sklearn.__version__

        # Check if model needs retraining by trying to load it
        try:
            import joblib
            artifact = joblib.load(model_path)
            if isinstance(artifact, Mapping):
                model_sklearn_version = artifact.get("sklearn_version")
                model_data_version = artifact.get("model_data_version")
                
                if model_sklearn_version == sklearn_version and model_data_version == MODEL_DATA_VERSION:
                    logger.info("Model is compatible, no retraining needed")
                    return False
                    
                logger.warning(
                    "Model version mismatch detected. Model sklearn=%s (current=%s), data_version=%s (current=%s)",
                    model_sklearn_version, sklearn_version, model_data_version, MODEL_DATA_VERSION
                )
        except Exception as e:
            logger.warning("Could not check model version: %s", e)
            
        # Try to retrain
        retrain_script = model_path.parent.parent.parent / "retrain_model.py"
        if retrain_script.exists():
            logger.info("Attempting to retrain model using %s", retrain_script)
            result = subprocess.run(
                [sys.executable, str(retrain_script)],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                logger.info("Model retrained successfully")
                return True
            else:
                logger.error("Model retraining failed: %s", result.stderr)
                return False
        else:
            logger.warning("Retrain script not found at %s", retrain_script)
            return False
            
    except Exception as e:
        logger.error("Error during model retraining check: %s", e)
        return False
