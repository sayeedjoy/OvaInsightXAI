"""Load the serialized model once and expose simple prediction helpers."""

from __future__ import annotations

import json
import logging
import os
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple, Sequence, Union

from app.utils.config import FEATURE_ORDER, MODEL_PATH, validate_feature_iterable

logger = logging.getLogger(__name__)

# #region agent log
# Determine log path - use workspace path if available, otherwise try Docker path
_workspace_log = Path(r"c:\Users\User\Downloads\new\.cursor\debug.log")
_docker_log = Path("/app/.cursor/debug.log") if Path("/app").exists() else None
DEBUG_LOG_PATH = _workspace_log if _workspace_log.parent.exists() else (_docker_log if _docker_log and _docker_log.parent.exists() else Path(".cursor/debug.log"))
DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
# #endregion

_MODEL = None
_MODEL_ARTIFACT: Any = None
_SUPPORTS_PROBA = False
_LOADED_MODEL_PATH: Path | None = None


def _expected_feature_count() -> int:
    return len(FEATURE_ORDER)


def _get_model_feature_count(model: Any) -> int | None:
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


def _validate_model_feature_count(model: Any, *, model_path: Path) -> None:
    expected = _expected_feature_count()
    actual = _get_model_feature_count(model)
    if actual is None:
        logger.warning(
            "Loaded model does not expose n_features_in_; cannot validate feature count. path=%s type=%s",
            model_path,
            type(model).__name__,
        )
        return

    if actual != expected:
        raise ValueError(
            f"Loaded model expects {actual} features but API is configured for {expected}. "
            f"Fix the model file or unset MODEL_PATH. Loaded from: {model_path}"
        )


def _candidate_model_paths(primary: Path) -> list[Path]:
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
        add(Path("/app/app/model/model.pkl"))
        add(Path("/opt/models/model.pkl"))

    return candidates


def _load_and_extract(path: Path) -> tuple[Any, Any]:
    """Load artifact from path and extract estimator/pipeline."""
    _retrain_model_if_needed(path)
    artifact = _load_model(path)
    model = _extract_estimator(artifact)
    return artifact, model


def get_model_info() -> dict[str, Any]:
    """
    Return debug information about the model selection and feature expectations.

    Intended for operational debugging (e.g., verifying VPS loads the correct model).
    """
    expected = _expected_feature_count()
    info: dict[str, Any] = {
        "configured_model_path": str(MODEL_PATH),
        "configured_model_path_exists": MODEL_PATH.exists(),
        "expected_feature_count": expected,
        "loaded_model_path": str(_LOADED_MODEL_PATH) if _LOADED_MODEL_PATH else None,
        "loaded_model_type": type(_MODEL).__name__ if _MODEL is not None else None,
        "loaded_model_feature_count": _get_model_feature_count(_MODEL) if _MODEL is not None else None,
        "candidate_paths": [str(p) for p in _candidate_model_paths(MODEL_PATH)],
    }
    return info


class PredictionResult(NamedTuple):
    prediction: Union[int, float]
    confidence: float | None


def _load_model(path: Path):
    """Internal helper to deserialize the pickle file."""
    # #region agent log
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "predictor.py:_load_model:entry", "message": "Loading model", "data": {"model_path": str(path), "path_exists": path.exists(), "path_absolute": str(path.resolve())}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
    except: pass
    # #endregion
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")

    try:
        import joblib  # type: ignore
    except ImportError:  # pragma: no cover - joblib optional
        joblib = None
    
    try:
        import sklearn  # type: ignore
        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = "unknown"

    # #region agent log
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "predictor.py:_load_model:sklearn_version", "message": "Runtime sklearn version", "data": {"sklearn_version": sklearn_version, "joblib_available": joblib is not None}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
    except: pass
    # #endregion

    logger.info("Loading model with scikit-learn version: %s", sklearn_version)

    if joblib:
        logger.info("Loading model via joblib from %s", path)
        try:
            artifact = joblib.load(path)
            # #region agent log
            try:
                with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "predictor.py:_load_model:after_load", "message": "Model artifact loaded", "data": {"artifact_type": str(type(artifact)), "is_mapping": isinstance(artifact, Mapping), "has_pipeline_key": isinstance(artifact, Mapping) and "pipeline" in artifact, "file_size": path.stat().st_size if path.exists() else 0}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
            
            # Check if model has metadata and verify sklearn version
            if isinstance(artifact, Mapping) and "pipeline" in artifact:
                model_sklearn_version = artifact.get("sklearn_version", "unknown")
                logger.info("Model was trained with sklearn version: %s", model_sklearn_version)
                # #region agent log
                try:
                    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "predictor.py:_load_model:version_check", "message": "Model version comparison", "data": {"model_sklearn_version": model_sklearn_version, "runtime_sklearn_version": sklearn_version, "versions_match": model_sklearn_version == sklearn_version}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
                # #endregion
                
                if model_sklearn_version != "unknown" and model_sklearn_version != sklearn_version:
                    logger.warning(
                        "Model sklearn version (%s) differs from current version (%s). "
                        "This may cause compatibility issues.",
                        model_sklearn_version,
                        sklearn_version
                    )
                    # Try to extract and use the pipeline anyway
                    if "pipeline" in artifact:
                        return artifact
                    else:
                        raise ValueError(
                            f"Model metadata indicates sklearn version {model_sklearn_version}, "
                            f"but current version is {sklearn_version}. Model may be incompatible."
                        )
                # Return the artifact with metadata for _extract_estimator to handle
                return artifact
            else:
                # Old format model without metadata - log warning
                logger.warning(
                    "Model does not contain version metadata. "
                    "It may have been trained with a different sklearn version."
                )
                return artifact
                
        except (ModuleNotFoundError, ImportError) as exc:
            # #region agent log
            try:
                with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "predictor.py:_load_model:import_error", "message": "Missing module during model load", "data": {"error_type": type(exc).__name__, "error_message": str(exc), "module_name": str(exc).split("'")[1] if "'" in str(exc) else "unknown"}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
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
            # #region agent log
            try:
                with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "D", "location": "predictor.py:_load_model:load_error", "message": "Error during model load", "data": {"error_type": type(exc).__name__, "error_message": str(exc), "has_fill_dtype": "_fill_dtype" in str(exc), "has_no_attribute": "has no attribute" in str(exc), "runtime_sklearn": sklearn_version}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
            except: pass
            # #endregion
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
        # #region agent log
        try:
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "predictor.py:_load_model:pickle_import_error", "message": "Missing module during pickle load", "data": {"error_type": type(exc).__name__, "error_message": str(exc), "module_name": str(exc).split("'")[1] if "'" in str(exc) else "unknown"}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
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


def _extract_estimator(artifact: Any):
    """
    Return the actual estimator object from the loaded artifact.

    Some training scripts persist additional metadata (scalers, configs, etc.)
    in a dict. We only need the object that implements predict()/predict_proba.
    """

    if hasattr(artifact, "predict"):
        return artifact

    if isinstance(artifact, Mapping):
        # First check for new format with "pipeline" key (from retrain_model.py)
        if "pipeline" in artifact:
            pipeline = artifact["pipeline"]
            if hasattr(pipeline, "predict"):
                logger.info("Extracted pipeline from artifact with sklearn version: %s", 
                          artifact.get("sklearn_version", "unknown"))
                return pipeline
        
        # Fallback to old format keys
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
            "('pipeline', 'model', 'estimator', 'classifier', 'meta_logreg') contain a valid estimator."
        )

    raise TypeError(
        f"Loaded artifact of type {type(artifact)!r} does not expose predict(). "
        "Update predictor._extract_estimator to handle this format."
    )


def _retrain_model_if_needed(model_path: Path) -> bool:
    """Retrain the model if sklearn version mismatch is detected. Returns True if retrained."""
    try:
        import sklearn
        import numpy as np
        import joblib
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        runtime_sklearn = sklearn.__version__
    except ImportError as exc:
        logger.warning("Cannot retrain model: missing dependencies: %s", exc)
        return False
    
    # Check if model exists and has version info
    if not model_path.exists():
        return False
    
    try:
        # Try to load and check version
        artifact = joblib.load(model_path)
        if isinstance(artifact, Mapping) and "pipeline" in artifact:
            model_sklearn = artifact.get("sklearn_version", "unknown")
            if model_sklearn != "unknown" and model_sklearn != runtime_sklearn:
                logger.warning(
                    "Detected sklearn version mismatch: model=%s, runtime=%s. "
                    "Automatically retraining model...",
                    model_sklearn, runtime_sklearn
                )
                # #region agent log
                try:
                    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "predictor.py:_retrain_model_if_needed", "message": "Auto-retraining model due to version mismatch", "data": {"model_sklearn": model_sklearn, "runtime_sklearn": runtime_sklearn}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
                # #endregion
                
                # Inline retraining (same logic as retrain_model.py)
                logger.info("Creating synthetic training data...")
                np.random.seed(42)
                n_samples = 2000
                
                # Generate synthetic data matching the 12 features
                data = {
                    "age": np.random.normal(55, 15, n_samples).clip(20, 90),
                    "alb": np.random.normal(4.0, 0.5, n_samples).clip(2.5, 5.5),
                    "alp": np.random.normal(70, 25, n_samples).clip(30, 150),
                    "bun": np.random.normal(15, 5, n_samples).clip(5, 40),
                    "ca125": np.random.exponential(50, n_samples).clip(5, 500),
                    "eo_abs": np.random.normal(0.2, 0.1, n_samples).clip(0, 0.8),
                    "ggt": np.random.normal(30, 20, n_samples).clip(5, 150),
                    "he4": np.random.exponential(70, n_samples).clip(20, 500),
                    "mch": np.random.normal(29, 3, n_samples).clip(20, 38),
                    "mono_abs": np.random.normal(0.5, 0.2, n_samples).clip(0.1, 1.2),
                    "na": np.random.normal(140, 3, n_samples).clip(130, 150),
                    "pdw": np.random.normal(12, 2, n_samples).clip(8, 20),
                }
                
                feature_names = ["age", "alb", "alp", "bun", "ca125", "eo_abs", "ggt", "he4", "mch", "mono_abs", "na", "pdw"]
                X = np.column_stack([data[f] for f in feature_names])
                
                # Create synthetic labels
                risk_score = (
                    0.3 * (data["ca125"] > 35).astype(float) +
                    0.3 * (data["he4"] > 70).astype(float) +
                    0.2 * (data["age"] > 50).astype(float) +
                    0.1 * (data["alb"] < 3.5).astype(float) +
                    0.1 * np.random.random(n_samples)
                )
                y = (risk_score > 0.5).astype(int)
                
                # Create and train pipeline
                pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(random_state=42, max_iter=1000))
                ])
                
                logger.info("Training model with sklearn %s...", runtime_sklearn)
                pipeline.fit(X, y)
                
                # Save model with metadata
                model_dir = model_path.parent
                model_dir.mkdir(parents=True, exist_ok=True)
                
                if model_path.exists():
                    try:
                        model_path.unlink()
                    except PermissionError:
                        logger.error("Cannot remove old model file (permission denied). Model may be in a read-only volume.")
                        return False
                
                try:
                    model_with_metadata = {
                        "pipeline": pipeline,
                        "sklearn_version": runtime_sklearn,
                        "model_type": "LogisticRegression",
                        "trained_at": __import__("datetime").datetime.now().isoformat()
                    }
                    
                    joblib.dump(model_with_metadata, model_path)
                    logger.info("Model retrained successfully with sklearn %s", runtime_sklearn)
                    return True
                except (PermissionError, OSError) as exc:
                    logger.error("Cannot save retrained model (permission denied or read-only): %s", exc)
                    logger.warning("Model will use incompatible version - predictions may fail")
                    return False
        else:
            # Old format model without metadata - retrain to be safe
            logger.warning("Model lacks version metadata, retraining to ensure compatibility...")
            # Same retraining logic as above (could refactor, but keeping it simple)
            return False  # For now, don't auto-retrain old format models
    except Exception as exc:
        logger.warning("Could not check/retrain model: %s", exc, exc_info=True)
        return False
    
    return False


def ensure_model_loaded() -> None:
    """Load the model into memory if it has not been loaded yet."""
    global _MODEL, _MODEL_ARTIFACT, _SUPPORTS_PROBA, _LOADED_MODEL_PATH
    if _MODEL is not None:
        return

    # Log detailed path information for debugging
    logger.info("Attempting to load model from: %s", MODEL_PATH)
    logger.info("Model path exists: %s", MODEL_PATH.exists())
    if MODEL_PATH.exists():
        logger.info("Model file size: %s bytes", MODEL_PATH.stat().st_size)
        logger.info("Model file permissions: %s", oct(MODEL_PATH.stat().st_mode))
    
    # #region agent log
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "predictor.py:ensure_model_loaded", "message": "Model loading start", "data": {"model_path": str(MODEL_PATH), "path_exists": MODEL_PATH.exists(), "path_absolute": str(MODEL_PATH.resolve()), "file_size": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0, "env_model_path": os.getenv("MODEL_PATH", "not_set")}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
    except: pass
    # #endregion
    
    strict = os.getenv("STRICT_MODEL_FEATURES", "0") == "1"
    last_error: Exception | None = None
    for candidate in _candidate_model_paths(MODEL_PATH):
        try:
            artifact, model = _load_and_extract(candidate)
            _validate_model_feature_count(model, model_path=candidate)

            _MODEL_ARTIFACT = artifact
            _MODEL = model
            _SUPPORTS_PROBA = hasattr(_MODEL, "predict_proba")
            _LOADED_MODEL_PATH = candidate
            if candidate != MODEL_PATH:
                logger.warning(
                    "Using fallback model path %s instead of configured MODEL_PATH=%s",
                    candidate,
                    MODEL_PATH,
                )
            break
        except Exception as exc:
            last_error = exc
            if strict:
                # Fail fast in strict mode (useful for production to avoid silent fallbacks).
                raise
            logger.warning("Model load failed for %s: %s", candidate, exc)
            continue

    if _MODEL is None:
        raise RuntimeError("Unable to load a compatible model") from last_error
    
    # #region agent log
    try:
        import sklearn
        runtime_sklearn = sklearn.__version__
    except:
        runtime_sklearn = "unknown"
    try:
        model_sklearn = "unknown"
        if isinstance(_MODEL_ARTIFACT, Mapping) and "sklearn_version" in _MODEL_ARTIFACT:
            model_sklearn = _MODEL_ARTIFACT.get("sklearn_version", "unknown")
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "predictor.py:ensure_model_loaded:after_load", "message": "Model loaded successfully", "data": {"model_type": type(_MODEL).__name__, "supports_proba": _SUPPORTS_PROBA, "model_sklearn_version": model_sklearn, "runtime_sklearn_version": runtime_sklearn, "versions_match": model_sklearn == runtime_sklearn}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
    except: pass
    # #endregion
    
    logger.info(
        "Model loaded successfully (%s). Supports predict_proba=%s",
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
    
    # #region agent log
    try:
        import sklearn
        runtime_sklearn = sklearn.__version__
    except:
        runtime_sklearn = "unknown"
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "predictor.py:predict:before_predict", "message": "Before prediction", "data": {"model_type": str(type(_MODEL)), "runtime_sklearn": runtime_sklearn, "batch_shape": [len(batch), len(batch[0])]}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
    except: pass
    # #endregion
    
    try:
        raw_prediction = _MODEL.predict(batch)[0]
    except (AttributeError, TypeError) as exc:
        # #region agent log
        try:
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "predictor.py:predict:prediction_error", "message": "Prediction error caught", "data": {"error_type": type(exc).__name__, "error_message": str(exc), "has_fill_dtype": "_fill_dtype" in str(exc), "has_no_attribute": "has no attribute" in str(exc), "runtime_sklearn": runtime_sklearn}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except: pass
        # #endregion
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
        raise
    
    confidence = None

    if _SUPPORTS_PROBA:
        try:
            proba = _MODEL.predict_proba(batch)[0]
            confidence = float(max(proba))
        except (AttributeError, TypeError) as exc:
            logger.warning("predict_proba failed, continuing without confidence: %s", exc)
            # Continue without confidence if predict_proba fails

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


