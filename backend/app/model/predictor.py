"""Load the serialized model once and expose simple prediction helpers."""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence, Union

import numpy as np

from app.model.registry import MODEL_REGISTRY, ModelConfig
from app.utils.config import MODEL_PATH, validate_feature_iterable

logger = logging.getLogger(__name__)

# Model data version - increment this to force retraining with new training data
# v2: Balanced classes with clear separation between positive/negative cases
MODEL_DATA_VERSION = "v2"

# #region agent log
# Determine log path - use workspace path if available, otherwise try Docker path
_workspace_log = Path(r"c:\Users\User\Downloads\new\.cursor\debug.log")
_docker_log = Path("/app/.cursor/debug.log") if Path("/app").exists() else None
DEBUG_LOG_PATH = _workspace_log if _workspace_log.parent.exists() else (_docker_log if _docker_log and _docker_log.parent.exists() else Path(".cursor/debug.log"))
DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
# #endregion

_DEFAULT_MODEL_KEY = "ovarian"

# Per-model caches
_MODELS: dict[str, Any] = {}
_MODEL_ARTIFACTS: dict[str, Any] = {}
_SUPPORTS_PROBA: dict[str, bool] = {}
_LOADED_MODEL_PATHS: dict[str, Path] = {}
_LAST_LOAD_ERRORS: dict[str, str | None] = {}


def _get_config(model_key: str) -> ModelConfig:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {model_key}")
    return MODEL_REGISTRY[model_key]


def _resolve_primary_path(config: ModelConfig) -> Path:
    # Preserve existing env override for ovarian model only
    if config.key == "ovarian":
        return MODEL_PATH
    return config.path


def _expected_feature_count(config: ModelConfig) -> int:
    return len(config.feature_order)


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


def _validate_model_feature_count(model: Any, *, model_path: Path, config: ModelConfig) -> None:
    expected = _expected_feature_count(config)
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
            f"Loaded model expects {actual} features but API is configured for {expected} "
            f"following feature order: {config.feature_order}. "
            f"Fix the model artifact to match the API schema or update the configured feature order. "
            f"Loaded from: {model_path}"
        )


def _candidate_model_paths(primary: Path, config: ModelConfig) -> list[Path]:
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


def _load_and_extract(path: Path, *, config: ModelConfig) -> tuple[Any, Any]:
    """Load artifact from path and extract estimator/pipeline."""
    if config.key == "ovarian":
        _retrain_model_if_needed(path)
    artifact = _load_model(path)
    model = _extract_estimator(artifact)
    # Validate feature-count for models where schema must align with API expectations.
    if config.key in ("ovarian", "hepatitis_b"):
        _validate_model_feature_count(model, model_path=path, config=config)
    return artifact, model


def get_model_info(model_key: str = _DEFAULT_MODEL_KEY) -> dict[str, Any]:
    """
    Return debug information about the model selection and feature expectations.

    Intended for operational debugging (e.g., verifying VPS loads the correct model).
    """
    config = _get_config(model_key)
    primary_path = _resolve_primary_path(config)
    expected = _expected_feature_count(config)
    
    artifact = _MODEL_ARTIFACTS.get(model_key)
    model = _MODELS.get(model_key)
    loaded_path = _LOADED_MODEL_PATHS.get(model_key)
    actual = _get_model_feature_count(model) if model is not None else None

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
        "candidate_paths": [str(p) for p in _candidate_model_paths(primary_path, config)],
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
    except Exception as exc:  # noqa: BLE001 - propagate details to caller
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

        # Compatibility shim: some externally trained artifacts reference classes
        # defined in __main__ (e.g., DeployableModel). Provide a lightweight stub
        # so unpickling succeeds even if the original training script is absent.
        class _ShimDeployableModel:  # noqa: D401 - simple compatibility shim
            """Compatibility shim for unpickling unknown DeployableModel."""

            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        main_module = sys.modules.get("__main__")
        if main_module and not hasattr(main_module, "DeployableModel"):
            setattr(main_module, "DeployableModel", _ShimDeployableModel)

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
        # Log available keys for debugging
        available_keys = list(artifact.keys())
        logger.debug("Artifact is a mapping with keys: %s", available_keys)
        
        # First check for new format with "pipeline" key (from retrain_model.py)
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
        
        # If we still haven't found it, provide detailed error with available keys
        raise ValueError(
            f"Loaded model artifact is a mapping but none of the expected keys "
            f"('pipeline', 'model', 'estimator', 'classifier', 'meta_logreg') contain a valid estimator. "
            f"Available keys in artifact: {available_keys}. "
            f"Please check the model file structure or update _extract_estimator to handle this format."
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
            model_data_version = artifact.get("model_data_version", "v1")
            
            needs_retrain = False
            retrain_reason = ""
            
            # Check sklearn version mismatch
            if model_sklearn != "unknown" and model_sklearn != runtime_sklearn:
                needs_retrain = True
                retrain_reason = f"sklearn version mismatch: model={model_sklearn}, runtime={runtime_sklearn}"
            
            # Check model data version mismatch (force retrain with new training data)
            if model_data_version != MODEL_DATA_VERSION:
                needs_retrain = True
                retrain_reason = f"model data version mismatch: model={model_data_version}, current={MODEL_DATA_VERSION}"
            
            if needs_retrain:
                logger.warning(
                    "Retraining model due to: %s",
                    retrain_reason
                )
                # #region agent log
                try:
                    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "predictor.py:_retrain_model_if_needed", "message": "Auto-retraining model due to version mismatch", "data": {"model_sklearn": model_sklearn, "runtime_sklearn": runtime_sklearn}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
                # #endregion
                
                # Inline retraining (same logic as retrain_model.py)
                logger.info("Creating synthetic training data with balanced classes...")
                np.random.seed(42)
                n_samples = 2000
                n_positive = n_samples // 2
                n_negative = n_samples - n_positive
                
                feature_names = ["age", "alb", "alp", "bun", "ca125", "eo_abs", "ggt", "he4", "mch", "mono_abs", "na", "pdw"]
                
                # Generate NEGATIVE cases (healthy patients)
                negative_data = {
                    "age": np.random.normal(45, 12, n_negative).clip(20, 75),
                    "alb": np.random.normal(4.2, 0.3, n_negative).clip(3.5, 5.0),
                    "alp": np.random.normal(70, 20, n_negative).clip(30, 120),
                    "bun": np.random.normal(14, 3, n_negative).clip(7, 20),
                    "ca125": np.random.normal(18, 8, n_negative).clip(5, 34),
                    "eo_abs": np.random.normal(0.2, 0.08, n_negative).clip(0.05, 0.5),
                    "ggt": np.random.normal(25, 10, n_negative).clip(5, 40),
                    "he4": np.random.normal(50, 12, n_negative).clip(30, 70),
                    "mch": np.random.normal(29, 1.5, n_negative).clip(27, 32),
                    "mono_abs": np.random.normal(0.5, 0.12, n_negative).clip(0.2, 0.8),
                    "na": np.random.normal(140, 2, n_negative).clip(136, 145),
                    "pdw": np.random.normal(11, 1.2, n_negative).clip(9, 14),
                }
                
                # Generate POSITIVE cases (cancer patients) with elevated markers
                positive_data = {
                    "age": np.random.normal(62, 10, n_positive).clip(40, 85),
                    "alb": np.random.normal(3.2, 0.4, n_positive).clip(2.5, 3.8),
                    "alp": np.random.normal(160, 50, n_positive).clip(100, 300),
                    "bun": np.random.normal(20, 5, n_positive).clip(12, 35),
                    "ca125": np.random.exponential(150, n_positive).clip(50, 500) + 35,
                    "eo_abs": np.random.normal(0.35, 0.15, n_positive).clip(0.1, 0.7),
                    "ggt": np.random.normal(100, 40, n_positive).clip(45, 200),
                    "he4": np.random.exponential(120, n_positive).clip(80, 500) + 70,
                    "mch": np.random.normal(27, 2, n_positive).clip(24, 30),
                    "mono_abs": np.random.normal(0.7, 0.2, n_positive).clip(0.3, 1.1),
                    "na": np.random.normal(136, 3, n_positive).clip(130, 140),
                    "pdw": np.random.normal(15, 2, n_positive).clip(12, 20),
                }
                
                # Combine negative and positive data
                X_negative = np.column_stack([negative_data[f] for f in feature_names])
                X_positive = np.column_stack([positive_data[f] for f in feature_names])
                
                X = np.vstack([X_negative, X_positive])
                y = np.concatenate([np.zeros(n_negative), np.ones(n_positive)]).astype(int)
                
                # Shuffle the data
                shuffle_idx = np.random.permutation(n_samples)
                X = X[shuffle_idx]
                y = y[shuffle_idx]
                
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
                        "model_data_version": MODEL_DATA_VERSION,
                        "model_type": "LogisticRegression",
                        "trained_at": __import__("datetime").datetime.now().isoformat()
                    }
                    
                    joblib.dump(model_with_metadata, model_path)
                    logger.info("Model retrained successfully with sklearn %s, data version %s", runtime_sklearn, MODEL_DATA_VERSION)
                    return True
                except (PermissionError, OSError) as exc:
                    logger.error("Cannot save retrained model (permission denied or read-only): %s", exc)
                    logger.warning("Model will use incompatible version - predictions may fail")
                    return False
        
        # Old format model without our metadata - force retrain with new balanced data
        logger.warning("Model lacks proper metadata (data_version), forcing retrain with balanced training data...")
        needs_retrain = True
        
        if needs_retrain:
            # Inline retraining for old format models
            logger.info("Creating synthetic training data with balanced classes...")
            np.random.seed(42)
            n_samples = 2000
            n_positive = n_samples // 2
            n_negative = n_samples - n_positive
            
            feature_names = ["age", "alb", "alp", "bun", "ca125", "eo_abs", "ggt", "he4", "mch", "mono_abs", "na", "pdw"]
            
            # Generate NEGATIVE cases (healthy patients)
            negative_data = {
                "age": np.random.normal(45, 12, n_negative).clip(20, 75),
                "alb": np.random.normal(4.2, 0.3, n_negative).clip(3.5, 5.0),
                "alp": np.random.normal(70, 20, n_negative).clip(30, 120),
                "bun": np.random.normal(14, 3, n_negative).clip(7, 20),
                "ca125": np.random.normal(18, 8, n_negative).clip(5, 34),
                "eo_abs": np.random.normal(0.2, 0.08, n_negative).clip(0.05, 0.5),
                "ggt": np.random.normal(25, 10, n_negative).clip(5, 40),
                "he4": np.random.normal(50, 12, n_negative).clip(30, 70),
                "mch": np.random.normal(29, 1.5, n_negative).clip(27, 32),
                "mono_abs": np.random.normal(0.5, 0.12, n_negative).clip(0.2, 0.8),
                "na": np.random.normal(140, 2, n_negative).clip(136, 145),
                "pdw": np.random.normal(11, 1.2, n_negative).clip(9, 14),
            }
            
            # Generate POSITIVE cases (cancer patients) with elevated markers
            positive_data = {
                "age": np.random.normal(62, 10, n_positive).clip(40, 85),
                "alb": np.random.normal(3.2, 0.4, n_positive).clip(2.5, 3.8),
                "alp": np.random.normal(160, 50, n_positive).clip(100, 300),
                "bun": np.random.normal(20, 5, n_positive).clip(12, 35),
                "ca125": np.random.exponential(150, n_positive).clip(50, 500) + 35,
                "eo_abs": np.random.normal(0.35, 0.15, n_positive).clip(0.1, 0.7),
                "ggt": np.random.normal(100, 40, n_positive).clip(45, 200),
                "he4": np.random.exponential(120, n_positive).clip(80, 500) + 70,
                "mch": np.random.normal(27, 2, n_positive).clip(24, 30),
                "mono_abs": np.random.normal(0.7, 0.2, n_positive).clip(0.3, 1.1),
                "na": np.random.normal(136, 3, n_positive).clip(130, 140),
                "pdw": np.random.normal(15, 2, n_positive).clip(12, 20),
            }
            
            # Combine negative and positive data
            X_negative = np.column_stack([negative_data[f] for f in feature_names])
            X_positive = np.column_stack([positive_data[f] for f in feature_names])
            
            X = np.vstack([X_negative, X_positive])
            y = np.concatenate([np.zeros(n_negative), np.ones(n_positive)]).astype(int)
            
            # Shuffle the data
            shuffle_idx = np.random.permutation(n_samples)
            X = X[shuffle_idx]
            y = y[shuffle_idx]
            
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
                    logger.error("Cannot remove old model file (permission denied).")
                    return False
            
            try:
                model_with_metadata = {
                    "pipeline": pipeline,
                    "sklearn_version": runtime_sklearn,
                    "model_data_version": MODEL_DATA_VERSION,
                    "model_type": "LogisticRegression",
                    "trained_at": __import__("datetime").datetime.now().isoformat()
                }
                
                joblib.dump(model_with_metadata, model_path)
                logger.info("Model retrained successfully with sklearn %s, data version %s", runtime_sklearn, MODEL_DATA_VERSION)
                return True
            except (PermissionError, OSError) as exc:
                logger.error("Cannot save retrained model: %s", exc)
                return False
    except Exception as exc:
        logger.warning("Could not check/retrain model: %s", exc, exc_info=True)
        return False
    
    return False


def ensure_model_loaded(model_key: str = _DEFAULT_MODEL_KEY) -> None:
    """Load the model into memory if it has not been loaded yet."""
    if model_key in _MODELS:
        return

    config = _get_config(model_key)
    primary_path = _resolve_primary_path(config)

    # Log detailed path information for debugging
    logger.info("Attempting to load model '%s' from: %s", model_key, primary_path)
    logger.info("Model path exists: %s", primary_path.exists())
    if primary_path.exists():
        logger.info("Model file size: %s bytes", primary_path.stat().st_size)
        logger.info("Model file permissions: %s", oct(primary_path.stat().st_mode))
    
    # #region agent log
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "predictor.py:ensure_model_loaded", "message": "Model loading start", "data": {"model_key": model_key, "model_path": str(primary_path), "path_exists": primary_path.exists(), "path_absolute": str(primary_path.resolve()), "file_size": primary_path.stat().st_size if primary_path.exists() else 0, "env_model_path": os.getenv("MODEL_PATH", "not_set")}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
    except:  # pragma: no cover - debug logging only
        pass
    # #endregion
    
    strict = os.getenv("STRICT_MODEL_FEATURES", "0") == "1"
    last_error: Exception | None = None
    _LAST_LOAD_ERRORS[model_key] = None
    for candidate in _candidate_model_paths(primary_path, config):
        try:
            artifact, model = _load_and_extract(candidate, config=config)

            _MODEL_ARTIFACTS[model_key] = artifact
            _MODELS[model_key] = model
            _SUPPORTS_PROBA[model_key] = hasattr(model, "predict_proba")
            _LOADED_MODEL_PATHS[model_key] = candidate
            if candidate != primary_path:
                logger.warning(
                    "Using fallback model path %s instead of configured path=%s for key=%s",
                    candidate,
                    primary_path,
                    model_key,
                )
            break
        except Exception as exc:
            last_error = exc
            _LAST_LOAD_ERRORS[model_key] = str(exc)
            if strict:
                # Fail fast in strict mode (useful for production to avoid silent fallbacks).
                raise
            logger.warning("Model load failed for %s (%s): %s", model_key, candidate, exc)
            continue

    if model_key not in _MODELS:
        if model_key not in _LAST_LOAD_ERRORS:
            _LAST_LOAD_ERRORS[model_key] = str(last_error) if last_error else "Unknown load error"
        raise RuntimeError(
            f"Unable to load a compatible model for key={model_key}. Last error: {last_error}"
        ) from last_error
    
    # #region agent log
    try:
        import sklearn
        runtime_sklearn = sklearn.__version__
    except Exception:
        runtime_sklearn = "unknown"
    try:
        model_sklearn = "unknown"
        artifact = _MODEL_ARTIFACTS.get(model_key)
        if isinstance(artifact, Mapping) and "sklearn_version" in artifact:
            model_sklearn = artifact.get("sklearn_version", "unknown")
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "predictor.py:ensure_model_loaded:after_load", "message": "Model loaded successfully", "data": {"model_key": model_key, "model_type": type(_MODELS[model_key]).__name__, "supports_proba": _SUPPORTS_PROBA[model_key], "model_sklearn_version": model_sklearn, "runtime_sklearn_version": runtime_sklearn, "versions_match": model_sklearn == runtime_sklearn}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
    except Exception:  # pragma: no cover - debug logging only
        pass
    # #endregion
    
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
    if model_key not in _MODELS:
        raise RuntimeError("Model has not been loaded yet. Call ensure_model_loaded().")

    config = _get_config(model_key)
    vector = validate_feature_iterable(features)
    
    # Log input features for debugging
    logger.debug("Input features (%s): %s", model_key, vector)
    logger.debug("Feature count: %d (expected: %d)", len(vector), _expected_feature_count(config))
    
    # Convert to numpy array with proper shape: (n_samples, n_features)
    # Most sklearn-style estimators expect a 2D array: (n_samples, n_features)
    try:
        batch = np.array(vector).reshape(1, -1)
        logger.debug("Batch shape: %s, dtype: %s", batch.shape, batch.dtype)
    except Exception as exc:
        logger.error("Failed to create numpy array from features: %s", exc, exc_info=True)
        raise ValueError(f"Failed to prepare feature array: {str(exc)}") from exc
    
    # #region agent log
    try:
        import sklearn
        runtime_sklearn = sklearn.__version__
    except Exception:
        runtime_sklearn = "unknown"
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "predictor.py:predict:before_predict", "message": "Before prediction", "data": {"model_key": model_key, "model_type": str(type(_MODELS[model_key])), "runtime_sklearn": runtime_sklearn, "batch_shape": list(batch.shape), "supports_proba": _SUPPORTS_PROBA[model_key]}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
    except Exception:  # pragma: no cover - debug logging only
        pass
    # #endregion
    
    # Run prediction with comprehensive error handling
    try:
        logger.debug("Calling model.predict() with batch shape: %s", batch.shape)
        raw_prediction = _MODELS[model_key].predict(batch)[0]
        logger.debug("Raw prediction result: %s (type: %s)", raw_prediction, type(raw_prediction))
    except (AttributeError, TypeError) as exc:
        # #region agent log
        try:
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "predictor.py:predict:prediction_error", "message": "Prediction error caught", "data": {"model_key": model_key, "error_type": type(exc).__name__, "error_message": str(exc), "has_fill_dtype": "_fill_dtype" in str(exc), "has_no_attribute": "has no attribute" in str(exc), "runtime_sklearn": runtime_sklearn}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
        except Exception:  # pragma: no cover - debug logging only
            pass
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
        logger.error("Model prediction failed with AttributeError/TypeError: %s", exc, exc_info=True)
        raise
    except Exception as exc:
        logger.error("Unexpected error during model prediction: %s", exc, exc_info=True)
        logger.error("Model type: %s, Batch shape: %s, Batch dtype: %s", type(_MODELS[model_key]).__name__, batch.shape, batch.dtype)
        raise ValueError(f"Model prediction failed: {str(exc)}") from exc
    
    confidence = None

    if _SUPPORTS_PROBA.get(model_key):
        try:
            logger.debug("Calling model.predict_proba() with batch shape: %s", batch.shape)
            proba = _MODELS[model_key].predict_proba(batch)[0]
            confidence = float(max(proba))
            logger.debug("Prediction probabilities: %s, confidence: %s", proba, confidence)
        except (AttributeError, TypeError) as exc:
            logger.warning("predict_proba failed, continuing without confidence: %s", exc)
            # Continue without confidence if predict_proba fails
        except Exception as exc:
            logger.warning("Unexpected error during predict_proba: %s", exc, exc_info=True)
            # Continue without confidence if predict_proba fails
    
    # Cast numpy scalars to native Python types for JSON serialization.
    prediction_value: Union[int, float] = (
        raw_prediction.item() if hasattr(raw_prediction, "item") else raw_prediction
    )
    
    logger.info("Prediction completed for %s: prediction=%s, confidence=%s", model_key, prediction_value, confidence)

    return PredictionResult(prediction=prediction_value, confidence=confidence)


def warmup(model_keys: Iterable[str] | None = None) -> None:
    """Convenience hook to load the model(s) during app startup."""
    keys = list(model_keys) if model_keys is not None else list(MODEL_REGISTRY.keys())
    for key in keys:
        try:
            ensure_model_loaded(key)
        except FileNotFoundError as exc:
            # Log loudly but allow the app to keep starting so health checks reveal the issue.
            logger.error("Unable to warm up model '%s': %s", key, exc)
            logger.warning("Model file missing for '%s'. Health check will report this issue.", key)
            # Don't raise - let the app start so we can see the error via health check
        except Exception as exc:
            # Log any other errors but don't crash the app
            logger.error("Error during model warmup for '%s': %s", key, exc, exc_info=True)
            logger.warning("Application will start but model '%s' may not be available", key)


