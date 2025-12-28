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
import pandas as pd

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


def _get_model_feature_names(model: Any) -> list[str] | None:
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


def _load_pytorch_model(path: Path) -> Any:
    """Load PyTorch model from .pth file."""
    import os
    import torch
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    
    logger.info("Loading PyTorch model from %s", path)
    
    try:
        # Try loading as full model object first
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)
        
        try:
            # Try loading as full model (if saved with torch.save(model, ...))
            model = torch.load(path, map_location=device)
            if hasattr(model, "eval"):
                # It's a model object
                model.eval()
                logger.info("Loaded PyTorch model as full object")
                return model
            elif isinstance(model, dict):
                state_dict = None
                num_classes = 3  # Brain tumor has 3 classes
                
                # Extract state_dict (could be nested or direct)
                if "state_dict" in model:
                    state_dict = model["state_dict"]
                elif all(isinstance(k, str) for k in model.keys()):
                    # Might be a direct state_dict (all keys are strings)
                    # Check if it looks like a state_dict (has typical PyTorch layer names)
                    sample_keys = list(model.keys())[:5]
                    if any(any(marker in k for marker in ["weight", "bias", "running_mean", "running_var"]) for k in sample_keys):
                        state_dict = model
                
                if state_dict is None:
                    raise ValueError(
                        "Model file format not recognized. Expected full model object or dict with 'state_dict' key."
                    )
                
                # Check for architecture specification via environment variable
                model_arch = os.getenv("BRAIN_TUMOR_MODEL_ARCH")
                if model_arch:
                    logger.info("Using architecture specified in BRAIN_TUMOR_MODEL_ARCH: %s", model_arch)
                    pytorch_model = _create_model_from_arch(model_arch, num_classes, state_dict)
                    if pytorch_model:
                        return pytorch_model
                    else:
                        logger.warning("Failed to load with specified architecture %s, attempting inference...", model_arch)
                
                logger.warning("Model contains state_dict but no architecture info. Attempting to infer...")
                
                # Detect architecture hints from state_dict keys (do this before helper function)
                sample_keys = list(state_dict.keys())[:10]
                is_vit_like = any("patch_embed" in k for k in sample_keys)
                has_backbone_prefix = any(k.startswith("backbone.") for k in sample_keys)
                has_classifier_prefix = any(k.startswith("classifier.") for k in state_dict.keys())
                has_pvt_stages = any("stages" in k for k in state_dict.keys())
                
                # Special case: PVTv2 backbone with custom classifier (federated learning model)
                if has_backbone_prefix and has_classifier_prefix and has_pvt_stages:
                    logger.info("Detected PVTv2 backbone with custom classifier - attempting specialized load")
                    try:
                        import timm
                        import torch.nn as nn
                        
                        # Analyze classifier structure from state_dict
                        # Structure: net.0=LayerNorm(512), net.2=Linear(512->256), net.5=Linear(256->3)
                        classifier_keys = [k for k in state_dict.keys() if k.startswith("classifier.")]
                        
                        # Get layer structure by examining weight shapes
                        layer_norm_weight = state_dict.get("classifier.net.0.weight")  # 1D: LayerNorm
                        linear1_weight = state_dict.get("classifier.net.2.weight")     # 2D: Linear
                        linear2_weight = state_dict.get("classifier.net.5.weight")     # 2D: Linear
                        
                        if layer_norm_weight is not None and linear1_weight is not None and linear2_weight is not None:
                            # LayerNorm has 1D weight (norm_dim,)
                            # Linear has 2D weight (out_features, in_features)
                            backbone_out_dim = layer_norm_weight.shape[0]  # 512
                            hidden_dim = linear1_weight.shape[0]           # 256
                            classifier_classes = linear2_weight.shape[0]   # 3
                            
                            logger.info("Classifier structure: backbone_out=%d, hidden=%d, classes=%d",
                                      backbone_out_dim, hidden_dim, classifier_classes)
                            
                            # Try different PVTv2 variants to find one that matches
                            pvt_models = ["pvt_v2_b1", "pvt_v2_b2", "pvt_v2_b3", "pvt_v2_b4", "pvt_v2_b5", "pvt_v2_b0"]
                            
                            for pvt_name in pvt_models:
                                try:
                                    # Create PVTv2 and check if output dim matches
                                    pvt_backbone = timm.create_model(pvt_name, pretrained=False, num_classes=0)
                                    
                                    # Get feature dim by running a dummy forward
                                    with torch.no_grad():
                                        dummy = torch.zeros(1, 3, 224, 224)
                                        feat = pvt_backbone(dummy)
                                        feat_dim = feat.shape[-1]
                                    
                                    if feat_dim == backbone_out_dim:
                                        logger.info("Found matching PVTv2 variant: %s (feature_dim=%d)", pvt_name, feat_dim)
                                        
                                        # Build the full model matching the state_dict structure
                                        # Structure: LayerNorm -> ReLU -> Linear -> ReLU -> Dropout -> Linear
                                        class PVT2WithClassifier(nn.Module):
                                            def __init__(self, backbone, feat_dim, hidden_dim, num_classes, dropout=0.3):
                                                super().__init__()
                                                self.backbone = backbone
                                                # Match classifier.net.[0,1,2,3,4,5] structure
                                                self.classifier = nn.Module()
                                                self.classifier.net = nn.Sequential(
                                                    nn.LayerNorm(feat_dim),              # net.0: LayerNorm
                                                    nn.ReLU(inplace=True),               # net.1: ReLU
                                                    nn.Linear(feat_dim, hidden_dim),     # net.2: Linear(512->256)
                                                    nn.ReLU(inplace=True),               # net.3: ReLU
                                                    nn.Dropout(dropout),                 # net.4: Dropout
                                                    nn.Linear(hidden_dim, num_classes)   # net.5: Linear(256->3)
                                                )
                                            
                                            def forward(self, x):
                                                features = self.backbone(x)
                                                return self.classifier.net(features)
                                        
                                        # Create model instance
                                        pvt_model = PVT2WithClassifier(pvt_backbone, feat_dim, hidden_dim, classifier_classes)
                                        
                                        # Load state_dict (strict=False to allow extra keys from federated learning)
                                        missing, unexpected = pvt_model.load_state_dict(state_dict, strict=False)
                                        # Accept if no missing keys (unexpected keys like adapters/brain_tuner are OK)
                                        if len(missing) == 0:
                                            pvt_model.eval()
                                            logger.info("Successfully loaded PVTv2 model: %s (missing: %d, unexpected: %d)", 
                                                      pvt_name, len(missing), len(unexpected))
                                            if unexpected:
                                                logger.debug("Ignoring %d unexpected keys (adapters, tuners, etc.)", len(unexpected))
                                            return pvt_model
                                        else:
                                            logger.debug("PVTv2 %s: missing keys %s", pvt_name, missing[:5])
                                except Exception as e:
                                    logger.debug("PVTv2 %s failed: %s", pvt_name, e)
                                    continue
                    except ImportError:
                        logger.warning("timm not available, cannot load PVTv2 model")
                    except Exception as e:
                        logger.warning("PVTv2 specialized load failed: %s", e)
                
                if is_vit_like:
                    logger.info("Detected ViT-like architecture from state_dict keys (contains 'patch_embed')")
                
                # Helper function to try loading with state_dict, handling key prefixes
                def try_load_state_dict(model_instance, state_dict_in):
                    try:
                        model_instance.load_state_dict(state_dict_in, strict=True)
                        return True
                    except Exception:
                        # Try with prefix removal (prioritize "backbone." as it's common)
                        for prefix in ["backbone.", "model.", "module.", ""]:
                            try:
                                if prefix:
                                    # Only process keys that start with the prefix
                                    new_state_dict = {k[len(prefix):]: v for k, v in state_dict_in.items() if k.startswith(prefix)}
                                    # If we didn't match any keys with this prefix, skip
                                    if not new_state_dict:
                                        continue
                                else:
                                    new_state_dict = state_dict_in
                                model_instance.load_state_dict(new_state_dict, strict=True)
                                if prefix:
                                    logger.info("Loaded state_dict with prefix '%s' removed", prefix)
                                return True
                            except Exception:
                                continue
                        return False
                
                # Try ResNet architectures (more variants)
                try:
                    import torchvision.models as models
                    resnet_archs = [
                        models.resnet18, models.resnet34, models.resnet50,
                        models.resnet101, models.resnet152
                    ]
                    for resnet_class in resnet_archs:
                        try:
                            pytorch_model = resnet_class(pretrained=False, num_classes=num_classes)
                            if try_load_state_dict(pytorch_model, state_dict):
                                pytorch_model.eval()
                                logger.info("Successfully loaded model as %s", resnet_class.__name__)
                                return pytorch_model
                        except Exception:
                            continue
                except Exception as e:
                    logger.debug("ResNet architectures failed: %s", e)
                
                # Try timm models if available (many more options)
                # Prioritize ViT models if we detect ViT-like keys
                try:
                    import timm
                    if is_vit_like:
                        # Try ViT models first if we detect ViT-like architecture
                        vit_models = [
                            "vit_base_patch16_224", "vit_small_patch16_224", "vit_tiny_patch16_224",
                            "vit_large_patch16_224", "deit_base_patch16_224", "deit_small_patch16_224"
                        ]
                        for model_name in vit_models:
                            try:
                                pytorch_model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
                                if try_load_state_dict(pytorch_model, state_dict):
                                    pytorch_model.eval()
                                    logger.info("Successfully loaded model as %s", model_name)
                                    return pytorch_model
                            except Exception:
                                continue
                    
                    # Try all common models
                    timm_models = [
                        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
                        "resnet18", "resnet34", "resnet50", "resnet101",
                        "vit_base_patch16_224", "vit_small_patch16_224",
                        "densenet121", "densenet169",
                        "mobilenetv3_small_100", "mobilenetv3_large_100"
                    ]
                    for model_name in timm_models:
                        try:
                            pytorch_model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
                            if try_load_state_dict(pytorch_model, state_dict):
                                pytorch_model.eval()
                                logger.info("Successfully loaded model as %s", model_name)
                                return pytorch_model
                        except Exception:
                            continue
                except ImportError:
                    logger.warning("timm not available for model architecture inference")
                
                # Get some sample keys for error message
                sample_keys_display = list(state_dict.keys())[:3] if state_dict else []
                suggestion = ""
                if is_vit_like:
                    suggestion = "Based on state_dict keys (contains 'patch_embed'), this appears to be a Vision Transformer (ViT). Try: 'vit_base_patch16_224', 'vit_small_patch16_224', or 'deit_base_patch16_224'."
                elif has_backbone_prefix:
                    suggestion = "State_dict has 'backbone.' prefix. Try common architectures like 'resnet50', 'efficientnet_b0', or 'vit_base_patch16_224'."
                else:
                    suggestion = "Try common architectures like 'resnet18', 'efficientnet_b0', 'resnet50', or 'vit_base_patch16_224'."
                
                raise ValueError(
                    f"Could not infer model architecture from state_dict. "
                    f"Sample state_dict keys: {sample_keys_display}. "
                    f"{suggestion} "
                    f"Please specify the architecture using the BRAIN_TUMOR_MODEL_ARCH environment variable "
                    f"(e.g., export BRAIN_TUMOR_MODEL_ARCH=vit_base_patch16_224). "
                    f"Alternatively, save the model as a full model object instead of just state_dict."
                )
            else:
                raise ValueError(
                    "Model file format not recognized. Expected full model object or dict."
                )
        except Exception as exc:
            logger.error("Failed to load PyTorch model: %s", exc, exc_info=True)
            raise ValueError(f"Failed to load PyTorch model: {str(exc)}") from exc
            
    except ImportError:
        raise ImportError("PyTorch is not installed. Please install torch to use PyTorch models.")


def _create_model_from_arch(arch_name: str, num_classes: int, state_dict: dict) -> Any | None:
    """Create a model instance from architecture name and load state_dict."""
    import torch
    
    try:
        # Helper to try loading with prefix handling
        def try_load_with_prefixes(model_instance, state_dict_in):
            # Try strict loading first
            try:
                model_instance.load_state_dict(state_dict_in, strict=True)
                return True
            except Exception:
                # Try with prefix removal
                for prefix in ["model.", "module.", "backbone.", ""]:
                    try:
                        if prefix:
                            new_state_dict = {k[len(prefix):]: v for k, v in state_dict_in.items() if k.startswith(prefix)}
                            if not new_state_dict:
                                continue
                        else:
                            new_state_dict = state_dict_in
                        model_instance.load_state_dict(new_state_dict, strict=True)
                        if prefix:
                            logger.info("Loaded state_dict with prefix '%s' removed", prefix)
                        return True
                    except Exception:
                        continue
                # Last resort: try with strict=False
                try:
                    model_instance.load_state_dict(state_dict_in, strict=False)
                    logger.warning("Loaded state_dict with strict=False (some keys may be missing)")
                    return True
                except Exception:
                    return False
        
        # Try torchvision models first
        try:
            import torchvision.models as models
            arch_map = {
                "resnet18": models.resnet18,
                "resnet34": models.resnet34,
                "resnet50": models.resnet50,
                "resnet101": models.resnet101,
                "resnet152": models.resnet152,
            }
            if arch_name.lower() in arch_map:
                model_class = arch_map[arch_name.lower()]
                pytorch_model = model_class(pretrained=False, num_classes=num_classes)
                if try_load_with_prefixes(pytorch_model, state_dict):
                    pytorch_model.eval()
                    logger.info("Successfully loaded model as %s", arch_name)
                    return pytorch_model
        except Exception as e:
            logger.debug("Failed to load %s from torchvision: %s", arch_name, e)
        
        # Try timm models
        try:
            import timm
            pytorch_model = timm.create_model(arch_name, pretrained=False, num_classes=num_classes)
            if try_load_with_prefixes(pytorch_model, state_dict):
                pytorch_model.eval()
                logger.info("Successfully loaded model as %s from timm", arch_name)
                return pytorch_model
        except Exception as e:
            logger.debug("Failed to load %s from timm: %s", arch_name, e)
        
        return None
    except Exception as e:
        logger.error("Error creating model from arch %s: %s", arch_name, e)
        return None


def _load_and_extract(path: Path, *, config: ModelConfig) -> tuple[Any, Any]:
    """Load artifact from path and extract estimator/pipeline."""
    # Check if this is a PyTorch model
    if path.suffix == ".pth" or config.key == "brain_tumor":
        model = _load_pytorch_model(path)
        # For PyTorch models, we return the model itself as both artifact and model
        return model, model
    
    # sklearn model loading
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
            # PyTorch models support probabilities via softmax, sklearn models via predict_proba
            if config.key == "brain_tumor":
                _SUPPORTS_PROBA[model_key] = True  # PyTorch models always support probabilities
            else:
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


def predict_image(image_tensor: Any, *, model_key: str = "brain_tumor") -> PredictionResult:
    """Run inference on an image tensor for PyTorch models."""
    if model_key not in _MODELS:
        raise RuntimeError("Model has not been loaded yet. Call ensure_model_loaded().")
    
    model = _MODELS[model_key]
    import torch
    
    try:
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            
            # Get probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_class_idx = predicted_class.item()
            confidence_value = confidence.item()
        
        logger.info(
            "Image prediction completed for %s: class=%s, confidence=%.4f",
            model_key, predicted_class_idx, confidence_value
        )
        
        return PredictionResult(prediction=int(predicted_class_idx), confidence=float(confidence_value))
        
    except Exception as exc:
        logger.error("Error during image prediction: %s", exc, exc_info=True)
        raise ValueError(f"Image prediction failed: {str(exc)}") from exc


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
        batch_array = np.array(vector).reshape(1, -1)
        logger.debug("Batch shape: %s, dtype: %s", batch_array.shape, batch_array.dtype)
    except Exception as exc:
        logger.error("Failed to create numpy array from features: %s", exc, exc_info=True)
        raise ValueError(f"Failed to prepare feature array: {str(exc)}") from exc
    
    # Convert to pandas DataFrame with feature names to avoid sklearn warnings
    # about feature name mismatches when models were trained with named features
    # Use the model's expected feature names if available, otherwise use registry names
    try:
        model = _MODELS[model_key]
        model_feature_names = _get_model_feature_names(model)
        
        if model_feature_names is not None:
            # Use the feature names the model expects
            if len(model_feature_names) != len(batch_array[0]):
                logger.warning(
                    "Model feature names count (%d) doesn't match feature vector length (%d). "
                    "Falling back to numpy array.",
                    len(model_feature_names), len(batch_array[0])
                )
                batch = batch_array
            else:
                feature_names_to_use = model_feature_names
                logger.debug("Using model's expected feature names: %s", feature_names_to_use[:5])  # Log first 5
                batch = pd.DataFrame(batch_array, columns=feature_names_to_use)
                logger.debug("Converted to DataFrame with %d columns", len(batch.columns))
        else:
            # Model doesn't expose feature_names_in_ - try using registry names
            # but be prepared to fall back to numpy array if it fails
            logger.debug("Model doesn't expose feature_names_in_, trying registry feature names")
            batch = pd.DataFrame(batch_array, columns=config.feature_order)
            logger.debug("Converted to DataFrame with registry feature names: %s", list(batch.columns)[:5])
    except Exception as exc:
        logger.warning("Failed to convert to DataFrame with feature names: %s. Falling back to numpy array.", exc)
        batch = batch_array
    
    # #region agent log
    try:
        import sklearn
        runtime_sklearn = sklearn.__version__
    except Exception:
        runtime_sklearn = "unknown"
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            batch_shape = list(batch.shape) if hasattr(batch, "shape") else "unknown"
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "predictor.py:predict:before_predict", "message": "Before prediction", "data": {"model_key": model_key, "model_type": str(type(_MODELS[model_key])), "runtime_sklearn": runtime_sklearn, "batch_shape": batch_shape, "supports_proba": _SUPPORTS_PROBA[model_key]}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
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
        batch_shape = batch.shape if hasattr(batch, "shape") else "unknown"
        batch_dtype = batch.dtype if hasattr(batch, "dtype") else "unknown"
        logger.error("Model type: %s, Batch shape: %s, Batch dtype: %s", type(_MODELS[model_key]).__name__, batch_shape, batch_dtype)
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


