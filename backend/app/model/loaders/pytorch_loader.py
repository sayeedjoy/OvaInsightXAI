"""PyTorch model loading logic for brain tumor and other image-based models."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_pytorch_model(path: Path) -> Any:
    """Load PyTorch model from .pth file."""
    import torch
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    
    logger.info("Loading PyTorch model from %s", path)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)
        
        try:
            # Try loading as full model object first
            model = torch.load(path, map_location=device)
            
            # Extract normalization statistics from checkpoint if available
            if isinstance(model, dict):
                train_mean = model.get("train_mean_rgb")
                train_std = model.get("train_std_rgb")
                if train_mean is not None and train_std is not None:
                    try:
                        # Convert to list if tensor
                        if hasattr(train_mean, "tolist"):
                            mean_list = train_mean.tolist()
                        elif isinstance(train_mean, (list, tuple)):
                            mean_list = list(train_mean)
                        else:
                            mean_list = None
                        
                        if hasattr(train_std, "tolist"):
                            std_list = train_std.tolist()
                        elif isinstance(train_std, (list, tuple)):
                            std_list = list(train_std)
                        else:
                            std_list = None
                        
                        if mean_list and std_list and len(mean_list) == 3 and len(std_list) == 3:
                            from app.services import image_preprocessing
                            image_preprocessing.set_dataset_normalization(mean_list, std_list)
                            logger.info(
                                "Loaded normalization stats from checkpoint: mean=%s, std=%s",
                                mean_list, std_list
                            )
                    except Exception as exc:
                        logger.warning("Failed to extract normalization stats from checkpoint: %s", exc)
            
            if hasattr(model, "eval"):
                model.eval()
                logger.info("Loaded PyTorch model as full object")
                return model
            elif isinstance(model, dict):
                state_dict = None
                num_classes = 4  # Brain tumor has 4 classes
                
                # Extract state_dict
                if "state_dict" in model:
                    state_dict = model["state_dict"]
                elif all(isinstance(k, str) for k in model.keys()):
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
                    pytorch_model = create_model_from_arch(model_arch, num_classes, state_dict)
                    if pytorch_model:
                        return pytorch_model
                    else:
                        logger.warning("Failed to load with specified architecture %s, attempting inference...", model_arch)
                
                logger.warning("Model contains state_dict but no architecture info. Attempting to infer...")
                
                # Detect architecture hints from state_dict keys
                sample_keys = list(state_dict.keys())[:10]
                is_vit_like = any("patch_embed" in k for k in sample_keys)
                has_backbone_prefix = any(k.startswith("backbone.") for k in sample_keys)
                has_classifier_prefix = any(k.startswith("classifier.") for k in state_dict.keys())
                has_pvt_stages = any("stages" in k for k in state_dict.keys())
                
                # Special case: PVTv2 backbone with custom classifier
                if has_backbone_prefix and has_classifier_prefix and has_pvt_stages:
                    pytorch_model = _try_load_pvt2_model(state_dict, device)
                    if pytorch_model:
                        return pytorch_model
                
                # Try ResNet architectures
                pytorch_model = _try_load_resnet(state_dict, num_classes)
                if pytorch_model:
                    return pytorch_model
                
                # Try timm models
                pytorch_model = _try_load_timm_model(state_dict, num_classes, is_vit_like)
                if pytorch_model:
                    return pytorch_model
                
                # Provide helpful error message
                sample_keys_display = list(state_dict.keys())[:3]
                suggestion = _get_architecture_suggestion(is_vit_like, has_backbone_prefix)
                
                raise ValueError(
                    f"Could not infer model architecture from state_dict. "
                    f"Sample state_dict keys: {sample_keys_display}. "
                    f"{suggestion} "
                    f"Please specify the architecture using the BRAIN_TUMOR_MODEL_ARCH environment variable."
                )
            else:
                raise ValueError("Model file format not recognized. Expected full model object or dict.")
                
        except Exception as exc:
            logger.error("Failed to load PyTorch model: %s", exc, exc_info=True)
            raise ValueError(f"Failed to load PyTorch model: {str(exc)}") from exc
            
    except ImportError:
        raise ImportError("PyTorch is not installed. Please install torch to use PyTorch models.")


def _try_load_pvt2_model(state_dict: dict, device) -> Any | None:
    """Try to load PVTv2 backbone with custom classifier."""
    import torch
    import torch.nn as nn
    
    try:
        import timm
        
        logger.info("Detected PVTv2 backbone with custom classifier - attempting specialized load")
        
        # Analyze classifier structure from state_dict
        layer_norm_weight = state_dict.get("classifier.net.0.weight")
        linear1_weight = state_dict.get("classifier.net.2.weight")
        linear2_weight = state_dict.get("classifier.net.5.weight")
        
        if layer_norm_weight is None or linear1_weight is None or linear2_weight is None:
            return None
            
        backbone_out_dim = layer_norm_weight.shape[0]
        hidden_dim = linear1_weight.shape[0]
        classifier_classes = linear2_weight.shape[0]
        
        logger.info("Classifier structure: backbone_out=%d, hidden=%d, classes=%d",
                  backbone_out_dim, hidden_dim, classifier_classes)
        
        # Try different PVTv2 variants
        pvt_models = ["pvt_v2_b1", "pvt_v2_b2", "pvt_v2_b3", "pvt_v2_b4", "pvt_v2_b5", "pvt_v2_b0"]
        
        for pvt_name in pvt_models:
            try:
                pvt_backbone = timm.create_model(pvt_name, pretrained=False, num_classes=0)
                
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 224, 224)
                    feat = pvt_backbone(dummy)
                    feat_dim = feat.shape[-1]
                
                if feat_dim == backbone_out_dim:
                    logger.info("Found matching PVTv2 variant: %s (feature_dim=%d)", pvt_name, feat_dim)
                    
                    class PVT2WithClassifier(nn.Module):
                        def __init__(self, backbone, feat_dim, hidden_dim, num_classes, dropout=0.3):
                            super().__init__()
                            self.backbone = backbone
                            self.classifier = nn.Module()
                            self.classifier.net = nn.Sequential(
                                nn.LayerNorm(feat_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(feat_dim, hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim, num_classes)
                            )
                        
                        def forward(self, x):
                            features = self.backbone(x)
                            return self.classifier.net(features)
                    
                    pvt_model = PVT2WithClassifier(pvt_backbone, feat_dim, hidden_dim, classifier_classes)
                    missing, unexpected = pvt_model.load_state_dict(state_dict, strict=False)
                    
                    if len(missing) == 0:
                        pvt_model.eval()
                        logger.info("Successfully loaded PVTv2 model: %s", pvt_name)
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
    
    return None


def _try_load_resnet(state_dict: dict, num_classes: int) -> Any | None:
    """Try to load ResNet architectures."""
    try:
        import torchvision.models as models
        
        resnet_archs = [
            models.resnet18, models.resnet34, models.resnet50,
            models.resnet101, models.resnet152
        ]
        
        for resnet_class in resnet_archs:
            try:
                pytorch_model = resnet_class(pretrained=False, num_classes=num_classes)
                if _try_load_state_dict(pytorch_model, state_dict):
                    pytorch_model.eval()
                    logger.info("Successfully loaded model as %s", resnet_class.__name__)
                    return pytorch_model
            except Exception:
                continue
    except Exception as e:
        logger.debug("ResNet architectures failed: %s", e)
    
    return None


def _try_load_timm_model(state_dict: dict, num_classes: int, is_vit_like: bool) -> Any | None:
    """Try to load model using timm library."""
    try:
        import timm
        
        if is_vit_like:
            vit_models = [
                "vit_base_patch16_224", "vit_small_patch16_224", "vit_tiny_patch16_224",
                "vit_large_patch16_224", "deit_base_patch16_224", "deit_small_patch16_224"
            ]
            for model_name in vit_models:
                try:
                    pytorch_model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
                    if _try_load_state_dict(pytorch_model, state_dict):
                        pytorch_model.eval()
                        logger.info("Successfully loaded model as %s", model_name)
                        return pytorch_model
                except Exception:
                    continue
        
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
                if _try_load_state_dict(pytorch_model, state_dict):
                    pytorch_model.eval()
                    logger.info("Successfully loaded model as %s", model_name)
                    return pytorch_model
            except Exception:
                continue
                
    except ImportError:
        logger.warning("timm not available for model architecture inference")
    
    return None


def _try_load_state_dict(model_instance, state_dict_in) -> bool:
    """Try loading state dict with various prefix handling."""
    try:
        model_instance.load_state_dict(state_dict_in, strict=True)
        return True
    except Exception:
        for prefix in ["backbone.", "model.", "module.", ""]:
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
        return False


def _get_architecture_suggestion(is_vit_like: bool, has_backbone_prefix: bool) -> str:
    """Get helpful suggestion for model architecture."""
    if is_vit_like:
        return "Based on state_dict keys, this appears to be a Vision Transformer (ViT). Try: 'vit_base_patch16_224'."
    elif has_backbone_prefix:
        return "State_dict has 'backbone.' prefix. Try: 'resnet50', 'efficientnet_b0', or 'vit_base_patch16_224'."
    else:
        return "Try common architectures like 'resnet18', 'efficientnet_b0', 'resnet50', or 'vit_base_patch16_224'."


def create_model_from_arch(arch_name: str, num_classes: int, state_dict: dict) -> Any | None:
    """Create a model instance from architecture name and load state_dict."""
    import torch
    
    try:
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
                if _try_load_state_dict(pytorch_model, state_dict):
                    pytorch_model.eval()
                    logger.info("Successfully loaded model as %s", arch_name)
                    return pytorch_model
        except Exception as e:
            logger.debug("Failed to load %s from torchvision: %s", arch_name, e)
        
        # Try timm models
        try:
            import timm
            pytorch_model = timm.create_model(arch_name, pretrained=False, num_classes=num_classes)
            if _try_load_state_dict(pytorch_model, state_dict):
                pytorch_model.eval()
                logger.info("Successfully loaded model as %s from timm", arch_name)
                return pytorch_model
        except Exception as e:
            logger.debug("Failed to load %s from timm: %s", arch_name, e)
        
        return None
    except Exception as e:
        logger.error("Error creating model from arch %s: %s", arch_name, e)
        return None
