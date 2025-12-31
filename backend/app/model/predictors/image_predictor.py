"""Image-based prediction logic for PyTorch models like brain tumor classifier."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List

from app.model.base import PredictionResult, _MODELS

logger = logging.getLogger(__name__)


@dataclass
class ImagePredictionResult:
    """Extended prediction result with per-class probabilities."""
    prediction: int
    confidence: float
    all_probabilities: List[float]


def predict_image(
    image_tensor: Any,
    *,
    model_key: str = "brain_tumor",
    use_tta: bool | None = None
) -> PredictionResult:
    """Run inference on an image tensor for PyTorch models.
    
    Args:
        image_tensor: Preprocessed image tensor of shape (1, 3, H, W)
        model_key: Model identifier to use
        use_tta: Whether to use Test-Time Augmentation. If None, uses env var setting.
        
    Returns:
        PredictionResult with prediction class index and confidence
    """
    if model_key not in _MODELS:
        raise RuntimeError("Model has not been loaded yet. Call ensure_model_loaded().")
    
    model = _MODELS[model_key]
    import torch
    
    try:
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # Check if TTA should be used
        from app.services.tta_predictor import is_tta_enabled, predict_with_tta
        
        should_use_tta = use_tta if use_tta is not None else is_tta_enabled()
        
        if should_use_tta:
            logger.info("Using TTA for prediction on %s", model_key)
            predicted_class_idx, confidence_value, all_probs = predict_with_tta(
                model, image_tensor, device
            )
        else:
            # Standard single-pass inference
            model.eval()
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                predicted_class_idx = predicted_class.item()
                confidence_value = confidence.item()
                all_probs = probabilities.squeeze().cpu().tolist()
        
        logger.info(
            "Image prediction completed for %s: class=%s, confidence=%.4f, tta=%s",
            model_key, predicted_class_idx, confidence_value, should_use_tta
        )
        
        # Store all_probs in a module-level cache for access by the endpoint
        _store_last_probabilities(model_key, all_probs)
        
        return PredictionResult(
            prediction=int(predicted_class_idx),
            confidence=float(confidence_value)
        )
        
    except Exception as exc:
        logger.error("Error during image prediction: %s", exc, exc_info=True)
        raise ValueError(f"Image prediction failed: {str(exc)}") from exc


# Cache for last prediction probabilities (per model)
_LAST_PROBABILITIES: dict[str, List[float]] = {}


def _store_last_probabilities(model_key: str, probs: List[float]) -> None:
    """Store probabilities from last prediction for retrieval."""
    _LAST_PROBABILITIES[model_key] = probs


def get_last_probabilities(model_key: str = "brain_tumor") -> List[float] | None:
    """Get per-class probabilities from the last prediction."""
    return _LAST_PROBABILITIES.get(model_key)

