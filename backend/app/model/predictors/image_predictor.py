"""Image-based prediction logic for PyTorch models like brain tumor classifier."""

from __future__ import annotations

import logging
from typing import Any

from app.model.base import PredictionResult, _MODELS

logger = logging.getLogger(__name__)


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
