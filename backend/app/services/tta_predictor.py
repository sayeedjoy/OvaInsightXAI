"""Test-Time Augmentation (TTA) for brain tumor image classification.

Applies multiple augmentations to input images and averages the predictions
for more robust inference.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Configuration via environment variables
TTA_ENABLED = os.getenv("BRAIN_TUMOR_TTA_ENABLED", "true").lower() == "true"
TTA_PASSES = int(os.getenv("BRAIN_TUMOR_TTA_PASSES", "5"))


def _horizontal_flip(tensor: torch.Tensor) -> torch.Tensor:
    """Flip image horizontally."""
    return torch.flip(tensor, dims=[-1])


def _vertical_flip(tensor: torch.Tensor) -> torch.Tensor:
    """Flip image vertically."""
    return torch.flip(tensor, dims=[-2])


def _rotate_90(tensor: torch.Tensor) -> torch.Tensor:
    """Rotate image 90 degrees clockwise."""
    return torch.rot90(tensor, k=1, dims=[-2, -1])


def _rotate_180(tensor: torch.Tensor) -> torch.Tensor:
    """Rotate image 180 degrees."""
    return torch.rot90(tensor, k=2, dims=[-2, -1])


def _rotate_270(tensor: torch.Tensor) -> torch.Tensor:
    """Rotate image 270 degrees clockwise."""
    return torch.rot90(tensor, k=3, dims=[-2, -1])


def _center_crop_resize(tensor: torch.Tensor, crop_ratio: float = 0.9) -> torch.Tensor:
    """Apply center crop then resize back to original size."""
    _, _, h, w = tensor.shape
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
    
    cropped = tensor[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]
    # Resize back to original size
    resized = F.interpolate(cropped, size=(h, w), mode='bilinear', align_corners=False)
    return resized


# Define augmentation strategies - lightweight for performance
TTA_AUGMENTATIONS: List[Tuple[str, callable]] = [
    ("original", lambda x: x),
    ("hflip", _horizontal_flip),
    ("vflip", _vertical_flip),
    ("rotate_90", _rotate_90),
    ("center_crop", _center_crop_resize),
]


def predict_with_tta(
    model: Any,
    image_tensor: torch.Tensor,
    device: torch.device,
    num_passes: int | None = None
) -> Tuple[int, float, dict]:
    """
    Run prediction with Test-Time Augmentation.
    
    Args:
        model: PyTorch model in eval mode
        image_tensor: Input tensor of shape (1, 3, H, W)
        device: Device to run inference on
        num_passes: Number of TTA passes (uses TTA_PASSES env var if None)
        
    Returns:
        Tuple of (predicted_class_idx, confidence, all_probabilities_dict)
    """
    if num_passes is None:
        num_passes = TTA_PASSES
    
    # Limit passes to available augmentations
    num_passes = min(num_passes, len(TTA_AUGMENTATIONS))
    
    image_tensor = image_tensor.to(device)
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for i, (aug_name, aug_fn) in enumerate(TTA_AUGMENTATIONS[:num_passes]):
            try:
                augmented = aug_fn(image_tensor)
                outputs = model(augmented)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs)
                logger.debug("TTA pass %d (%s): probs=%s", i, aug_name, probs.cpu().numpy())
            except Exception as e:
                logger.warning("TTA augmentation %s failed: %s", aug_name, e)
                continue
    
    if not all_probs:
        raise ValueError("All TTA passes failed")
    
    # Average probabilities across all augmentations
    avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    
    # Get final prediction
    confidence, predicted_class = torch.max(avg_probs, dim=1)
    predicted_class_idx = predicted_class.item()
    confidence_value = confidence.item()
    
    # Build per-class probability dict
    all_class_probs = avg_probs.squeeze().cpu().tolist()
    
    logger.info(
        "TTA prediction: class=%d, confidence=%.4f, passes=%d",
        predicted_class_idx, confidence_value, len(all_probs)
    )
    
    return predicted_class_idx, confidence_value, all_class_probs


def is_tta_enabled() -> bool:
    """Check if TTA is enabled via environment variable."""
    return TTA_ENABLED


def get_tta_config() -> dict:
    """Return current TTA configuration."""
    return {
        "enabled": TTA_ENABLED,
        "passes": TTA_PASSES,
        "available_augmentations": [name for name, _ in TTA_AUGMENTATIONS],
    }
