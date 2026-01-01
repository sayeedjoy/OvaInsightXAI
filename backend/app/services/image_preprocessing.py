"""Image preprocessing utilities for brain tumor MRI classification."""

from __future__ import annotations

import logging
import os
from io import BytesIO

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet normalization statistics (fallback if dataset-specific stats not available)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard image size for vision models
IMAGE_SIZE = 224

# Cache for dataset-specific normalization stats (loaded from checkpoint or env vars)
_DATASET_MEAN: list[float] | None = None
_DATASET_STD: list[float] | None = None


def _load_normalization_from_env() -> tuple[list[float] | None, list[float] | None]:
    """Load normalization statistics from environment variables.
    
    Returns:
        Tuple of (mean, std) lists, or (None, None) if not set.
        Environment variables: BRAIN_TUMOR_MEAN_RGB and BRAIN_TUMOR_STD_RGB (comma-separated floats)
    """
    mean_str = os.getenv("BRAIN_TUMOR_MEAN_RGB")
    std_str = os.getenv("BRAIN_TUMOR_STD_RGB")
    
    mean = None
    std = None
    
    if mean_str:
        try:
            mean = [float(x.strip()) for x in mean_str.split(",")]
            if len(mean) != 3:
                logger.warning("BRAIN_TUMOR_MEAN_RGB must have 3 values (RGB), got %d", len(mean))
                mean = None
        except ValueError as exc:
            logger.warning("Failed to parse BRAIN_TUMOR_MEAN_RGB: %s", exc)
    
    if std_str:
        try:
            std = [float(x.strip()) for x in std_str.split(",")]
            if len(std) != 3:
                logger.warning("BRAIN_TUMOR_STD_RGB must have 3 values (RGB), got %d", len(std))
                std = None
        except ValueError as exc:
            logger.warning("Failed to parse BRAIN_TUMOR_STD_RGB: %s", exc)
    
    return mean, std


def set_dataset_normalization(mean: list[float] | None, std: list[float] | None) -> None:
    """Set dataset-specific normalization statistics (e.g., from model checkpoint).
    
    Args:
        mean: List of 3 floats for RGB mean values
        std: List of 3 floats for RGB std values
    """
    global _DATASET_MEAN, _DATASET_STD
    if mean is not None and len(mean) == 3:
        _DATASET_MEAN = mean
        logger.info("Set dataset mean: %s", _DATASET_MEAN)
    if std is not None and len(std) == 3:
        _DATASET_STD = std
        logger.info("Set dataset std: %s", _DATASET_STD)


def get_normalization_stats() -> tuple[list[float], list[float]]:
    """Get normalization statistics (dataset-specific if available, else ImageNet).
    
    Returns:
        Tuple of (mean, std) lists, each with 3 RGB values
    """
    global _DATASET_MEAN, _DATASET_STD
    
    # Check cache first
    if _DATASET_MEAN is not None and _DATASET_STD is not None:
        return _DATASET_MEAN, _DATASET_STD
    
    # Try environment variables
    mean, std = _load_normalization_from_env()
    if mean is not None and std is not None:
        _DATASET_MEAN = mean
        _DATASET_STD = std
        return mean, std
    
    # Fallback to ImageNet
    logger.warning(
        "Using ImageNet normalization stats as fallback. "
        "For best results, set BRAIN_TUMOR_MEAN_RGB and BRAIN_TUMOR_STD_RGB "
        "or load from model checkpoint."
    )
    return IMAGENET_MEAN, IMAGENET_STD


def preprocess_brain_mri_image(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocess a brain MRI image for model inference.
    
    Matches training preprocessing:
    - Load as RGB
    - Resize to 224x224 using bilinear interpolation (torchvision default)
    - Normalize to [0, 1] range
    - Apply dataset-specific normalization: (x - MEAN) / STD
    
    Args:
        image_bytes: Raw image bytes (JPEG or PNG)
        
    Returns:
        Preprocessed image tensor with shape (1, 3, 224, 224)
    """
    try:
        # Load image from bytes
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB (matches training: Image.open(...).convert("RGB"))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize to standard size (224x224) using bilinear interpolation
        # torchvision.transforms.Resize uses bilinear by default
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        
        # Convert PIL image to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1] range (matches training: transforms.ToTensor())
        img_array = img_array / 255.0
        
        # Get normalization statistics (dataset-specific if available, else ImageNet)
        mean_list, std_list = get_normalization_stats()
        mean = np.array(mean_list, dtype=np.float32)
        std = np.array(std_list, dtype=np.float32)
        
        # Apply normalization: (x - MEAN) / STD
        # Reshape mean/std to (1, 3, 1, 1) for broadcasting, but we'll do it per-channel
        # since we're working with HWC format first
        for c in range(3):
            img_array[:, :, c] = (img_array[:, :, c] - mean[c]) / std[c]
        
        # Convert to CHW format (channels first)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Convert to tensor
        tensor = torch.from_numpy(img_array).float()
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        logger.debug(
            "Preprocessed image: shape=%s, dtype=%s, mean=%s, std=%s",
            tensor.shape, tensor.dtype, mean_list, std_list
        )
        
        return tensor
        
    except Exception as exc:
        logger.error("Error preprocessing image: %s", exc, exc_info=True)
        raise ValueError(f"Failed to preprocess image: {str(exc)}") from exc

