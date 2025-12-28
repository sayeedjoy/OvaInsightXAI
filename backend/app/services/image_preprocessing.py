"""Image preprocessing utilities for brain tumor MRI classification."""

from __future__ import annotations

import logging
from io import BytesIO

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet normalization statistics (standard for most vision models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard image size for vision models
IMAGE_SIZE = 224


def preprocess_brain_mri_image(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocess a brain MRI image for model inference.
    
    Args:
        image_bytes: Raw image bytes (JPEG or PNG)
        
    Returns:
        Preprocessed image tensor with shape (1, 3, 224, 224)
    """
    try:
        # Load image from bytes
        image = Image.open(BytesIO(image_bytes))
        
        # Convert grayscale to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize to standard size (224x224)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        
        # Convert PIL image to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Apply ImageNet normalization
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Convert to CHW format (channels first)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Convert to tensor
        tensor = torch.from_numpy(img_array).float()
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        logger.debug("Preprocessed image: shape=%s, dtype=%s", tensor.shape, tensor.dtype)
        
        return tensor
        
    except Exception as exc:
        logger.error("Error preprocessing image: %s", exc, exc_info=True)
        raise ValueError(f"Failed to preprocess image: {str(exc)}") from exc

