"""SHAP explanation computation for image-based models."""

from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from app.model import predictor

logger = logging.getLogger(__name__)

# Environment-configurable sample sizes for performance tuning
DEFAULT_BACKGROUND_SAMPLES = int(os.getenv("SHAP_IMAGE_BACKGROUND_SAMPLES", "10"))
# Larger patch/stride for faster computation (32px patch, 16px stride)
DEFAULT_PATCH_SIZE = int(os.getenv("SHAP_IMAGE_PATCH_SIZE", "32"))
DEFAULT_STRIDE = int(os.getenv("SHAP_IMAGE_STRIDE", "16"))
# Maximum batch size for inference (balance memory vs speed)
MAX_BATCH_SIZE = int(os.getenv("SHAP_IMAGE_BATCH_SIZE", "32"))


def compute_image_shap_explanation(
    model_key: str,
    image_tensor: Any,
    background_samples: int | None = None
) -> dict[str, Any]:
    """Compute SHAP-like values for an image prediction using occlusion sensitivity.
    
    Uses occlusion-based sensitivity analysis instead of GradientExplainer
    to avoid autograd issues with custom model architectures (e.g., PVTv2).
    
    OPTIMIZED: Uses batch inference to process multiple occluded images at once.
    
    Args:
        model_key: The model key (e.g., "brain_tumor")
        image_tensor: Preprocessed image tensor (PyTorch tensor)
        background_samples: Not used, kept for API compatibility
    
    Returns:
        Dictionary with SHAP-like explanation data including heatmap as base64 image
    """
    try:
        import torch
        logger.debug("Computing occlusion-based sensitivity for SHAP-like explanation")
    except ImportError as exc:
        logger.error("PyTorch not available: %s", exc, exc_info=True)
        return {"error": "PyTorch not installed", "error_type": "ImportError", "details": str(exc)}

    try:
        # Ensure model is loaded
        predictor.ensure_model_loaded(model_key)
        model = predictor.get_model(model_key)
        
        # Get device
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # Remove batch dimension if present (preprocessing returns (1, C, H, W))
        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            image_tensor_no_batch = image_tensor.squeeze(0)  # (C, H, W)
        else:
            image_tensor_no_batch = image_tensor
        
        # Ensure model is in eval mode
        model.eval()
        
        c, h, w = image_tensor_no_batch.shape
        
        # Get original prediction
        with torch.no_grad():
            original_tensor = image_tensor_no_batch.unsqueeze(0).to(device)
            original_outputs = model(original_tensor)
            original_probs = torch.nn.functional.softmax(original_outputs, dim=1)
            predicted_class = torch.argmax(original_probs, dim=1).item()
            original_prob = original_probs[0, predicted_class].item()
        
        # Use configurable patch/stride for faster computation
        patch_size = DEFAULT_PATCH_SIZE
        stride = DEFAULT_STRIDE
        
        # Calculate heatmap dimensions
        heatmap_h = (h - patch_size) // stride + 1
        heatmap_w = (w - patch_size) // stride + 1
        
        # Pre-create all occluded tensors for batch processing
        occluded_tensors = []
        grid_positions = []
        
        for i in range(heatmap_h):
            for j in range(heatmap_w):
                # Create occluded image
                occluded_tensor = image_tensor_no_batch.clone()
                y_start = i * stride
                y_end = min(y_start + patch_size, h)
                x_start = j * stride
                x_end = min(x_start + patch_size, w)
                
                # Occlude with zeros (black patch in normalized space)
                occluded_tensor[:, y_start:y_end, x_start:x_end] = 0
                
                occluded_tensors.append(occluded_tensor)
                grid_positions.append((i, j))
        
        # Process in batches for efficiency
        sensitivity_values = []
        batch_size = min(MAX_BATCH_SIZE, len(occluded_tensors))
        
        with torch.no_grad():
            for batch_start in range(0, len(occluded_tensors), batch_size):
                batch_end = min(batch_start + batch_size, len(occluded_tensors))
                batch_tensors = occluded_tensors[batch_start:batch_end]
                
                # Stack into batch and run inference
                batch = torch.stack(batch_tensors).to(device)
                outputs = model(batch)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                occluded_probs = probs[:, predicted_class].cpu().numpy()
                
                # Calculate sensitivity (drop in probability when region is occluded)
                for prob in occluded_probs:
                    sensitivity_values.append(original_prob - prob)
        
        # Reshape to heatmap
        sensitivity_map = np.array(sensitivity_values).reshape(heatmap_h, heatmap_w)
        
        # Resize sensitivity map to original image size
        from PIL import Image as PILImage
        sensitivity_resized = np.array(
            PILImage.fromarray(sensitivity_map.astype(np.float32)).resize(
                (w, h), PILImage.Resampling.BILINEAR
            )
        )
        
        # Normalize heatmap to [0, 1]
        heatmap = np.abs(sensitivity_resized)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Convert heatmap to base64 image for frontend
        heatmap_image = _heatmap_to_base64(heatmap)
        
        # Also return the raw heatmap data as a flattened array for frontend processing
        heatmap_data = heatmap.flatten().tolist()
        
        # Get prediction probabilities
        probs = original_probs[0].cpu().numpy().tolist()
        
        return {
            "heatmap_image": heatmap_image,  # Base64 encoded image
            "heatmap_data": heatmap_data,  # Flattened array for custom rendering
            "heatmap_shape": list(heatmap.shape),  # Original shape [H, W]
            "probabilities": probs,
            "predicted_class": int(predicted_class),
        }
            
    except ImportError as exc:
        logger.error("Import error during computation for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"Import failed: {str(exc)}", "error_type": "ImportError", "details": str(exc)}
    except OSError as exc:
        logger.error("System library error for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"System library error: {str(exc)}", "error_type": "OSError", "details": str(exc)}
    except RuntimeError as exc:
        logger.error("Runtime error for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"Runtime error: {str(exc)}", "error_type": "RuntimeError", "details": str(exc)}
    except MemoryError as exc:
        logger.error("Memory error for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"Memory error: {str(exc)}", "error_type": "MemoryError", "details": str(exc)}
    except Exception as exc:
        logger.error("Error computing explanation for %s: %s (type: %s)", model_key, exc, type(exc).__name__, exc_info=True)
        return {"error": str(exc), "error_type": type(exc).__name__, "details": str(exc)}


def _generate_background_images(image_tensor: Any, n_samples: int = 10) -> Any:
    """Generate background images for SHAP explainer.
    
    Creates masked versions of the input image by randomly zeroing out regions.
    """
    import torch
    
    # Get image shape (assuming [C, H, W] or [1, C, H, W])
    if image_tensor.dim() == 4:
        _, c, h, w = image_tensor.shape
        single_image = image_tensor[0]
    else:
        c, h, w = image_tensor.shape
        single_image = image_tensor
    
    # Create background samples by masking random regions
    background_list = []
    for _ in range(n_samples):
        # Create a mask with random regions set to zero
        mask = torch.ones_like(single_image)
        # Randomly mask out 30-70% of the image
        mask_ratio = np.random.uniform(0.3, 0.7)
        num_pixels = int(h * w * mask_ratio)
        
        # Randomly select pixels to mask
        flat_mask = mask.view(c, -1)
        indices = torch.randperm(h * w)[:num_pixels]
        for idx in indices:
            flat_mask[:, idx] = 0
        
        masked_image = single_image * mask
        background_list.append(masked_image)
    
    # Stack into batch tensor
    background = torch.stack(background_list)
    return background


def _heatmap_to_base64(heatmap: np.ndarray) -> str:
    """Convert heatmap numpy array to base64-encoded PNG image.
    
    Args:
        heatmap: 2D numpy array with values in [0, 1]
    
    Returns:
        Base64-encoded PNG image string
    """
    # Normalize to [0, 255]
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap (using PIL's built-in colormap)
    # Convert to PIL Image
    img = Image.fromarray(heatmap_uint8, mode='L')
    
    # Apply colormap (red for high values, blue for low values)
    # We'll use a simple approach: convert grayscale to RGB with colormap
    img_rgb = img.convert('RGB')
    
    # Apply a colormap effect (red-yellow for high importance)
    img_array = np.array(img_rgb)
    heatmap_normalized = heatmap_uint8 / 255.0
    
    # Create red-yellow colormap
    red_channel = (heatmap_normalized * 255).astype(np.uint8)
    green_channel = (heatmap_normalized * 200).astype(np.uint8)  # Yellow component
    blue_channel = ((1 - heatmap_normalized) * 100).astype(np.uint8)  # Blue for low values
    
    img_colored = np.stack([red_channel, green_channel, blue_channel], axis=2)
    
    # Convert back to PIL Image
    img_final = Image.fromarray(img_colored, mode='RGB')
    
    # Convert to base64
    buffer = BytesIO()
    img_final.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_str}"
