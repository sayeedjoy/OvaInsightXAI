"""Partial Dependence Plot (PDP) computation for image-based models using patch-based approach."""

from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image
import torch

from app.model import predictor

logger = logging.getLogger(__name__)

# Environment-configurable parameters
# Reduced grid size (3x3=9 patches) and grid points (5) for faster computation
DEFAULT_GRID_SIZE = int(os.getenv("PDP_IMAGE_GRID_SIZE", "3"))
DEFAULT_N_GRID_POINTS = int(os.getenv("PDP_IMAGE_N_GRID_POINTS", "5"))
# Batch size for inference
PDP_BATCH_SIZE = int(os.getenv("PDP_IMAGE_BATCH_SIZE", "16"))


def compute_image_pdp_explanation(
    model_key: str,
    image_tensor: Any,
    grid_size: int | None = None,
    n_grid_points: int | None = None
) -> dict[str, Any]:
    """Compute PDP for an image prediction using patch-based approach.
    
    OPTIMIZED: Uses batch inference to process multiple intensity variations at once.
    
    Divides the image into patches and measures how varying each patch's
    intensity affects the prediction.
    
    Args:
        model_key: The model key (e.g., "brain_tumor")
        image_tensor: Preprocessed image tensor (PyTorch tensor) with shape (1, C, H, W) or (C, H, W)
        grid_size: Size of the grid (e.g., 3 means 3x3 = 9 patches)
        n_grid_points: Number of intensity levels to test per patch (default: 5)
    
    Returns:
        Dictionary with PDP plots for each patch region
    """
    if grid_size is None:
        grid_size = DEFAULT_GRID_SIZE
    if n_grid_points is None:
        n_grid_points = DEFAULT_N_GRID_POINTS
    
    try:
        # Ensure model is loaded
        predictor.ensure_model_loaded(model_key)
        model = predictor.get_model(model_key)
        
        # Get device
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # Remove batch dimension if present
        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            image_tensor_no_batch = image_tensor.squeeze(0)  # (C, H, W)
        else:
            image_tensor_no_batch = image_tensor
        
        # Ensure model is in eval mode
        model.eval()
        
        # Get image dimensions
        c, h, w = image_tensor_no_batch.shape
        
        # Calculate patch size
        patch_h = h // grid_size
        patch_w = w // grid_size
        
        # Get original prediction for reference
        with torch.no_grad():
            original_tensor = image_tensor_no_batch.unsqueeze(0).to(device)
            original_outputs = model(original_tensor)
            original_probs = torch.nn.functional.softmax(original_outputs, dim=1)
            original_predicted_class = torch.argmax(original_probs, dim=1).item()
        
        # ImageNet normalization constants
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        
        # Create grid of intensity values to test
        intensity_values = np.linspace(0.0, 1.0, n_grid_points)
        
        # Compute PDP for each patch using batch inference
        pdp_plots = []
        
        for patch_idx in range(grid_size * grid_size):
            try:
                # Calculate patch coordinates
                row = patch_idx // grid_size
                col = patch_idx % grid_size
                y_start = row * patch_h
                y_end = (row + 1) * patch_h if row < grid_size - 1 else h
                x_start = col * patch_w
                x_end = (col + 1) * patch_w if col < grid_size - 1 else w
                
                # Pre-create all modified tensors for this patch
                modified_tensors = []
                
                for intensity in intensity_values:
                    # Create modified image with patch intensity changed
                    modified_tensor = image_tensor_no_batch.clone()
                    
                    # Apply intensity to the patch region
                    for channel in range(c):
                        patch_region = modified_tensor[channel, y_start:y_end, x_start:x_end]
                        
                        # Denormalize, apply intensity, then renormalize
                        denormalized_patch = patch_region * IMAGENET_STD[channel] + IMAGENET_MEAN[channel]
                        modified_patch = denormalized_patch * intensity
                        normalized_patch = (modified_patch - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
                        
                        modified_tensor[channel, y_start:y_end, x_start:x_end] = normalized_patch
                    
                    modified_tensors.append(modified_tensor)
                
                # Batch inference for all intensity levels
                predictions = []
                batch_size = min(PDP_BATCH_SIZE, len(modified_tensors))
                
                with torch.no_grad():
                    for batch_start in range(0, len(modified_tensors), batch_size):
                        batch_end = min(batch_start + batch_size, len(modified_tensors))
                        batch = torch.stack(modified_tensors[batch_start:batch_end]).to(device)
                        outputs = model(batch)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        batch_preds = probs[:, original_predicted_class].cpu().numpy()
                        predictions.extend([float(p) for p in batch_preds])
                
                pdp_plots.append({
                    "patch_index": int(patch_idx),
                    "patch_row": int(row),
                    "patch_col": int(col),
                    "patch_coords": {
                        "y_start": int(y_start),
                        "y_end": int(y_end),
                        "x_start": int(x_start),
                        "x_end": int(x_end)
                    },
                    "intensity_values": intensity_values.tolist(),
                    "predictions": predictions,
                })
                
            except Exception as patch_exc:
                logger.warning("Error computing PDP for patch %d: %s", patch_idx, patch_exc)
                continue
        
        return {
            "pdp_plots": pdp_plots,
            "grid_size": grid_size,
            "patch_size": {"height": patch_h, "width": patch_w},
            "image_size": {"height": h, "width": w},
            "predicted_class": int(original_predicted_class),
        }
        
    except ImportError as exc:
        logger.error("PDP import error during computation for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"PDP import failed: {str(exc)}", "error_type": "ImportError", "details": str(exc)}
    except OSError as exc:
        logger.error("PDP system library error for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"PDP system library error: {str(exc)}", "error_type": "OSError", "details": str(exc)}
    except RuntimeError as exc:
        logger.error("PDP runtime error for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"PDP runtime error: {str(exc)}", "error_type": "RuntimeError", "details": str(exc)}
    except MemoryError as exc:
        logger.error("PDP memory error for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"PDP memory error: {str(exc)}", "error_type": "MemoryError", "details": str(exc)}
    except Exception as exc:
        logger.error("Error computing PDP explanation for %s: %s (type: %s)", model_key, exc, type(exc).__name__, exc_info=True)
        return {"error": str(exc), "error_type": type(exc).__name__, "details": str(exc)}
