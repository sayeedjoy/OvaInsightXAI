"""Accumulated Local Effects (ALE) computation for image-based models using patch-based approach."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch

from app.model import predictor

logger = logging.getLogger(__name__)

# Environment-configurable parameters
# Reduced parameters for faster computation while maintaining quality
DEFAULT_GRID_SIZE = int(os.getenv("ALE_IMAGE_GRID_SIZE", "4"))  # 4x4 = 16 patches
DEFAULT_N_BINS = int(os.getenv("ALE_IMAGE_N_BINS", "8"))  # Number of bins for ALE computation


def compute_image_ale_explanation(
    model_key: str,
    image_tensor: Any,
    grid_size: int | None = None,
    n_bins: int | None = None
) -> dict[str, Any]:
    """Compute ALE for an image prediction using patch-based approach.
    
    Divides the image into patches and computes accumulated local effects
    by binning patch intensities and measuring prediction changes.
    
    Args:
        model_key: The model key (e.g., "brain_tumor")
        image_tensor: Preprocessed image tensor (PyTorch tensor) with shape (1, C, H, W) or (C, H, W)
        grid_size: Size of the grid (e.g., 8 means 8x8 = 64 patches)
        n_bins: Number of bins for ALE computation (default: 10)
    
    Returns:
        Dictionary with ALE plots for each patch region
    """
    if grid_size is None:
        grid_size = DEFAULT_GRID_SIZE
    if n_bins is None:
        n_bins = DEFAULT_N_BINS
    
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
        
        # Compute ALE for each patch
        ale_plots = []
        
        # Limit number of patches to compute for performance
        max_patches = min(grid_size * grid_size, 9)  # Compute up to 9 patches
        
        for patch_idx in range(max_patches):
            try:
                # Calculate patch coordinates
                row = patch_idx // grid_size
                col = patch_idx % grid_size
                y_start = row * patch_h
                y_end = (row + 1) * patch_h if row < grid_size - 1 else h
                x_start = col * patch_w
                x_end = (col + 1) * patch_w if col < grid_size - 1 else w
                
                # Create bins for intensity values
        # Intensity range is [0, 1]
                bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Compute ALE values for each bin
                ale_values = []
                
                for i in range(len(bin_edges) - 1):
                    # Get bin boundaries
                    bin_low = bin_edges[i]
                    bin_high = bin_edges[i + 1]
                    
                    # Create modified images with patch intensity at bin boundaries
                    # We'll test multiple samples within the bin (reduced for performance)
                    n_samples_in_bin = 3
                    predictions_low = []
                    predictions_high = []
                    
                    for _ in range(n_samples_in_bin):
                        # Test with intensity at lower boundary
                        modified_tensor_low = image_tensor_no_batch.clone()
                        for channel in range(c):
                            patch_region = modified_tensor_low[channel, y_start:y_end, x_start:x_end]
                            IMAGENET_MEAN = 0.485 if channel == 0 else (0.456 if channel == 1 else 0.406)
                            IMAGENET_STD = 0.229 if channel == 0 else (0.224 if channel == 1 else 0.225)
                            denormalized_patch = patch_region * IMAGENET_STD + IMAGENET_MEAN
                            modified_patch = denormalized_patch * bin_low
                            normalized_patch = (modified_patch - IMAGENET_MEAN) / IMAGENET_STD
                            modified_tensor_low[channel, y_start:y_end, x_start:x_end] = normalized_patch
                        
                        with torch.no_grad():
                            modified_batch_low = modified_tensor_low.unsqueeze(0).to(device)
                            outputs_low = model(modified_batch_low)
                            probs_low = torch.nn.functional.softmax(outputs_low, dim=1)
                            pred_prob_low = probs_low[0, original_predicted_class].item()
                            predictions_low.append(pred_prob_low)
                        
                        # Test with intensity at upper boundary
                        modified_tensor_high = image_tensor_no_batch.clone()
                        for channel in range(c):
                            patch_region = modified_tensor_high[channel, y_start:y_end, x_start:x_end]
                            IMAGENET_MEAN = 0.485 if channel == 0 else (0.456 if channel == 1 else 0.406)
                            IMAGENET_STD = 0.229 if channel == 0 else (0.224 if channel == 1 else 0.225)
                            denormalized_patch = patch_region * IMAGENET_STD + IMAGENET_MEAN
                            modified_patch = denormalized_patch * bin_high
                            normalized_patch = (modified_patch - IMAGENET_MEAN) / IMAGENET_STD
                            modified_tensor_high[channel, y_start:y_end, x_start:x_end] = normalized_patch
                        
                        with torch.no_grad():
                            modified_batch_high = modified_tensor_high.unsqueeze(0).to(device)
                            outputs_high = model(modified_batch_high)
                            probs_high = torch.nn.functional.softmax(outputs_high, dim=1)
                            pred_prob_high = probs_high[0, original_predicted_class].item()
                            predictions_high.append(pred_prob_high)
                    
                    # Average difference in predictions
                    avg_diff = np.mean(predictions_high) - np.mean(predictions_low)
                    ale_values.append(float(avg_diff))
                
                # Accumulate effects
                accumulated = np.cumsum(ale_values).tolist()
                # Center around zero
                accumulated = [x - np.mean(accumulated) for x in accumulated]
                
                ale_plots.append({
                    "patch_index": int(patch_idx),
                    "patch_row": int(row),
                    "patch_col": int(col),
                    "patch_coords": {
                        "y_start": int(y_start),
                        "y_end": int(y_end),
                        "x_start": int(x_start),
                        "x_end": int(x_end)
                    },
                    "bin_centers": bin_centers.tolist(),
                    "ale_values": accumulated,
                })
                
            except Exception as patch_exc:
                logger.warning("Error computing ALE for patch %d: %s", patch_idx, patch_exc)
                continue
        
        return {
            "ale_plots": ale_plots,
            "grid_size": grid_size,
            "patch_size": {"height": patch_h, "width": patch_w},
            "image_size": {"height": h, "width": w},
            "predicted_class": int(original_predicted_class),
        }
        
    except ImportError as exc:
        logger.error("ALE import error during computation for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"ALE import failed: {str(exc)}", "error_type": "ImportError", "details": str(exc)}
    except OSError as exc:
        logger.error("ALE system library error for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"ALE system library error: {str(exc)}", "error_type": "OSError", "details": str(exc)}
    except RuntimeError as exc:
        logger.error("ALE runtime error for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"ALE runtime error: {str(exc)}", "error_type": "RuntimeError", "details": str(exc)}
    except MemoryError as exc:
        logger.error("ALE memory error for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"ALE memory error: {str(exc)}", "error_type": "MemoryError", "details": str(exc)}
    except Exception as exc:
        logger.error("Error computing ALE explanation for %s: %s (type: %s)", model_key, exc, type(exc).__name__, exc_info=True)
        return {"error": str(exc), "error_type": type(exc).__name__, "details": str(exc)}

