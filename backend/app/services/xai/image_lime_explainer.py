"""LIME explanation computation for image-based models."""

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

# Environment-configurable sample sizes for performance tuning
# Reduced from 300 to 100 for faster computation while maintaining quality
DEFAULT_NUM_SAMPLES = int(os.getenv("LIME_IMAGE_NUM_SAMPLES", "100"))
DEFAULT_NUM_FEATURES = int(os.getenv("LIME_IMAGE_NUM_FEATURES", "10"))
# Batch size for model inference (balance memory vs speed)
LIME_BATCH_SIZE = int(os.getenv("LIME_IMAGE_BATCH_SIZE", "16"))


def compute_image_lime_explanation(
    model_key: str,
    image_tensor: Any,
    num_samples: int | None = None,
    num_features: int | None = None
) -> dict[str, Any]:
    """Compute LIME explanation for an image prediction.
    
    OPTIMIZED: Uses batch inference in predict_fn for faster processing.
    
    Args:
        model_key: The model key (e.g., "brain_tumor")
        image_tensor: Preprocessed image tensor (PyTorch tensor)
        num_samples: Number of LIME perturbation samples (default: 100)
        num_features: Number of top features to return (default: 10)
    
    Returns:
        Dictionary with LIME explanation data including superpixel segmentation
    """
    if num_samples is None:
        num_samples = DEFAULT_NUM_SAMPLES
    if num_features is None:
        num_features = DEFAULT_NUM_FEATURES
    
    try:
        from lime import lime_image
        logger.debug("LIME library imported successfully")
    except ImportError as exc:
        logger.error("LIME library not available: %s", exc, exc_info=True)
        return {"error": "LIME library not installed", "error_type": "ImportError", "details": str(exc)}
    except Exception as exc:
        logger.error("Unexpected error importing LIME: %s", exc, exc_info=True)
        return {"error": f"Failed to import LIME: {str(exc)}", "error_type": type(exc).__name__, "details": str(exc)}

    # Initialize image_np to None to ensure it's always defined
    image_np = None
    
    try:
        # Ensure model is loaded
        predictor.ensure_model_loaded(model_key)
        model = predictor.get_model(model_key)
        
        # Get device
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # Remove batch dimension if present (preprocessing returns (1, C, H, W))
        # We need (C, H, W) for conversion to numpy
        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            image_tensor_no_batch = image_tensor.squeeze(0)  # (C, H, W)
        else:
            image_tensor_no_batch = image_tensor
        
        # Ensure model is in eval mode
        model.eval()
        
        # Convert tensor to numpy for LIME (LIME expects numpy arrays)
        # LIME expects images in format [H, W, C] with values in [0, 1] or [0, 255]
        image_np = _tensor_to_numpy_for_lime(image_tensor_no_batch)
        
        # Create LIME image explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Define prediction function that works with numpy arrays
        # OPTIMIZED: Process in batches for much faster inference
        def predict_fn(images: np.ndarray) -> np.ndarray:
            """Convert numpy images to tensors, run model in batches, return probabilities."""
            batch_size_total = images.shape[0]
            all_predictions = []
            
            # Process in batches
            for batch_start in range(0, batch_size_total, LIME_BATCH_SIZE):
                batch_end = min(batch_start + LIME_BATCH_SIZE, batch_size_total)
                batch_images = images[batch_start:batch_end]
                
                # Convert batch to tensors
                batch_tensors = []
                for i in range(len(batch_images)):
                    img = batch_images[i]
                    img_tensor = _numpy_to_tensor_for_model(img)
                    batch_tensors.append(img_tensor)
                
                # Stack into batch and run inference
                batch = torch.stack(batch_tensors).to(device)
                
                with torch.no_grad():
                    outputs = model(batch)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    probs = probabilities.cpu().numpy()
                
                all_predictions.append(probs)
            
            return np.concatenate(all_predictions, axis=0)
        
        # Explain the image
        explanation = explainer.explain_instance(
            image_np,
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )
        
        # Get the top label
        top_label = explanation.top_labels[0]
        
        # Get explanation for the top label
        temp, mask = explanation.get_image_and_mask(
            top_label,
            positive_only=True,
            num_features=num_features,
            hide_rest=False
        )
        
        # Get detailed explanation
        explanation_data = explanation.local_exp[top_label]
        
        # Convert explanation to feature importance list
        feature_importance = []
        for feature_idx, importance in explanation_data[:num_features]:
            # Get superpixel region
            superpixel_mask = (mask == feature_idx).astype(float)
            feature_importance.append({
                "feature_index": int(feature_idx),
                "importance": float(importance),
                "superpixel_mask": superpixel_mask.flatten().tolist(),
                "superpixel_shape": list(superpixel_mask.shape)
            })
        
        # Create visualization image
        # Overlay mask on original image
        visualization_image = _create_lime_visualization(image_np, temp, mask)
        
        # Get prediction probabilities
        # Ensure image_np is defined before using it
        if image_np is None:
            raise ValueError("Failed to convert image tensor to numpy array")
        
        with torch.no_grad():
            img_tensor = _numpy_to_tensor_for_model(image_np)
            img_tensor = img_tensor.to(device).unsqueeze(0)
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probs = probabilities[0].cpu().numpy().tolist()
        
        return {
            "visualization_image": visualization_image,  # Base64 encoded image
            "feature_importance": feature_importance,
            "top_label": int(top_label),
            "probabilities": probs,
            "mask_shape": list(mask.shape),
        }
        
    except ImportError as exc:
        logger.error("LIME import error during computation for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"LIME import failed: {str(exc)}", "error_type": "ImportError", "details": str(exc)}
    except OSError as exc:
        logger.error("LIME system library error for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"LIME system library error: {str(exc)}", "error_type": "OSError", "details": str(exc)}
    except RuntimeError as exc:
        logger.error("LIME runtime error for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"LIME runtime error: {str(exc)}", "error_type": "RuntimeError", "details": str(exc)}
    except MemoryError as exc:
        logger.error("LIME memory error for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"LIME memory error: {str(exc)}", "error_type": "MemoryError", "details": str(exc)}
    except Exception as exc:
        logger.error("Error computing LIME explanation for %s: %s (type: %s)", model_key, exc, type(exc).__name__, exc_info=True)
        return {"error": str(exc), "error_type": type(exc).__name__, "details": str(exc)}


def _tensor_to_numpy_for_lime(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert preprocessed PyTorch tensor to numpy array for LIME.
    
    LIME expects images in [H, W, C] format with values in [0, 1] or [0, 255].
    Our preprocessing creates [C, H, W] tensors with ImageNet normalization.
    We need to denormalize and convert to [H, W, C].
    """
    # ImageNet normalization constants
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    # Remove batch dimension if present
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    
    # Convert to numpy and move to CPU
    img_np = image_tensor.cpu().numpy()
    
    # Convert from [C, H, W] to [H, W, C]
    img_np = np.transpose(img_np, (1, 2, 0))
    
    # Denormalize
    img_np = img_np * IMAGENET_STD + IMAGENET_MEAN
    
    # Clip to [0, 1] range
    img_np = np.clip(img_np, 0, 1)
    
    return img_np


def _numpy_to_tensor_for_model(image_np: np.ndarray) -> torch.Tensor:
    """Convert numpy array back to preprocessed tensor for model inference.
    
    Converts from [H, W, C] numpy array to [C, H, W] tensor with ImageNet normalization.
    """
    # ImageNet normalization constants
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    # Normalize (use a different variable name to avoid shadowing)
    normalized = (image_np - IMAGENET_MEAN) / IMAGENET_STD
    
    # Convert from [H, W, C] to [C, H, W]
    transposed = np.transpose(normalized, (2, 0, 1))
    
    # Convert to tensor
    img_tensor = torch.from_numpy(transposed).float()
    
    return img_tensor


def _create_lime_visualization(original: np.ndarray, temp: np.ndarray, mask: np.ndarray) -> str:
    """Create visualization image showing LIME explanation overlay.
    
    Args:
        original: Original image [H, W, C]
        temp: LIME explanation image [H, W, C]
        mask: Superpixel mask [H, W]
    
    Returns:
        Base64-encoded PNG image string
    """
    # Create overlay: show original image with highlighted important regions
    # Convert to uint8
    original_uint8 = (np.clip(original, 0, 1) * 255).astype(np.uint8)
    temp_uint8 = (np.clip(temp, 0, 1) * 255).astype(np.uint8)
    
    # Create a colored mask overlay (red for important regions)
    mask_colored = np.zeros_like(original_uint8)
    mask_colored[:, :, 0] = mask * 255  # Red channel
    mask_colored[:, :, 1] = mask * 128  # Green channel (for yellow effect)
    
    # Blend original with mask
    alpha = 0.5
    overlay = (original_uint8 * (1 - alpha) + mask_colored * alpha).astype(np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(overlay, mode='RGB')
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_str}"
