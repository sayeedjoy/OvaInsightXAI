"""LIME explanation computation."""

from __future__ import annotations

import logging
import os
from typing import Any

from app.model.registry import MODEL_REGISTRY
from app.services.xai.utils import generate_training_data, get_model_and_features

logger = logging.getLogger(__name__)

# Environment-configurable sample sizes for performance tuning
DEFAULT_TRAINING_SAMPLES = int(os.getenv("LIME_TRAINING_SAMPLES", "200"))
DEFAULT_NUM_SAMPLES = int(os.getenv("LIME_NUM_SAMPLES", "1000"))  # LIME perturbation samples


def compute_lime_explanation(
    model_key: str,
    instance_features: list[float],
    num_features: int = 10,
    training_samples: int | None = None,
    num_samples: int | None = None
) -> dict[str, Any]:
    """Compute LIME explanation for the prediction."""
    # Use environment-configurable defaults
    if training_samples is None:
        training_samples = DEFAULT_TRAINING_SAMPLES
    if num_samples is None:
        num_samples = DEFAULT_NUM_SAMPLES
    
    try:
        from lime import lime_tabular
        logger.debug("LIME library imported successfully")
    except ImportError as exc:
        logger.error("LIME library not available: %s", exc, exc_info=True)
        return {"error": "LIME library not installed", "error_type": "ImportError", "details": str(exc)}
    except Exception as exc:
        logger.error("Unexpected error importing LIME: %s", exc, exc_info=True)
        return {"error": f"Failed to import LIME: {str(exc)}", "error_type": type(exc).__name__, "details": str(exc)}

    try:
        model, instance_array = get_model_and_features(model_key, instance_features)
        config = MODEL_REGISTRY[model_key]

        # Generate training data for LIME (as numpy array - LIME expects arrays)
        # Reduced sample size for better performance
        X_train, _ = generate_training_data(model_key, n_samples=training_samples, return_dataframe=False)

        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=config.feature_order,
            mode="classification",
            random_state=42
        )

        # Explain the instance with reduced sample count for speed
        explanation = explainer.explain_instance(
            instance_array[0],
            model.predict_proba if hasattr(model, "predict_proba") else model.predict,
            num_features=num_features,
            top_labels=1,
            num_samples=num_samples  # Reduced from default 5000 for better performance
        )

        # Extract explanation
        # When top_labels is used, we need to get the label from available_labels
        available_labels = explanation.available_labels()
        if available_labels:
            # Use the first available label (positive class for binary classification)
            label = available_labels[0]
            exp_list = explanation.as_list(label=label)
        else:
            # Fallback: try without label parameter
            exp_list = explanation.as_list()
        
        feature_importance = [
            {
                "feature": item[0],
                "importance": float(item[1]),
            }
            for item in exp_list
        ]

        return {
            "feature_importance": feature_importance,
            "prediction": float(model.predict_proba(instance_array)[0][1]) if hasattr(model, "predict_proba") else None,
        }
    except ImportError as exc:
        logger.error("LIME import error during computation for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"LIME import failed: {str(exc)}", "error_type": "ImportError", "details": str(exc)}
    except OSError as exc:
        logger.error("LIME system library error for %s (missing system dependencies?): %s", model_key, exc, exc_info=True)
        return {"error": f"LIME system library error: {str(exc)}", "error_type": "OSError", "details": str(exc)}
    except RuntimeError as exc:
        logger.error("LIME runtime error for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"LIME runtime error: {str(exc)}", "error_type": "RuntimeError", "details": str(exc)}
    except MemoryError as exc:
        logger.error("LIME memory error for %s (insufficient memory): %s", model_key, exc, exc_info=True)
        return {"error": f"LIME memory error: {str(exc)}", "error_type": "MemoryError", "details": str(exc)}
    except Exception as exc:
        logger.error("Error computing LIME explanation for %s: %s (type: %s)", model_key, exc, type(exc).__name__, exc_info=True)
        return {"error": str(exc), "error_type": type(exc).__name__, "details": str(exc)}

