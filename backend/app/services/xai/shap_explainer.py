"""SHAP explanation computation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.model.registry import MODEL_REGISTRY
from app.services.xai.utils import generate_training_data, get_model_and_features

logger = logging.getLogger(__name__)


def compute_shap_explanation(
    model_key: str,
    instance_features: list[float],
    background_samples: int = 50
) -> dict[str, Any]:
    """Compute SHAP values for the prediction."""
    try:
        import shap
        logger.debug("SHAP library imported successfully")
    except ImportError as exc:
        logger.error("SHAP library not available: %s", exc, exc_info=True)
        return {"error": "SHAP library not installed", "error_type": "ImportError", "details": str(exc)}
    except Exception as exc:
        logger.error("Unexpected error importing SHAP: %s", exc, exc_info=True)
        return {"error": f"Failed to import SHAP: {str(exc)}", "error_type": type(exc).__name__, "details": str(exc)}

    try:
        model, instance_array = get_model_and_features(model_key, instance_features)
        config = MODEL_REGISTRY[model_key]

        # Get background data as DataFrame with feature names to avoid sklearn warnings
        X_background, _ = generate_training_data(model_key, n_samples=background_samples, return_dataframe=True)
        
        # Convert instance to DataFrame with feature names
        instance_df = pd.DataFrame(instance_array, columns=config.feature_order)

        # Extract the final estimator from Pipeline if needed
        if isinstance(model, Pipeline):
            # Get the final step (classifier/estimator)
            final_step_name, final_estimator = model.steps[-1]
            classifier_type = type(final_estimator).__name__
            logger.debug("Extracted classifier from Pipeline: %s (type: %s)", final_step_name, classifier_type)
        else:
            final_estimator = model
            classifier_type = type(model).__name__

        # Use appropriate explainer with fallback strategy for ovarian model
        explainer = _create_shap_explainer(model, model_key, classifier_type, final_estimator, X_background)

        # For LinearExplainer with Pipeline, we need to transform the instance too
        if isinstance(model, Pipeline) and ("Linear" in classifier_type or "Logistic" in classifier_type):
            # Transform instance through pipeline steps (except the final estimator)
            instance_transformed = instance_df
            for step_name, step_transformer in model.steps[:-1]:
                instance_transformed = step_transformer.transform(instance_transformed)
            shap_values = explainer(instance_transformed)
        else:
            # For other explainers, use the DataFrame with feature names
            shap_values = explainer(instance_df)

        # Extract and format SHAP values
        base_value, values = _extract_shap_values(shap_values)

        # Ensure values is a numpy array
        if not isinstance(values, np.ndarray):
            values = np.array(values)

        # For binary classification, get values for the positive class
        if len(values.shape) > 1:
            values = values[:, 1] if values.shape[1] > 1 else values[:, 0]
        elif len(values.shape) == 0:
            # Scalar value, convert to array
            values = np.array([values])

        feature_names = config.feature_order
        contributions = [
            {
                "feature": feature_names[i],
                "value": float(instance_features[i]),
                "shap_value": float(values[i]) if i < len(values) else 0.0,
            }
            for i in range(len(feature_names))
        ]

        return {
            "base_value": base_value,
            "contributions": contributions,
            "prediction": float(model.predict_proba(instance_array)[0][1]) if hasattr(model, "predict_proba") else None,
        }
    except ImportError as exc:
        logger.error("SHAP import error during computation for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"SHAP import failed: {str(exc)}", "error_type": "ImportError", "details": str(exc)}
    except OSError as exc:
        logger.error("SHAP system library error for %s (missing system dependencies?): %s", model_key, exc, exc_info=True)
        return {"error": f"SHAP system library error: {str(exc)}", "error_type": "OSError", "details": str(exc)}
    except RuntimeError as exc:
        logger.error("SHAP runtime error for %s: %s", model_key, exc, exc_info=True)
        return {"error": f"SHAP runtime error: {str(exc)}", "error_type": "RuntimeError", "details": str(exc)}
    except MemoryError as exc:
        logger.error("SHAP memory error for %s (insufficient memory): %s", model_key, exc, exc_info=True)
        return {"error": f"SHAP memory error: {str(exc)}", "error_type": "MemoryError", "details": str(exc)}
    except Exception as exc:
        logger.error("Error computing SHAP explanation for %s: %s (type: %s)", model_key, exc, type(exc).__name__, exc_info=True)
        return {"error": str(exc), "error_type": type(exc).__name__, "details": str(exc)}


def _create_shap_explainer(model, model_key: str, classifier_type: str, final_estimator, X_background):
    """Create appropriate SHAP explainer with fallback strategy."""
    import shap
    
    explainer = None
    explainer_error = None
    
    try:
        if "Tree" in classifier_type or "XGB" in classifier_type or "LGBM" in classifier_type:
            # TreeExplainer can work with the pipeline directly
            # For ovarian model, try with smaller background sample if it fails
            try:
                explainer = shap.TreeExplainer(model, X_background)
            except Exception as tree_exc:
                logger.warning("TreeExplainer failed for %s, trying with smaller background: %s", model_key, tree_exc)
                # Try with smaller background for ovarian model
                if model_key == "ovarian" and len(X_background) > 30:
                    if isinstance(X_background, pd.DataFrame):
                        X_background_small = X_background.sample(n=30, random_state=42)
                    else:
                        X_background_small = X_background[:30]
                    explainer = shap.TreeExplainer(model, X_background_small)
                else:
                    raise tree_exc
        elif "Linear" in classifier_type or "Logistic" in classifier_type:
            # LinearExplainer needs the actual estimator, not the pipeline
            # But we need to transform the background data through the pipeline steps first
            if isinstance(model, Pipeline):
                # Transform background data through pipeline steps (except the final estimator)
                X_transformed = X_background
                for step_name, step_transformer in model.steps[:-1]:
                    X_transformed = step_transformer.transform(X_transformed)
                # Use the transformed data and final estimator for LinearExplainer
                explainer = shap.LinearExplainer(final_estimator, X_transformed)
            else:
                explainer = shap.LinearExplainer(model, X_background)
        else:
            # Fallback to KernelExplainer (slower but works for any model)
            # This works with the full pipeline
            explainer = shap.KernelExplainer(model.predict_proba, X_background)
    except Exception as exc:
        explainer_error = exc
        logger.warning("Primary explainer failed for %s: %s, trying KernelExplainer fallback", model_key, exc)
        # Fallback to KernelExplainer for any model that fails with other explainers
        try:
            # Use smaller background for KernelExplainer to improve performance
            sample_size = min(50, len(X_background))
            if isinstance(X_background, pd.DataFrame):
                X_background_small = X_background.sample(n=sample_size, random_state=42)
            else:
                X_background_small = X_background[:sample_size]
            explainer = shap.KernelExplainer(model.predict_proba, X_background_small)
            logger.info("Using KernelExplainer fallback for %s", model_key)
        except Exception as fallback_exc:
            logger.error("KernelExplainer fallback also failed for %s: %s", model_key, fallback_exc)
            raise fallback_exc
    
    if explainer is None:
        raise RuntimeError(f"Failed to create explainer for {model_key}: {explainer_error}")
    
    return explainer


def _extract_shap_values(shap_values) -> tuple[float | None, np.ndarray]:
    """Extract base value and SHAP values from explainer output."""
    base_value = None
    
    if hasattr(shap_values, "values"):
        values = shap_values.values[0] if hasattr(shap_values.values, "__getitem__") else shap_values.values
        
        # Handle base_values - it might be an array or scalar
        if hasattr(shap_values, "base_values"):
            base_values = shap_values.base_values
            try:
                # Check if base_values[0] exists and what type it is
                if isinstance(base_values, np.ndarray):
                    if base_values.ndim == 0:
                        # Scalar array
                        base_value = float(base_values.item())
                    elif base_values.ndim == 1:
                        # 1D array: [base_0, base_1] for binary classification
                        # Take the positive class base value (index 1) if available, else index 0
                        base_value = float(base_values[1] if len(base_values) > 1 else base_values[0])
                    else:
                        # Multi-dimensional array, get first element and check if it's an array
                        first_elem = base_values[0]
                        if isinstance(first_elem, np.ndarray):
                            # Nested array case
                            if first_elem.size == 1:
                                base_value = float(first_elem.item())
                            else:
                                base_value = float(first_elem[1] if len(first_elem) > 1 else first_elem[0])
                        else:
                            base_value = float(first_elem)
                elif isinstance(base_values, (list, tuple)):
                    if len(base_values) > 0:
                        first_elem = base_values[0]
                        if isinstance(first_elem, (list, tuple, np.ndarray)):
                            base_value = float(first_elem[1] if len(first_elem) > 1 else first_elem[0])
                        else:
                            base_value = float(base_values[1] if len(base_values) > 1 else first_elem)
                    else:
                        base_value = None
                else:
                    # Scalar
                    base_value = float(base_values) if base_values is not None else None
            except (IndexError, TypeError, ValueError) as exc:
                logger.warning("Could not extract base_value: %s", exc)
                base_value = None
    else:
        # For KernelExplainer or when shap_values is a list
        if isinstance(shap_values, list) and len(shap_values) > 0:
            first_item = shap_values[0]
            if hasattr(first_item, "values"):
                values = first_item.values[0] if hasattr(first_item.values, "__getitem__") else first_item.values
            else:
                values = first_item
                
            if hasattr(first_item, "base_values"):
                base_vals = first_item.base_values
                if isinstance(base_vals, np.ndarray):
                    base_value = float(base_vals[1] if len(base_vals) > 1 else base_vals[0])
                else:
                    base_value = float(base_vals[1] if isinstance(base_vals, (list, tuple)) and len(base_vals) > 1 else base_vals[0] if isinstance(base_vals, (list, tuple)) else base_vals) if base_vals is not None else None
        else:
            values = shap_values.values if hasattr(shap_values, "values") else shap_values
            if hasattr(shap_values, "base_values"):
                base_vals = shap_values.base_values
                if isinstance(base_vals, np.ndarray):
                    base_value = float(base_vals[1] if len(base_vals) > 1 else base_vals[0])
                else:
                    base_value = float(base_vals[1] if isinstance(base_vals, (list, tuple)) and len(base_vals) > 1 else base_vals[0] if isinstance(base_vals, (list, tuple)) else base_vals) if base_vals is not None else None
    
    return base_value, values

