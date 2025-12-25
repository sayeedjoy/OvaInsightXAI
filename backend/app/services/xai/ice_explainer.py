"""Individual Conditional Expectation (ICE) computation."""

from __future__ import annotations

import logging
from typing import Any

from sklearn.inspection import partial_dependence

from app.model.registry import MODEL_REGISTRY
from app.services.xai.utils import generate_training_data, get_model_and_features

logger = logging.getLogger(__name__)


def compute_ice_1d(
    model_key: str,
    feature_index: int | None = None,
    n_grid_points: int = 50,
    n_samples: int = 30
) -> dict[str, Any]:
    """Compute 1D Individual Conditional Expectation for all features or a specific feature."""
    try:
        model, _ = get_model_and_features(model_key, [0.0] * len(MODEL_REGISTRY[model_key].feature_order))
        config = MODEL_REGISTRY[model_key]

        # Generate background data (use fewer samples for ICE)
        X_background, _ = generate_training_data(model_key, n_samples=min(n_samples, 100))

        feature_names = config.feature_order
        results = []

        # If feature_index is specified, compute only for that feature
        features_to_compute = [feature_index] if feature_index is not None else range(len(feature_names))

        for idx in features_to_compute:
            try:
                ice_result = partial_dependence(
                    model,
                    X_background,
                    features=[idx],
                    grid_resolution=min(n_grid_points, 50),
                    kind="individual"
                )

                grid_values = ice_result["grid_values"][0].tolist()
                individual_predictions = ice_result["individual"][0].tolist()  # Shape: (n_samples, n_grid_points)

                # Limit number of curves for performance
                max_curves = 30
                if len(individual_predictions) > max_curves:
                    step = len(individual_predictions) // max_curves
                    individual_predictions = individual_predictions[::step]

                curves = [
                    {
                        "sample_index": i,
                        "predictions": pred.tolist() if hasattr(pred, "tolist") else pred,
                    }
                    for i, pred in enumerate(individual_predictions)
                ]

                results.append({
                    "feature": feature_names[idx],
                    "feature_index": idx,
                    "grid_values": grid_values,
                    "curves": curves,
                })
            except Exception as exc:
                logger.warning("Error computing ICE for feature %s: %s", feature_names[idx], exc)
                continue

        return {"ice_plots": results}
    except Exception as exc:
        logger.error("Error computing ICE: %s", exc, exc_info=True)
        return {"error": str(exc)}

