"""Partial Dependence Plot (PDP) computation."""

from __future__ import annotations

import logging
from typing import Any

from sklearn.inspection import partial_dependence

from app.model.registry import MODEL_REGISTRY
from app.services.xai.utils import generate_training_data, get_model_and_features

logger = logging.getLogger(__name__)


def compute_pdp_1d(
    model_key: str,
    feature_index: int | None = None,
    n_grid_points: int = 50
) -> dict[str, Any]:
    """Compute 1D Partial Dependence Plot for all features or a specific feature."""
    try:
        model, _ = get_model_and_features(model_key, [0.0] * len(MODEL_REGISTRY[model_key].feature_order))
        config = MODEL_REGISTRY[model_key]

        # Generate background data (reduced from 500 to 300 for better performance)
        X_background, _ = generate_training_data(model_key, n_samples=300)

        feature_names = config.feature_order
        results = []

        # If feature_index is specified, compute only for that feature
        features_to_compute = [feature_index] if feature_index is not None else range(len(feature_names))

        for idx in features_to_compute:
            try:
                pdp_result = partial_dependence(
                    model,
                    X_background,
                    features=[idx],
                    grid_resolution=min(n_grid_points, 50),
                    kind="average"
                )

                grid_values = pdp_result["grid_values"][0].tolist()
                average_predictions = pdp_result["average"][0].tolist()

                results.append({
                    "feature": feature_names[idx],
                    "feature_index": idx,
                    "grid_values": grid_values,
                    "predictions": average_predictions,
                })
            except Exception as exc:
                logger.warning("Error computing PDP for feature %s: %s", feature_names[idx], exc)
                continue

        return {"pdp_plots": results}
    except Exception as exc:
        logger.error("Error computing PDP: %s", exc, exc_info=True)
        return {"error": str(exc)}

