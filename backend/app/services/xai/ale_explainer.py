"""Accumulated Local Effects (ALE) computation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.model.registry import MODEL_REGISTRY
from app.services.xai.utils import generate_training_data, get_model_and_features

logger = logging.getLogger(__name__)


def compute_ale_1d(
    model_key: str,
    feature_index: int | None = None,
    n_bins: int = 20
) -> dict[str, Any]:
    """Compute 1D Accumulated Local Effects for all features or a specific feature."""
    try:
        model, _ = get_model_and_features(model_key, [0.0] * len(MODEL_REGISTRY[model_key].feature_order))
        config = MODEL_REGISTRY[model_key]

        # Generate background data (reduced for better performance)
        X_background, _ = generate_training_data(model_key, n_samples=200)

        feature_names = config.feature_order
        results = []

        # If feature_index is specified, compute only for that feature
        features_to_compute = [feature_index] if feature_index is not None else range(len(feature_names))

        for idx in features_to_compute:
            try:
                feature_values = X_background[:, idx]
                feature_min, feature_max = float(feature_values.min()), float(feature_values.max())

                # Create bins
                bin_edges = np.linspace(feature_min, feature_max, n_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # Compute ALE
                ale_values = []
                for i in range(len(bin_edges) - 1):
                    # Find samples in this bin
                    mask = (feature_values >= bin_edges[i]) & (feature_values < bin_edges[i + 1])
                    if i == len(bin_edges) - 2:  # Include right edge for last bin
                        mask = (feature_values >= bin_edges[i]) & (feature_values <= bin_edges[i + 1])

                    if mask.sum() == 0:
                        ale_values.append(0.0)
                        continue

                    # Create modified samples: set feature to bin boundaries
                    X_low = X_background[mask].copy()
                    X_high = X_background[mask].copy()
                    X_low[:, idx] = bin_edges[i]
                    X_high[:, idx] = bin_edges[i + 1]

                    # Get predictions
                    if hasattr(model, "predict_proba"):
                        pred_low = model.predict_proba(X_low)[:, 1]
                        pred_high = model.predict_proba(X_high)[:, 1]
                    else:
                        pred_low = model.predict(X_low)
                        pred_high = model.predict(X_high)

                    # Average difference
                    diff = np.mean(pred_high - pred_low)
                    ale_values.append(float(diff))

                # Accumulate effects
                accumulated = np.cumsum(ale_values).tolist()
                # Center around zero
                accumulated = [x - np.mean(accumulated) for x in accumulated]

                results.append({
                    "feature": feature_names[idx],
                    "feature_index": idx,
                    "bin_centers": bin_centers.tolist(),
                    "ale_values": accumulated,
                })
            except Exception as exc:
                logger.warning("Error computing ALE for feature %s: %s", feature_names[idx], exc)
                continue

        return {"ale_plots": results}
    except Exception as exc:
        logger.error("Error computing ALE: %s", exc, exc_info=True)
        return {"error": str(exc)}

