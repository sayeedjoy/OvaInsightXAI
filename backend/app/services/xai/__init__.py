"""XAI (Explainable AI) service package for computing model explanations."""

from __future__ import annotations

from typing import Any

from app.services.xai.ale_explainer import compute_ale_1d
from app.services.xai.ice_explainer import compute_ice_1d
from app.services.xai.lime_explainer import compute_lime_explanation
from app.services.xai.pdp_explainer import compute_pdp_1d
from app.services.xai.shap_explainer import compute_shap_explanation

__all__ = [
    "compute_shap_explanation",
    "compute_lime_explanation",
    "compute_pdp_1d",
    "compute_ice_1d",
    "compute_ale_1d",
    "compute_all_xai_explanations",
]


def compute_all_xai_explanations(
    model_key: str,
    instance_features: list[float]
) -> dict[str, Any]:
    """Compute all XAI explanations for a prediction instance."""
    return {
        "shap": compute_shap_explanation(model_key, instance_features),
        "lime": compute_lime_explanation(model_key, instance_features),
        "pdp_1d": compute_pdp_1d(model_key),
        "ice_1d": compute_ice_1d(model_key),
        "ale_1d": compute_ale_1d(model_key),
    }

