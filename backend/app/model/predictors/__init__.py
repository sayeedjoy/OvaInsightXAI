"""Predictors package - provides prediction functions for different model types."""

from app.model.predictors.sklearn_predictor import predict_tabular
from app.model.predictors.image_predictor import predict_image

__all__ = [
    "predict_tabular",
    "predict_image",
]
