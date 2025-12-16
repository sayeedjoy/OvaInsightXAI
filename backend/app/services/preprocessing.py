"""Feature preprocessing utilities."""

from __future__ import annotations

import logging
from typing import List

from app.schemas.input_schema import PredictionRequest
from app.utils.config import ordered_feature_vector

logger = logging.getLogger(__name__)


def request_to_features(request: PredictionRequest) -> List[float]:
    """
    Convert a PredictionRequest into an ordered feature vector.

    Additional preprocessing hooks (scaling, clipping, etc.) can be added here
    later without touching the route or model code.
    """

    payload = request.model_dump()
    feature_vector = ordered_feature_vector(payload)
    logger.debug("Prepared feature vector: %s", feature_vector)
    return feature_vector


