"""Feature preprocessing utilities."""

from __future__ import annotations

import logging
from typing import List, Mapping

from app.utils.config import ordered_feature_vector

logger = logging.getLogger(__name__)


def request_to_features(payload: Mapping[str, float], feature_order: List[str]) -> List[float]:
    """
    Convert a request payload into an ordered feature vector for the given schema.

    Additional preprocessing hooks (scaling, clipping, etc.) can be added here
    later without touching the route or model code.
    """

    feature_vector = ordered_feature_vector(payload, feature_order=feature_order)
    logger.debug("Prepared feature vector: %s", feature_vector)
    return feature_vector


