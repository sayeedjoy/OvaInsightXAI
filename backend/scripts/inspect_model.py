#!/usr/bin/env python3
"""
Inspect a saved sklearn/joblib model artifact and print its expected feature count.

Usage:
  python scripts/inspect_model.py
  MODEL_PATH=/opt/models/model.pkl python scripts/inspect_model.py
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib


def _extract_estimator(artifact):
    if hasattr(artifact, "predict"):
        return artifact
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact["pipeline"]
    for key in ("model", "estimator", "classifier", "meta_logreg"):
        if isinstance(artifact, dict) and key in artifact and hasattr(artifact[key], "predict"):
            return artifact[key]
    return None


def main() -> int:
    model_path = Path(os.getenv("MODEL_PATH", "/app/app/model/model.pkl"))
    print(f"MODEL_PATH={model_path}")
    print(f"exists={model_path.exists()}")
    if model_path.exists():
        print(f"size_bytes={model_path.stat().st_size}")

    artifact = joblib.load(model_path)
    est = _extract_estimator(artifact)
    print(f"artifact_type={type(artifact).__name__}")
    print(f"estimator_type={type(est).__name__ if est is not None else None}")

    n_features = getattr(est, "n_features_in_", None) if est is not None else None
    print(f"n_features_in_={n_features}")

    if isinstance(artifact, dict) and "sklearn_version" in artifact:
        print(f"trained_sklearn_version={artifact.get('sklearn_version')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


