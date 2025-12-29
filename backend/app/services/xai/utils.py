"""Utility functions for XAI services."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.model import predictor
from app.model.registry import MODEL_REGISTRY

# Cache for synthetic training data per model
_TRAINING_DATA_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def get_model_and_features(model_key: str, instance_features: list[float]) -> tuple[Any, np.ndarray]:
    """Get the model and prepare feature array."""
    predictor.ensure_model_loaded(model_key)
    model = predictor.get_model(model_key)
    instance_array = np.array(instance_features).reshape(1, -1)
    return model, instance_array


def generate_training_data(
    model_key: str, 
    n_samples: int = 1000, 
    return_dataframe: bool = False
) -> tuple[np.ndarray | pd.DataFrame, np.ndarray]:
    """Generate synthetic training data for XAI methods that need background data.
    
    Args:
        model_key: The model key to generate data for
        n_samples: Number of samples to generate
        return_dataframe: If True, return pandas DataFrame with feature names instead of numpy array
    """
    cache_key = f"{model_key}_{return_dataframe}"
    if cache_key in _TRAINING_DATA_CACHE:
        return _TRAINING_DATA_CACHE[cache_key]

    config = MODEL_REGISTRY[model_key]
    feature_order = config.feature_order
    n_features = len(feature_order)

    # Generate synthetic data based on reasonable feature ranges
    # These ranges are based on typical medical biomarker values
    np.random.seed(42)
    
    if model_key == "ovarian":
        # Ovarian cancer model features
        X = np.column_stack([
            np.random.normal(50, 15, n_samples).clip(20, 85),  # age
            np.random.normal(4.0, 0.5, n_samples).clip(2.5, 5.5),  # alb
            np.random.normal(100, 40, n_samples).clip(30, 300),  # alp
            np.random.normal(15, 5, n_samples).clip(5, 40),  # bun
            np.random.exponential(50, n_samples).clip(5, 500) + 20,  # ca125
            np.random.normal(0.3, 0.15, n_samples).clip(0.05, 1.0),  # eo_abs
            np.random.exponential(40, n_samples).clip(5, 300),  # ggt
            np.random.exponential(80, n_samples).clip(30, 500) + 50,  # he4
            np.random.normal(28, 2, n_samples).clip(24, 33),  # mch
            np.random.normal(0.6, 0.2, n_samples).clip(0.2, 1.2),  # mono_abs
            np.random.normal(140, 3, n_samples).clip(130, 150),  # na
            np.random.normal(12, 2, n_samples).clip(9, 20),  # pdw
        ])
    elif model_key == "hepatitis_b":
        # Hepatitis B model features
        X = np.column_stack([
            np.random.normal(45, 15, n_samples).clip(18, 80),  # Age
            np.random.binomial(1, 0.6, n_samples),  # Sex
            np.random.binomial(1, 0.4, n_samples),  # Fatigue
            np.random.binomial(1, 0.3, n_samples),  # Malaise
            np.random.binomial(1, 0.3, n_samples),  # Liver_big
            np.random.binomial(1, 0.2, n_samples),  # Spleen_palpable
            np.random.binomial(1, 0.2, n_samples),  # Spiders
            np.random.binomial(1, 0.15, n_samples),  # Ascites
            np.random.binomial(1, 0.1, n_samples),  # Varices
            np.random.exponential(2, n_samples).clip(0.1, 10),  # Bilirubin
            np.random.normal(100, 40, n_samples).clip(30, 300),  # Alk_phosphate
            np.random.exponential(50, n_samples).clip(10, 300),  # Sgot
            np.random.normal(3.8, 0.6, n_samples).clip(2.0, 5.5),  # Albumin
            np.random.normal(12, 2, n_samples).clip(8, 20),  # Protime
            np.random.binomial(1, 0.3, n_samples),  # Histology
        ])
    elif model_key == "pcos":
        # PCOS model features
        X = np.column_stack([
            np.random.normal(5, 3, n_samples).clip(0, 15),  # Marraige Status (Yrs)
            np.random.binomial(1, 0.5, n_samples),  # Cycle(R/I)
            np.random.normal(75, 10, n_samples).clip(50, 100),  # Pulse rate(bpm)
            np.random.normal(8, 3, n_samples).clip(2, 20),  # FSH(mIU/mL)
            np.random.normal(28, 6, n_samples).clip(18, 45),  # Age (yrs)
            np.random.normal(8, 4, n_samples).clip(0, 20),  # Follicle No. (L)
            np.random.normal(26, 5, n_samples).clip(18, 40),  # BMI
            np.random.binomial(1, 0.3, n_samples),  # Skin darkening (Y/N)
            np.random.normal(5, 3, n_samples).clip(0, 20),  # II beta-HCG(mIU/mL)
            np.random.normal(80, 10, n_samples).clip(60, 100),  # BP _Diastolic (mmHg)
            np.random.binomial(1, 0.4, n_samples),  # hair growth(Y/N)
            np.random.normal(8, 2, n_samples).clip(4, 15),  # Avg. F size (L) (mm)
            np.random.normal(8, 2, n_samples).clip(4, 15),  # Avg. F size (R) (mm)
            np.random.normal(0.85, 0.1, n_samples).clip(0.7, 1.1),  # Waist:Hip Ratio
            np.random.normal(65, 12, n_samples).clip(45, 100),  # Weight (Kg)
            np.random.binomial(1, 0.5, n_samples),  # Weight gain(Y/N)
            np.random.normal(10, 4, n_samples).clip(2, 25),  # LH(mIU/mL)
            np.random.normal(8, 4, n_samples).clip(0, 20),  # Follicle No. (R)
            np.random.normal(38, 4, n_samples).clip(30, 50),  # Hip(inch)
            np.random.normal(32, 4, n_samples).clip(24, 44),  # Waist(inch)
        ])
    else:
        # Generic fallback: uniform distribution
        X = np.random.uniform(0, 100, (n_samples, n_features))

    # Generate synthetic labels (binary classification)
    # Use a simple rule based on some features to create realistic labels
    if model_key == "ovarian":
        # Higher CA125 and HE4 increase probability
        y = ((X[:, 4] > 35) | (X[:, 7] > 70)).astype(int)
    elif model_key == "hepatitis_b":
        # Higher bilirubin and lower albumin increase probability
        y = ((X[:, 9] > 2) | (X[:, 12] < 3.5)).astype(int)
    elif model_key == "pcos":
        # Higher BMI and irregular cycle increase probability
        y = ((X[:, 6] > 25) | (X[:, 1] == 1)).astype(int)
    else:
        y = np.random.binomial(1, 0.3, n_samples)

    # Convert to DataFrame with feature names if requested
    if return_dataframe:
        X_df = pd.DataFrame(X, columns=feature_order)
        _TRAINING_DATA_CACHE[cache_key] = (X_df, y)
        return X_df, y
    else:
        _TRAINING_DATA_CACHE[cache_key] = (X, y)
        return X, y


def is_image_model(model_key: str) -> bool:
    """Check if a model is image-based (PyTorch) or tabular (sklearn).
    
    Args:
        model_key: The model key to check
    
    Returns:
        True if the model is image-based, False if tabular
    """
    return model_key == "brain_tumor"


def get_image_model(model_key: str) -> Any:
    """Get the loaded image model.
    
    Args:
        model_key: The model key (e.g., "brain_tumor")
    
    Returns:
        The loaded PyTorch model
    """
    predictor.ensure_model_loaded(model_key)
    return predictor.get_model(model_key)
