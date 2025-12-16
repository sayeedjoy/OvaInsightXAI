"""
Re-train and save the model with the current scikit-learn version.

This script creates a simple LogisticRegression pipeline with SimpleImputer
and StandardScaler to match the expected model structure for ovarian cancer prediction.

Run this script once to regenerate model.pkl compatible with your sklearn version:
    python retrain_model.py
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

# Feature names (12 biomarkers for ovarian cancer prediction)
FEATURE_NAMES = [
    "age",
    "alb",
    "alp",
    "bun",
    "ca125",
    "eo_abs",
    "ggt",
    "he4",
    "mch",
    "mono_abs",
    "na",
    "pdw",
]

def create_synthetic_data(n_samples=1000, random_state=42):
    """Create synthetic training data for demonstration."""
    np.random.seed(random_state)
    
    # Generate synthetic feature data with realistic ranges
    data = {
        "age": np.random.normal(55, 15, n_samples).clip(20, 90),
        "alb": np.random.normal(4.0, 0.5, n_samples).clip(2.5, 5.5),
        "alp": np.random.normal(70, 25, n_samples).clip(30, 150),
        "bun": np.random.normal(15, 5, n_samples).clip(5, 40),
        "ca125": np.random.exponential(50, n_samples).clip(5, 500),
        "eo_abs": np.random.normal(0.2, 0.1, n_samples).clip(0, 0.8),
        "ggt": np.random.normal(30, 20, n_samples).clip(5, 150),
        "he4": np.random.exponential(70, n_samples).clip(20, 500),
        "mch": np.random.normal(29, 3, n_samples).clip(20, 38),
        "mono_abs": np.random.normal(0.5, 0.2, n_samples).clip(0.1, 1.2),
        "na": np.random.normal(140, 3, n_samples).clip(130, 150),
        "pdw": np.random.normal(12, 2, n_samples).clip(8, 20),
    }
    
    X = np.column_stack([data[f] for f in FEATURE_NAMES])
    
    # Create synthetic labels based on key biomarkers (CA125 and HE4 are important for ovarian cancer)
    # This is a simplified model for demonstration
    risk_score = (
        0.3 * (data["ca125"] > 35).astype(float) +
        0.3 * (data["he4"] > 70).astype(float) +
        0.2 * (data["age"] > 50).astype(float) +
        0.1 * (data["alb"] < 3.5).astype(float) +
        0.1 * np.random.random(n_samples)
    )
    y = (risk_score > 0.5).astype(int)
    
    return X, y


def train_and_save_model():
    """Train a new model and save it."""
    print("Creating synthetic training data...")
    X, y = create_synthetic_data(n_samples=2000)
    
    print(f"Training data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Create pipeline matching the expected structure
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    print("Training model...")
    pipeline.fit(X, y)
    
    # Save the model
    model_path = Path(__file__).parent / "app" / "model" / "model.pkl"
    print(f"Saving model to {model_path}...")
    joblib.dump(pipeline, model_path)
    
    # Verify the model works
    print("Verifying model...")
    test_sample = X[:1]
    prediction = pipeline.predict(test_sample)
    proba = pipeline.predict_proba(test_sample)
    print(f"Test prediction: {prediction[0]}, confidence: {max(proba[0]):.3f}")
    
    print("Done! Model saved successfully.")


if __name__ == "__main__":
    train_and_save_model()

