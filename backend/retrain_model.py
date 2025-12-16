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
import sklearn

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
    
    # Determine model path - use absolute path for Docker compatibility
    # In Docker, __file__ will be /app/retrain_model.py, so parent is /app
    # Model should be at /app/app/model/model.pkl
    base_path = Path(__file__).parent.resolve()
    model_path = base_path / "app" / "model" / "model.pkl"
    model_path = model_path.resolve()  # Ensure absolute path
    model_dir = model_path.parent
    
    # Ensure model directory exists
    print(f"Ensuring model directory exists: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove any existing old model to ensure we use the new one
    if model_path.exists():
        print(f"Removing old model file at {model_path}...")
        model_path.unlink()
    
    # Log sklearn version for debugging
    print(f"scikit-learn version: {sklearn.__version__}")
    
    # Save the model
    print(f"Saving model to {model_path} (absolute: {model_path.resolve()})...")
    joblib.dump(pipeline, model_path)
    
    # Verify the model file was created
    if not model_path.exists():
        raise FileNotFoundError(f"Model file was not created at {model_path}")
    
    file_size = model_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Model file was created but is empty (0 bytes) at {model_path}")
    
    print(f"Model file created successfully (size: {file_size} bytes)")
    print(f"Model file absolute path: {model_path.resolve()}")
    print(f"Model file is readable: {model_path.is_file()}")
    
    # Verify the model works by loading and testing it
    print("Verifying model...")
    test_sample = X[:1]
    prediction = pipeline.predict(test_sample)
    proba = pipeline.predict_proba(test_sample)
    print(f"Test prediction: {prediction[0]}, confidence: {max(proba[0]):.3f}")
    
    # Verify we can load the saved model
    print("Verifying saved model can be loaded...")
    loaded_model = joblib.load(model_path)
    loaded_prediction = loaded_model.predict(test_sample)
    loaded_proba = loaded_model.predict_proba(test_sample)
    print(f"Loaded model test prediction: {loaded_prediction[0]}, confidence: {max(loaded_proba[0]):.3f}")
    
    if prediction[0] != loaded_prediction[0]:
        raise ValueError("Saved model produces different predictions than original model")
    
    print("Done! Model saved and verified successfully.")


if __name__ == "__main__":
    train_and_save_model()

