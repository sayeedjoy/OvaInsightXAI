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
    """Create synthetic training data with balanced classes and clear separation."""
    np.random.seed(random_state)
    
    # Create balanced dataset: 50% positive, 50% negative
    n_positive = n_samples // 2
    n_negative = n_samples - n_positive
    
    # Generate NEGATIVE cases (healthy patients)
    negative_data = {
        "age": np.random.normal(45, 12, n_negative).clip(20, 75),
        "alb": np.random.normal(4.2, 0.3, n_negative).clip(3.5, 5.0),  # Normal albumin
        "alp": np.random.normal(70, 20, n_negative).clip(30, 120),  # Normal ALP
        "bun": np.random.normal(14, 3, n_negative).clip(7, 20),
        "ca125": np.random.normal(18, 8, n_negative).clip(5, 34),  # Normal CA125 < 35
        "eo_abs": np.random.normal(0.2, 0.08, n_negative).clip(0.05, 0.5),
        "ggt": np.random.normal(25, 10, n_negative).clip(5, 40),  # Normal GGT
        "he4": np.random.normal(50, 12, n_negative).clip(30, 70),  # Normal HE4 < 70
        "mch": np.random.normal(29, 1.5, n_negative).clip(27, 32),
        "mono_abs": np.random.normal(0.5, 0.12, n_negative).clip(0.2, 0.8),
        "na": np.random.normal(140, 2, n_negative).clip(136, 145),  # Normal sodium
        "pdw": np.random.normal(11, 1.2, n_negative).clip(9, 14),
    }
    
    # Generate POSITIVE cases (cancer patients) with elevated markers
    positive_data = {
        "age": np.random.normal(62, 10, n_positive).clip(40, 85),  # Older patients
        "alb": np.random.normal(3.2, 0.4, n_positive).clip(2.5, 3.8),  # Low albumin
        "alp": np.random.normal(160, 50, n_positive).clip(100, 300),  # Elevated ALP
        "bun": np.random.normal(20, 5, n_positive).clip(12, 35),
        "ca125": np.random.exponential(150, n_positive).clip(50, 500) + 35,  # Elevated CA125 > 35
        "eo_abs": np.random.normal(0.35, 0.15, n_positive).clip(0.1, 0.7),
        "ggt": np.random.normal(100, 40, n_positive).clip(45, 200),  # Elevated GGT
        "he4": np.random.exponential(120, n_positive).clip(80, 500) + 70,  # Elevated HE4 > 70
        "mch": np.random.normal(27, 2, n_positive).clip(24, 30),  # Slightly low
        "mono_abs": np.random.normal(0.7, 0.2, n_positive).clip(0.3, 1.1),
        "na": np.random.normal(136, 3, n_positive).clip(130, 140),  # Lower sodium
        "pdw": np.random.normal(15, 2, n_positive).clip(12, 20),  # Elevated PDW
    }
    
    # Combine negative and positive data
    X_negative = np.column_stack([negative_data[f] for f in FEATURE_NAMES])
    X_positive = np.column_stack([positive_data[f] for f in FEATURE_NAMES])
    
    X = np.vstack([X_negative, X_positive])
    y = np.concatenate([np.zeros(n_negative), np.ones(n_positive)]).astype(int)
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
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
    sklearn_version = sklearn.__version__
    print(f"scikit-learn version: {sklearn_version}")
    
    # Save the model with metadata including sklearn version
    # Wrap the pipeline in a dict with metadata for version verification
    model_with_metadata = {
        "pipeline": pipeline,
        "sklearn_version": sklearn_version,
        "model_type": "LogisticRegression",
        "trained_at": __import__("datetime").datetime.now().isoformat()
    }
    
    print(f"Saving model to {model_path} (absolute: {model_path.resolve()})...")
    print(f"Model trained with sklearn version: {sklearn_version}")
    joblib.dump(model_with_metadata, model_path)
    
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
    loaded_artifact = joblib.load(model_path)
    
    # Extract the pipeline from metadata dict
    if isinstance(loaded_artifact, dict) and "pipeline" in loaded_artifact:
        loaded_model = loaded_artifact["pipeline"]
        loaded_sklearn_version = loaded_artifact.get("sklearn_version", "unknown")
        print(f"Loaded model was trained with sklearn version: {loaded_sklearn_version}")
        if loaded_sklearn_version != sklearn_version:
            print(f"WARNING: Model sklearn version ({loaded_sklearn_version}) differs from current ({sklearn_version})")
    else:
        # Fallback for models saved without metadata
        loaded_model = loaded_artifact
        print("WARNING: Model does not contain version metadata")
    
    loaded_prediction = loaded_model.predict(test_sample)
    loaded_proba = loaded_model.predict_proba(test_sample)
    print(f"Loaded model test prediction: {loaded_prediction[0]}, confidence: {max(loaded_proba[0]):.3f}")
    
    if prediction[0] != loaded_prediction[0]:
        raise ValueError("Saved model produces different predictions than original model")
    
    print("Done! Model saved and verified successfully.")


if __name__ == "__main__":
    train_and_save_model()

