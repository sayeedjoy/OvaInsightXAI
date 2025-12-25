"""XAI (Explainable AI) service for computing model explanations."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.inspection import partial_dependence

from app.model import predictor
from app.model.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)

# Cache for synthetic training data per model
_TRAINING_DATA_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _get_model_and_features(model_key: str, instance_features: list[float]) -> tuple[Any, np.ndarray]:
    """Get the model and prepare feature array."""
    predictor.ensure_model_loaded(model_key)
    model = predictor.get_model(model_key)
    instance_array = np.array(instance_features).reshape(1, -1)
    return model, instance_array


def _generate_training_data(model_key: str, n_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data for XAI methods that need background data."""
    if model_key in _TRAINING_DATA_CACHE:
        return _TRAINING_DATA_CACHE[model_key]

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

    _TRAINING_DATA_CACHE[model_key] = (X, y)
    return X, y


def compute_shap_explanation(
    model_key: str,
    instance_features: list[float],
    background_samples: int = 100
) -> dict[str, Any]:
    """Compute SHAP values for the prediction."""
    try:
        import shap
    except ImportError:
        logger.error("SHAP library not available")
        return {"error": "SHAP library not installed"}

    try:
        model, instance_array = _get_model_and_features(model_key, instance_features)
        config = MODEL_REGISTRY[model_key]

        # Get background data
        X_background, _ = _generate_training_data(model_key, n_samples=background_samples)

        # Extract the final estimator from Pipeline if needed
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            # Get the final step (classifier/estimator)
            final_step_name, final_estimator = model.steps[-1]
            classifier_type = type(final_estimator).__name__
            logger.debug("Extracted classifier from Pipeline: %s (type: %s)", final_step_name, classifier_type)
        else:
            final_estimator = model
            classifier_type = type(model).__name__

        # Use appropriate explainer
        # For TreeExplainer and LinearExplainer, we need to use the pipeline for predictions
        # but can use the final estimator for explainer initialization
        if "Tree" in classifier_type or "XGB" in classifier_type or "LGBM" in classifier_type:
            # TreeExplainer can work with the pipeline directly
            explainer = shap.TreeExplainer(model, X_background)
        elif "Linear" in classifier_type or "Logistic" in classifier_type:
            # LinearExplainer needs the actual estimator, not the pipeline
            # But we need to transform the background data through the pipeline steps first
            if isinstance(model, Pipeline):
                # Transform background data through pipeline steps (except the final estimator)
                X_transformed = X_background
                for step_name, step_transformer in model.steps[:-1]:
                    X_transformed = step_transformer.transform(X_transformed)
                # Use the transformed data and final estimator for LinearExplainer
                explainer = shap.LinearExplainer(final_estimator, X_transformed)
            else:
                explainer = shap.LinearExplainer(model, X_background)
        else:
            # Fallback to KernelExplainer (slower but works for any model)
            # This works with the full pipeline
            explainer = shap.KernelExplainer(model.predict_proba, X_background)

        # For LinearExplainer with Pipeline, we need to transform the instance too
        if isinstance(model, Pipeline) and ("Linear" in classifier_type or "Logistic" in classifier_type):
            # Transform instance through pipeline steps (except the final estimator)
            instance_transformed = instance_array
            for step_name, step_transformer in model.steps[:-1]:
                instance_transformed = step_transformer.transform(instance_transformed)
            shap_values = explainer(instance_transformed)
        else:
            # For other explainers, use the original instance array
            shap_values = explainer(instance_array)

        # Extract values
        if hasattr(shap_values, "values"):
            values = shap_values.values[0]
            base_value = float(shap_values.base_values[0]) if hasattr(shap_values, "base_values") else None
        else:
            # For KernelExplainer or when shap_values is a list
            if isinstance(shap_values, list) and len(shap_values) > 0:
                values = shap_values[0].values if hasattr(shap_values[0], "values") else shap_values[0]
                base_value = float(shap_values[0].base_values) if hasattr(shap_values[0], "base_values") else None
            else:
                values = shap_values.values if hasattr(shap_values, "values") else shap_values
                base_value = float(shap_values.base_values) if hasattr(shap_values, "base_values") else None

        # Ensure values is a numpy array
        import numpy as np
        if not isinstance(values, np.ndarray):
            values = np.array(values)

        # For binary classification, get values for the positive class
        if len(values.shape) > 1:
            values = values[:, 1] if values.shape[1] > 1 else values[:, 0]
        elif len(values.shape) == 0:
            # Scalar value, convert to array
            values = np.array([values])

        feature_names = config.feature_order
        contributions = [
            {
                "feature": feature_names[i],
                "value": float(instance_features[i]),
                "shap_value": float(values[i]) if i < len(values) else 0.0,
            }
            for i in range(len(feature_names))
        ]

        return {
            "base_value": base_value,
            "contributions": contributions,
            "prediction": float(model.predict_proba(instance_array)[0][1]) if hasattr(model, "predict_proba") else None,
        }
    except Exception as exc:
        logger.error("Error computing SHAP explanation: %s", exc, exc_info=True)
        return {"error": str(exc)}


def compute_lime_explanation(
    model_key: str,
    instance_features: list[float],
    num_features: int = 10
) -> dict[str, Any]:
    """Compute LIME explanation for the prediction."""
    try:
        from lime import lime_tabular
    except ImportError:
        logger.error("LIME library not available")
        return {"error": "LIME library not installed"}

    try:
        model, instance_array = _get_model_and_features(model_key, instance_features)
        config = MODEL_REGISTRY[model_key]

        # Generate training data for LIME
        X_train, _ = _generate_training_data(model_key, n_samples=1000)

        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=config.feature_order,
            mode="classification",
            random_state=42
        )

        # Explain the instance
        explanation = explainer.explain_instance(
            instance_array[0],
            model.predict_proba if hasattr(model, "predict_proba") else model.predict,
            num_features=num_features,
            top_labels=1
        )

        # Extract explanation
        exp_list = explanation.as_list()
        feature_importance = [
            {
                "feature": item[0],
                "importance": float(item[1]),
            }
            for item in exp_list
        ]

        return {
            "feature_importance": feature_importance,
            "prediction": float(model.predict_proba(instance_array)[0][1]) if hasattr(model, "predict_proba") else None,
        }
    except Exception as exc:
        logger.error("Error computing LIME explanation: %s", exc, exc_info=True)
        return {"error": str(exc)}


def compute_pdp_1d(
    model_key: str,
    feature_index: int | None = None,
    n_grid_points: int = 50
) -> dict[str, Any]:
    """Compute 1D Partial Dependence Plot for all features or a specific feature."""
    try:
        model, _ = _get_model_and_features(model_key, [0.0] * len(MODEL_REGISTRY[model_key].feature_order))
        config = MODEL_REGISTRY[model_key]

        # Generate background data
        X_background, _ = _generate_training_data(model_key, n_samples=500)

        feature_names = config.feature_order
        results = []

        # If feature_index is specified, compute only for that feature
        features_to_compute = [feature_index] if feature_index is not None else range(len(feature_names))

        for idx in features_to_compute:
            try:
                pdp_result = partial_dependence(
                    model,
                    X_background,
                    features=[idx],
                    grid_resolution=min(n_grid_points, 50),
                    kind="average"
                )

                grid_values = pdp_result["grid_values"][0].tolist()
                average_predictions = pdp_result["average"][0].tolist()

                results.append({
                    "feature": feature_names[idx],
                    "feature_index": idx,
                    "grid_values": grid_values,
                    "predictions": average_predictions,
                })
            except Exception as exc:
                logger.warning("Error computing PDP for feature %s: %s", feature_names[idx], exc)
                continue

        return {"pdp_plots": results}
    except Exception as exc:
        logger.error("Error computing PDP: %s", exc, exc_info=True)
        return {"error": str(exc)}


def compute_ice_1d(
    model_key: str,
    feature_index: int | None = None,
    n_grid_points: int = 50,
    n_samples: int = 30
) -> dict[str, Any]:
    """Compute 1D Individual Conditional Expectation for all features or a specific feature."""
    try:
        model, _ = _get_model_and_features(model_key, [0.0] * len(MODEL_REGISTRY[model_key].feature_order))
        config = MODEL_REGISTRY[model_key]

        # Generate background data (use fewer samples for ICE)
        X_background, _ = _generate_training_data(model_key, n_samples=min(n_samples, 100))

        feature_names = config.feature_order
        results = []

        # If feature_index is specified, compute only for that feature
        features_to_compute = [feature_index] if feature_index is not None else range(len(feature_names))

        for idx in features_to_compute:
            try:
                ice_result = partial_dependence(
                    model,
                    X_background,
                    features=[idx],
                    grid_resolution=min(n_grid_points, 50),
                    kind="individual"
                )

                grid_values = ice_result["grid_values"][0].tolist()
                individual_predictions = ice_result["individual"][0].tolist()  # Shape: (n_samples, n_grid_points)

                # Limit number of curves for performance
                max_curves = 30
                if len(individual_predictions) > max_curves:
                    step = len(individual_predictions) // max_curves
                    individual_predictions = individual_predictions[::step]

                curves = [
                    {
                        "sample_index": i,
                        "predictions": pred.tolist() if hasattr(pred, "tolist") else pred,
                    }
                    for i, pred in enumerate(individual_predictions)
                ]

                results.append({
                    "feature": feature_names[idx],
                    "feature_index": idx,
                    "grid_values": grid_values,
                    "curves": curves,
                })
            except Exception as exc:
                logger.warning("Error computing ICE for feature %s: %s", feature_names[idx], exc)
                continue

        return {"ice_plots": results}
    except Exception as exc:
        logger.error("Error computing ICE: %s", exc, exc_info=True)
        return {"error": str(exc)}


def compute_ale_1d(
    model_key: str,
    feature_index: int | None = None,
    n_bins: int = 20
) -> dict[str, Any]:
    """Compute 1D Accumulated Local Effects for all features or a specific feature."""
    try:
        model, _ = _get_model_and_features(model_key, [0.0] * len(MODEL_REGISTRY[model_key].feature_order))
        config = MODEL_REGISTRY[model_key]

        # Generate background data
        X_background, _ = _generate_training_data(model_key, n_samples=500)

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

