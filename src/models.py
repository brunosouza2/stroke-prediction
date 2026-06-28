"""
src/models.py

Model training, evaluation, persistence and inference for the Stroke Prediction project.

Integrante 2 — key functions you need:
    load_model(model_name)        -> returns a trained sklearn estimator
    predict(model, X)             -> returns (predictions, probabilities)

Usage example (Integrante 2):
    from src.models import load_model, predict
    from src.preprocessing import prepare_pipeline

    _, X_test, _, y_test = prepare_pipeline()
    model = load_model("logistic_regression")
    preds, probs = predict(model, X_test)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Default results directory relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train, **kwargs) -> LogisticRegression:
    """
    Train a Logistic Regression model.

    Default hyperparameters match the Phase 1 baseline.
    Pass kwargs to override any sklearn parameter (used by the GA optimizer).

    Returns a fitted LogisticRegression instance.
    """
    params = {"max_iter": 1000, "random_state": 42}
    params.update(kwargs)
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, **kwargs) -> RandomForestClassifier:
    """
    Train a Random Forest model.

    Default hyperparameters match the Phase 1 baseline.
    Pass kwargs to override any sklearn parameter (used by the GA optimizer).

    Returns a fitted RandomForestClassifier instance.
    """
    params = {"n_estimators": 100, "random_state": 42}
    params.update(kwargs)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name: str = "model") -> dict:
    """
    Evaluate a trained model and return a metrics dictionary.

    Returns:
        {
            "model": str,
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1": float,
            "confusion_matrix": [[tn, fp], [fn, tp]]
        }
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "confusion_matrix": cm,
    }

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}  <-- primary metric")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"  Confusion matrix (TN FP / FN TP): {cm}")

    return metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, model_name: str, results_dir: str = None) -> str:
    """
    Save a trained model to disk as a .joblib file.

    Args:
        model:       Trained sklearn estimator.
        model_name:  File name without extension (e.g. "logistic_regression").
        results_dir: Directory to save to. Defaults to results/.

    Returns the full path of the saved file.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{model_name}.joblib")
    joblib.dump(model, path)
    print(f"Model saved: {path}")
    return path


def load_model(model_name: str, results_dir: str = None):
    """
    Load a trained model from disk.

    Args:
        model_name:  File name without extension (e.g. "logistic_regression").
        results_dir: Directory to load from. Defaults to results/.

    Returns a fitted sklearn estimator.

    Raises FileNotFoundError if the model file does not exist.
    (Run notebooks/01_baseline.ipynb first to generate the model files.)
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    path = os.path.join(results_dir, f"{model_name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: '{path}'.\n"
            "Run notebooks/01_baseline.ipynb to train and save the baseline models."
        )
    model = joblib.load(path)
    print(f"Model loaded: {path}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model, X) -> tuple:
    """
    Run inference on a trained model.

    Args:
        model: Fitted sklearn estimator (from load_model or train_*).
        X:     Feature matrix (DataFrame or numpy array).

    Returns:
        (predictions, probabilities)
        - predictions:   numpy array of 0/1 labels
        - probabilities: numpy array of positive-class probabilities (float)
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return predictions, probabilities


# ---------------------------------------------------------------------------
# Metrics persistence
# ---------------------------------------------------------------------------

def save_metrics(metrics_list: list, filename: str = "baseline_metrics.json", results_dir: str = None):
    """
    Save a list of metrics dicts to a JSON file in results/.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved: {path}")
    return path
