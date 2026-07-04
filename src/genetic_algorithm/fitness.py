"""
Fitness evaluation for GA individuals.

This module evaluates chromosome individuals with 5-fold stratified CV over
the training data only (no test-set leakage).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

from src.genetic_algorithm.chromosome import decode


def build_estimator(model_type: str, params: dict, random_state: int = 42):
    """
    Build a sklearn estimator from decoded chromosome parameters.
    """
    if model_type == "lr":
        estimator_params = {"random_state": random_state}
        estimator_params.update(params)
        return LogisticRegression(**estimator_params)

    if model_type == "rf":
        estimator_params = {"random_state": random_state, "n_jobs": -1}
        estimator_params.update(params)
        return RandomForestClassifier(**estimator_params)

    raise ValueError(f"Unknown model_type '{model_type}'. Use 'lr' or 'rf'.")


def calculate_fitness(
    recall_mean: float,
    f1_mean: float,
    recall_weight: float = 0.6,
    f1_weight: float = 0.4,
    min_recall: float = 0.30,
    penalty: float = 0.15,
) -> float:
    """
    Combine CV metrics into a single fitness score with low-recall penalty.
    """
    score = (recall_weight * recall_mean) + (f1_weight * f1_mean)
    if recall_mean < min_recall:
        score -= penalty
    return max(score, 0.0)


def evaluate_individual(
    individual: list,
    model_type: str,
    X_train,
    y_train,
    cv: int = 5,
    random_state: int = 42,
    recall_weight: float = 0.6,
    f1_weight: float = 0.4,
    min_recall: float = 0.30,
    penalty: float = 0.15,
) -> tuple[float]:
    """
    Evaluate one individual with stratified CV on training data only.

    Returns a single-objective DEAP-compatible tuple: (fitness_score,)
    """
    params = decode(individual, model_type)
    base_estimator = build_estimator(model_type, params, random_state=random_state)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    recalls: list[float] = []
    f1s: list[float] = []

    for train_idx, valid_idx in skf.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx] if hasattr(X_train, "iloc") else X_train[train_idx]
        y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, "iloc") else y_train[train_idx]
        X_fold_valid = X_train.iloc[valid_idx] if hasattr(X_train, "iloc") else X_train[valid_idx]
        y_fold_valid = y_train.iloc[valid_idx] if hasattr(y_train, "iloc") else y_train[valid_idx]

        model = clone(base_estimator)
        model.fit(X_fold_train, y_fold_train)
        preds = model.predict(X_fold_valid)

        recalls.append(recall_score(y_fold_valid, preds, zero_division=0))
        f1s.append(f1_score(y_fold_valid, preds, zero_division=0))

    recall_mean = float(np.mean(recalls))
    f1_mean = float(np.mean(f1s))
    fitness = calculate_fitness(
        recall_mean=recall_mean,
        f1_mean=f1_mean,
        recall_weight=recall_weight,
        f1_weight=f1_weight,
        min_recall=min_recall,
        penalty=penalty,
    )
    return (fitness,)

