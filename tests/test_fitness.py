import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.genetic_algorithm.fitness import (
    build_estimator,
    calculate_fitness,
    evaluate_individual,
)


def _make_dataset():
    X, y = make_classification(
        n_samples=300,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],
        random_state=42,
    )
    return pd.DataFrame(X), pd.Series(y)


def test_build_estimator_lr():
    model = build_estimator(
        "lr",
        {"C": 1.0, "solver": "liblinear", "max_iter": 300, "class_weight": "balanced"},
    )
    assert model.__class__.__name__ == "LogisticRegression"
    assert model.max_iter == 300


def test_build_estimator_rf():
    model = build_estimator(
        "rf",
        {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": None,
        },
    )
    assert model.__class__.__name__ == "RandomForestClassifier"
    assert model.n_estimators == 100


def test_calculate_fitness_applies_penalty():
    no_penalty = calculate_fitness(recall_mean=0.4, f1_mean=0.3, min_recall=0.3, penalty=0.15)
    with_penalty = calculate_fitness(recall_mean=0.2, f1_mean=0.3, min_recall=0.3, penalty=0.15)
    assert with_penalty < no_penalty
    assert with_penalty >= 0.0


def test_evaluate_individual_lr_returns_deap_tuple():
    X, y = _make_dataset()
    individual = [1.0, 1, 300, 1]  # lr
    fitness = evaluate_individual(individual, "lr", X, y, cv=5, random_state=42)

    assert isinstance(fitness, tuple)
    assert len(fitness) == 1
    assert np.isfinite(fitness[0])
    assert 0.0 <= fitness[0] <= 1.0


def test_evaluate_individual_rf_returns_deap_tuple():
    X, y = _make_dataset()
    individual = [120, 8, 2, 1, 1]  # rf
    fitness = evaluate_individual(individual, "rf", X, y, cv=5, random_state=42)

    assert isinstance(fitness, tuple)
    assert len(fitness) == 1
    assert np.isfinite(fitness[0])
    assert 0.0 <= fitness[0] <= 1.0


def test_invalid_model_type_raises():
    X, y = _make_dataset()
    with pytest.raises(ValueError):
        evaluate_individual([1, 2, 3], "xgb", X, y)

