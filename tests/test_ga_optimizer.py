import json

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.genetic_algorithm.ga_optimizer import run_ga, save_experiment


def _make_dataset():
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.88, 0.12],
        random_state=42,
    )
    return pd.DataFrame(X), pd.Series(y)


def test_run_ga_returns_result_and_saves_file(tmp_path):
    X, y = _make_dataset()
    output_path = str(tmp_path / "experiments.json")

    result = run_ga(
        model_type="lr",
        X_train=X,
        y_train=y,
        population_size=8,
        generations=4,
        cv=3,
        patience=3,
        random_state=42,
        output_path=output_path,
        save_results=True,
    )

    assert result["model_type"] == "lr"
    assert isinstance(result["best_fitness"], float)
    assert isinstance(result["best_individual"], list)
    assert isinstance(result["best_params"], dict)
    assert len(result["history"]) >= 1
    assert result["stop_reason"] in ("max_generations", "convergence")

    with open(output_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert isinstance(payload, list)
    assert len(payload) == 1


def test_run_ga_can_stop_by_convergence():
    X, y = _make_dataset()

    result = run_ga(
        model_type="lr",
        X_train=X,
        y_train=y,
        population_size=8,
        generations=10,
        cv=3,
        patience=1,
        min_delta=1.0,
        random_state=42,
        save_results=False,
    )

    assert result["stop_reason"] == "convergence"
    assert len(result["history"]) < 10


def test_save_experiment_appends_entries(tmp_path):
    output_path = str(tmp_path / "experiments.json")
    save_experiment({"model_type": "lr", "best_fitness": 0.1}, output_path=output_path)
    save_experiment({"model_type": "rf", "best_fitness": 0.2}, output_path=output_path)

    with open(output_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert len(payload) == 2
    assert payload[0]["model_type"] == "lr"
    assert payload[1]["model_type"] == "rf"


def test_run_ga_validates_config():
    X, y = _make_dataset()

    with pytest.raises(ValueError):
        run_ga("xgb", X, y, save_results=False)

    with pytest.raises(ValueError):
        run_ga("lr", X, y, population_size=1, save_results=False)
