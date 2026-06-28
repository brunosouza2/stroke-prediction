import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.genetic_algorithm.chromosome import (
    create_individual,
    create_population,
    decode,
    get_bounds,
)


def test_create_individual_lr_shape_and_bounds():
    ind = create_individual("lr")
    low, high = get_bounds("lr")

    assert len(ind) == 4
    for i, gene in enumerate(ind):
        assert low[i] <= gene <= high[i]


def test_create_individual_rf_shape_and_bounds():
    ind = create_individual("rf")
    low, high = get_bounds("rf")

    assert len(ind) == 5
    for i, gene in enumerate(ind):
        assert low[i] <= gene <= high[i]


def test_decode_lr_is_sklearn_compatible():
    ind = [1.5, 1, 300, 1]  # liblinear, class_weight='balanced'
    params = decode(ind, "lr")
    model = LogisticRegression(**params)

    assert model.solver == "liblinear"
    assert model.class_weight == "balanced"
    assert model.max_iter == 300


def test_decode_rf_depth_mapping_is_correct():
    base = [100, 0, 2, 1, 0]

    p0 = decode(base, "rf")
    assert p0["max_depth"] is None

    p1 = decode([100, 1, 2, 1, 0], "rf")
    assert p1["max_depth"] == 3

    p_last = decode([100, 28, 2, 1, 0], "rf")
    assert p_last["max_depth"] == 30

    model = RandomForestClassifier(**p_last)
    assert model.max_depth == 30


def test_create_population_returns_individuals_with_fitness():
    pop = create_population("lr", 6)
    assert len(pop) == 6
    assert hasattr(pop[0], "fitness")


def test_invalid_model_type_raises_value_error():
    with pytest.raises(ValueError):
        create_individual("xgb")

    with pytest.raises(ValueError):
        decode([1, 2, 3, 4], "xgb")
