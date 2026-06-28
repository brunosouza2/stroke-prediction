import random

import pytest

from src.genetic_algorithm.chromosome import create_individual, get_bounds
from src.genetic_algorithm.operators import (
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_MUTATION_RATE,
    DEFAULT_TOURNAMENT_K,
    mutate_individual,
    single_point_crossover,
    tournament_selection,
)


def test_default_rates_match_task_definition():
    assert DEFAULT_CROSSOVER_RATE == 0.8
    assert DEFAULT_MUTATION_RATE == 0.15
    assert DEFAULT_TOURNAMENT_K == 3


def test_tournament_selection_returns_k_individuals():
    random.seed(42)
    population = [create_individual("lr") for _ in range(10)]
    selected = tournament_selection(population, k=5, tournsize=3)

    assert len(selected) == 5
    for ind in selected:
        assert ind in population


def test_single_point_crossover_preserves_shape():
    random.seed(42)
    ind1 = create_individual("rf")
    ind2 = create_individual("rf")
    len_before = len(ind1)

    out1, out2 = single_point_crossover(ind1, ind2)
    assert len(out1) == len_before
    assert len(out2) == len_before


def test_single_point_crossover_requires_same_length():
    with pytest.raises(ValueError):
        single_point_crossover([1, 2, 3], [1, 2, 3, 4])


def test_mutate_lr_keeps_bounds_and_types():
    random.seed(7)
    ind = [1.0, 0, 200, 0]
    low, high = get_bounds("lr")
    mutated, = mutate_individual(ind, "lr", indpb=1.0, sigma=0.3)

    assert isinstance(mutated[0], float)
    assert isinstance(mutated[1], int)
    assert isinstance(mutated[2], int)
    assert isinstance(mutated[3], int)

    for i, gene in enumerate(mutated):
        assert low[i] <= gene <= high[i]


def test_mutate_lr_categorical_gene_changes_with_full_probability():
    random.seed(1)
    ind = [1.0, 0, 250, 0]
    mutated, = mutate_individual(ind, "lr", indpb=1.0, sigma=0.2)

    # class_weight_idx has only two values (0/1), so it must flip.
    assert mutated[3] == 1


def test_mutate_unknown_model_type_raises():
    with pytest.raises(ValueError):
        mutate_individual([1, 2, 3], "xgb")

