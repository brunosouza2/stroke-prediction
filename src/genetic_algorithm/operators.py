"""
Genetic operators for the GA optimizer.
"""

from __future__ import annotations

import random
from deap import tools

from src.genetic_algorithm.chromosome import get_bounds


DEFAULT_CROSSOVER_RATE = 0.8
DEFAULT_MUTATION_RATE = 0.15
DEFAULT_TOURNAMENT_K = 3

_CATEGORICAL_INDEXES = {
    "lr": {1, 3},  # solver_idx, class_weight_idx
    "rf": {1, 4},  # max_depth_idx, class_weight_idx
}


def tournament_selection(population: list, k: int, tournsize: int = DEFAULT_TOURNAMENT_K):
    """
    Select k individuals from the population using tournament selection.
    """
    return tools.selTournament(population, k=k, tournsize=tournsize)


def single_point_crossover(ind1: list, ind2: list):
    """
    Apply single-point crossover to two individuals of the same chromosome schema.
    """
    if len(ind1) != len(ind2):
        raise ValueError("Crossover requires individuals with the same chromosome length.")
    return tools.cxOnePoint(ind1, ind2)


def mutate_individual(
    individual: list,
    model_type: str,
    indpb: float = DEFAULT_MUTATION_RATE,
    sigma: float = 0.1,
):
    """
    Mutate one individual in-place and return DEAP-compatible tuple.

    - Continuous genes (float): Gaussian mutation with clipping.
    - Categorical genes (int indexes): Random category replacement.
    - Integer numeric genes: Gaussian step with rounding and clipping.
    """
    low, high = get_bounds(model_type)
    categorical = _CATEGORICAL_INDEXES.get(model_type)
    if categorical is None:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'lr' or 'rf'.")

    for idx, gene in enumerate(individual):
        if random.random() >= indpb:
            continue

        lo = low[idx]
        hi = high[idx]

        if isinstance(gene, float):
            step = random.gauss(0.0, sigma * (hi - lo))
            new_value = gene + step
            individual[idx] = float(min(max(new_value, lo), hi))
            continue

        if idx in categorical:
            choices = [value for value in range(int(lo), int(hi) + 1) if value != int(gene)]
            if choices:
                individual[idx] = int(random.choice(choices))
            continue

        step = random.gauss(0.0, sigma * (hi - lo))
        new_value = int(round(gene + step))
        individual[idx] = int(min(max(new_value, int(lo)), int(hi)))

    return (individual,)

