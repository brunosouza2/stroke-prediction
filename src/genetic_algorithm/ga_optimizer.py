"""
Main GA loop for hyperparameter optimization.
"""

from __future__ import annotations

import copy
import json
import os
import random
from datetime import datetime, timezone

import numpy as np

from src.genetic_algorithm.chromosome import create_population, decode
from src.genetic_algorithm.fitness import evaluate_individual
from src.genetic_algorithm.operators import (
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_MUTATION_RATE,
    DEFAULT_TOURNAMENT_K,
    mutate_individual,
    single_point_crossover,
    tournament_selection,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")


def run_ga(
    model_type: str,
    X_train,
    y_train,
    population_size: int = 30,
    generations: int = 30,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    elite_size: int = 2,
    tournament_k: int = DEFAULT_TOURNAMENT_K,
    cv: int = 5,
    patience: int = 8,
    min_delta: float = 1e-4,
    random_state: int = 42,
    save_results: bool = True,
    output_path: str | None = None,
) -> dict:
    """
    Run a full GA optimization cycle and return experiment results.
    """
    _validate_run_config(population_size, generations, elite_size, mutation_rate, crossover_rate, model_type)

    random.seed(random_state)
    np.random.seed(random_state)

    population = create_population(model_type, population_size)
    _evaluate_invalid(population, model_type, X_train, y_train, cv, random_state)

    best_overall = copy.deepcopy(max(population, key=lambda ind: ind.fitness.values[0]))
    best_score = float(best_overall.fitness.values[0])
    no_improvement_count = 0
    history: list[dict] = []
    stop_reason = "max_generations"

    for generation in range(1, generations + 1):
        generation_best = max(population, key=lambda ind: ind.fitness.values[0])
        generation_best_score = float(generation_best.fitness.values[0])
        fitness_values = [float(ind.fitness.values[0]) for ind in population]

        history.append(
            {
                "generation": generation,
                "best_fitness": generation_best_score,
                "avg_fitness": float(np.mean(fitness_values)),
                "worst_fitness": float(np.min(fitness_values)),
            }
        )

        if generation_best_score > best_score + min_delta:
            best_overall = copy.deepcopy(generation_best)
            best_score = generation_best_score
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            stop_reason = "convergence"
            break

        elites = _select_elites(population, elite_size)
        selected = tournament_selection(population, k=population_size - elite_size, tournsize=tournament_k)
        offspring = [copy.deepcopy(ind) for ind in selected]

        _apply_crossover(offspring, crossover_rate)
        _apply_mutation(offspring, model_type, mutation_rate)
        _evaluate_invalid(offspring, model_type, X_train, y_train, cv, random_state)

        population = elites + offspring

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        "best_fitness": best_score,
        "best_individual": list(best_overall),
        "best_params": decode(best_overall, model_type),
        "history": history,
        "stop_reason": stop_reason,
        "config": {
            "population_size": population_size,
            "generations": generations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "elite_size": elite_size,
            "tournament_k": tournament_k,
            "cv": cv,
            "patience": patience,
            "min_delta": min_delta,
            "random_state": random_state,
        },
    }

    if save_results:
        save_experiment(result, output_path=output_path)

    return result


def save_experiment(experiment_result: dict, output_path: str | None = None) -> str:
    """
    Append one experiment result into results/experiments.json.
    """
    if output_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(RESULTS_DIR, "experiments.json")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            content = json.load(f)
    else:
        content = []

    if isinstance(content, dict):
        content = [content]

    content.append(experiment_result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    return output_path


def _evaluate_invalid(population, model_type, X_train, y_train, cv, random_state):
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = evaluate_individual(
                ind,
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                cv=cv,
                random_state=random_state,
            )


def _select_elites(population, elite_size: int):
    sorted_population = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
    return [copy.deepcopy(ind) for ind in sorted_population[:elite_size]]


def _apply_crossover(offspring, crossover_rate: float):
    for idx in range(1, len(offspring), 2):
        if random.random() < crossover_rate:
            single_point_crossover(offspring[idx - 1], offspring[idx])
            if offspring[idx - 1].fitness.valid:
                del offspring[idx - 1].fitness.values
            if offspring[idx].fitness.valid:
                del offspring[idx].fitness.values


def _apply_mutation(offspring, model_type: str, mutation_rate: float):
    for ind in offspring:
        mutate_individual(ind, model_type=model_type, indpb=mutation_rate)
        if ind.fitness.valid:
            del ind.fitness.values


def _validate_run_config(
    population_size: int,
    generations: int,
    elite_size: int,
    mutation_rate: float,
    crossover_rate: float,
    model_type: str,
):
    if model_type not in ("lr", "rf"):
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'lr' or 'rf'.")
    if population_size < 2:
        raise ValueError("population_size must be >= 2.")
    if generations < 1:
        raise ValueError("generations must be >= 1.")
    if elite_size < 1 or elite_size >= population_size:
        raise ValueError("elite_size must be >= 1 and < population_size.")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0.0 and 1.0.")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be between 0.0 and 1.0.")
