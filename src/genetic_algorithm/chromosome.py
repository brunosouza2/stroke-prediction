"""
chromosome.py - DEAP chromosome definitions for hyperparameter optimization.

Each individual is a list of genes encoding hyperparameters for either
Logistic Regression (LR) or Random Forest (RF). Gene spaces use natural
value ranges (not normalized) for readability and easier debugging.

LR chromosome (4 genes):
  [0] C            float  in [0.001, 100.0]   — inverse regularization strength
  [1] solver_idx   int    in [0, 3]            — index into SOLVERS list
  [2] max_iter     int    in [100, 1000]       — maximum iterations
  [3] cw_idx       int    in [0, 1]            — index into CLASS_WEIGHTS list

RF chromosome (5 genes):
  [0] n_estimators      int  in [50, 500]     — number of trees
  [1] max_depth_idx     int  in [0, 28]       — 0 → None, 1..28 → depth 3..30
  [2] min_samples_split int  in [2, 20]       — min samples to split a node
  [3] min_samples_leaf  int  in [1, 10]       — min samples in a leaf
  [4] cw_idx            int  in [0, 1]        — index into CLASS_WEIGHTS list
"""

import random
from deap import base, creator

# ---------------------------------------------------------------------------
# Categorical lookup tables (index → sklearn value)
# ---------------------------------------------------------------------------
LR_SOLVERS = ["lbfgs", "liblinear", "saga", "newton-cg"]
CLASS_WEIGHTS = [None, "balanced"]

# ---------------------------------------------------------------------------
# DEAP types (created once at module load)
# ---------------------------------------------------------------------------
# Guard against re-registration when the module is reloaded (e.g., in notebooks)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


# ---------------------------------------------------------------------------
# Gene bounds
# ---------------------------------------------------------------------------
LR_BOUNDS = {
    "C":          {"type": float, "low": 0.001,  "high": 100.0},
    "solver_idx": {"type": int,   "low": 0,      "high": len(LR_SOLVERS) - 1},
    "max_iter":   {"type": int,   "low": 100,    "high": 1000},
    "cw_idx":     {"type": int,   "low": 0,      "high": len(CLASS_WEIGHTS) - 1},
}

RF_BOUNDS = {
    "n_estimators":      {"type": int, "low": 50,  "high": 500},
    "max_depth_idx":     {"type": int, "low": 0,   "high": 28},
    "min_samples_split": {"type": int, "low": 2,   "high": 20},
    "min_samples_leaf":  {"type": int, "low": 1,   "high": 10},
    "cw_idx":            {"type": int, "low": 0,   "high": len(CLASS_WEIGHTS) - 1},
}

BOUNDS = {"lr": LR_BOUNDS, "rf": RF_BOUNDS}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_individual(model_type: str) -> creator.Individual:
    """Return a randomly initialised DEAP Individual for the given model type.

    Parameters
    ----------
    model_type : str
        ``"lr"`` for Logistic Regression or ``"rf"`` for Random Forest.

    Returns
    -------
    creator.Individual
        A list of genes with an attached ``fitness`` attribute.

    Raises
    ------
    ValueError
        If ``model_type`` is not ``"lr"`` or ``"rf"``.

    Example
    -------
    >>> ind = create_individual("lr")
    >>> len(ind)
    4
    """
    bounds = _get_bounds(model_type)
    genes = [_sample_gene(spec) for spec in bounds.values()]
    return creator.Individual(genes)


def decode(individual: list, model_type: str) -> dict:
    """Decode a chromosome into a dict of sklearn-compatible hyperparameters.

    Parameters
    ----------
    individual : list
        Gene list produced by :func:`create_individual`.
    model_type : str
        ``"lr"`` or ``"rf"``.

    Returns
    -------
    dict
        Ready to unpack as ``**kwargs`` into the sklearn constructor.

    Example
    -------
    >>> ind = create_individual("lr")
    >>> params = decode(ind, "lr")
    >>> from sklearn.linear_model import LogisticRegression
    >>> model = LogisticRegression(**params)
    """
    if model_type == "lr":
        return _decode_lr(individual)
    if model_type == "rf":
        return _decode_rf(individual)
    raise ValueError(f"Unknown model_type '{model_type}'. Use 'lr' or 'rf'.")


def create_population(model_type: str, size: int) -> list:
    """Return a list of ``size`` randomly initialised individuals.

    Parameters
    ----------
    model_type : str
        ``"lr"`` or ``"rf"``.
    size : int
        Number of individuals in the population.

    Returns
    -------
    list[creator.Individual]

    Example
    -------
    >>> pop = create_population("rf", 20)
    >>> len(pop)
    20
    """
    return [create_individual(model_type) for _ in range(size)]


def get_bounds(model_type: str) -> tuple[list, list]:
    """Return (lower_bounds, upper_bounds) for mutation/crossover operators.

    Useful when configuring DEAP's ``tools.mutPolynomialBounded`` or similar.

    Parameters
    ----------
    model_type : str
        ``"lr"`` or ``"rf"``.

    Returns
    -------
    tuple[list, list]
        Two lists ``(low, high)`` with one entry per gene.
    """
    bounds = _get_bounds(model_type)
    low  = [spec["low"]  for spec in bounds.values()]
    high = [spec["high"] for spec in bounds.values()]
    return low, high


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_bounds(model_type: str) -> dict:
    if model_type not in BOUNDS:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'lr' or 'rf'.")
    return BOUNDS[model_type]


def _sample_gene(spec: dict):
    """Sample a single gene value according to its type and bounds."""
    if spec["type"] is float:
        return random.uniform(spec["low"], spec["high"])
    return random.randint(spec["low"], spec["high"])


def _decode_lr(ind: list) -> dict:
    """Decode a 4-gene LR chromosome into sklearn kwargs."""
    C, solver_idx, max_iter, cw_idx = ind
    return {
        "C":            float(C),
        "solver":       LR_SOLVERS[int(solver_idx)],
        "max_iter":     int(max_iter),
        "class_weight": CLASS_WEIGHTS[int(cw_idx)],
    }


def _decode_rf(ind: list) -> dict:
    """Decode a 5-gene RF chromosome into sklearn kwargs."""
    n_estimators, max_depth_idx, min_samples_split, min_samples_leaf, cw_idx = ind

    # Index 0 → None (unlimited depth), 1..28 → depths 3..30
    max_depth = None if int(max_depth_idx) == 0 else int(max_depth_idx) + 2

    return {
        "n_estimators":      int(n_estimators),
        "max_depth":         max_depth,
        "min_samples_split": int(min_samples_split),
        "min_samples_leaf":  int(min_samples_leaf),
        "class_weight":      CLASS_WEIGHTS[int(cw_idx)],
    }
