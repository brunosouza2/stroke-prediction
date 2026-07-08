from src.llm.prompts import (
    PREDICTION_LABELS,
    build_experiment_prompt,
    build_prediction_prompt,
)

PATIENT_DATA = {
    "age": 67,
    "hypertension": 1,
    "heart_disease": 0,
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "ever_married_Yes": True,
}

# Same shape as results/ga_summary.json (EXP-01 entry).
BASELINE_METRICS = {
    "model": "Logistic Regression (baseline)",
    "accuracy": 0.8337,
    "precision": 0.1341,
    "recall": 0.44,
    "f1": 0.2056,
    "confusion_matrix": [[830, 142], [28, 22]],
}

OPTIMIZED_METRICS = {
    "model": "EXP-01 Optimized LR",
    "accuracy": 0.8327,
    "precision": 0.1288,
    "recall": 0.42,
    "f1": 0.1972,
    "confusion_matrix": [[830, 142], [29, 21]],
}

BEST_PARAMS = {
    "C": 67.6702720428037,
    "solver": "lbfgs",
    "max_iter": 704,
    "class_weight": "balanced",
}


def test_build_prediction_prompt_includes_all_patient_features():
    prompt = build_prediction_prompt(PATIENT_DATA, prediction=1, probability=0.42)

    for name, value in PATIENT_DATA.items():
        assert name in prompt
        assert str(value) in prompt


def test_build_prediction_prompt_includes_positive_class_label():
    prompt = build_prediction_prompt(PATIENT_DATA, prediction=1, probability=0.42)

    assert PREDICTION_LABELS[1] in prompt
    assert "classe 1" in prompt


def test_build_prediction_prompt_includes_negative_class_label():
    prompt = build_prediction_prompt(PATIENT_DATA, prediction=0, probability=0.05)

    assert PREDICTION_LABELS[0] in prompt
    assert "classe 0" in prompt


def test_build_prediction_prompt_includes_formatted_probability():
    prompt = build_prediction_prompt(PATIENT_DATA, prediction=1, probability=0.4234)

    assert "42.3%" in prompt


def test_build_prediction_prompt_returns_plain_string():
    prompt = build_prediction_prompt(PATIENT_DATA, prediction=0, probability=0.1)

    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_experiment_prompt_includes_baseline_and_optimized_metrics():
    prompt = build_experiment_prompt(BASELINE_METRICS, OPTIMIZED_METRICS, BEST_PARAMS)

    for name, value in {**BASELINE_METRICS, **OPTIMIZED_METRICS}.items():
        assert name in prompt
        assert str(value) in prompt


def test_build_experiment_prompt_includes_best_params():
    prompt = build_experiment_prompt(BASELINE_METRICS, OPTIMIZED_METRICS, BEST_PARAMS)

    for name, value in BEST_PARAMS.items():
        assert name in prompt
        assert str(value) in prompt


def test_build_experiment_prompt_includes_recall_and_f1_deltas():
    prompt = build_experiment_prompt(BASELINE_METRICS, OPTIMIZED_METRICS, BEST_PARAMS)

    assert "-0.0200" in prompt  # recall: 0.42 - 0.44
    assert "-0.0084" in prompt  # f1: 0.1972 - 0.2056


def test_build_experiment_prompt_returns_plain_string():
    prompt = build_experiment_prompt(BASELINE_METRICS, OPTIMIZED_METRICS, BEST_PARAMS)

    assert isinstance(prompt, str)
    assert len(prompt) > 0
