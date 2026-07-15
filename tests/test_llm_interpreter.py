from unittest.mock import patch

from src.llm.interpreter import explain_prediction, summarize_experiment
from src.llm.prompts import build_experiment_prompt, build_prediction_prompt

PATIENT_DATA = {
    "age": 67,
    "hypertension": 1,
    "heart_disease": 0,
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
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


def test_explain_prediction_returns_generate_result():
    with patch("src.llm.interpreter.generate", return_value="explicação gerada") as mock_generate:
        result = explain_prediction(PATIENT_DATA, prediction=1, probability=0.42)

    assert result == "explicação gerada"
    mock_generate.assert_called_once()


def test_explain_prediction_builds_the_correct_prompt():
    expected_prompt = build_prediction_prompt(PATIENT_DATA, prediction=1, probability=0.42)

    with patch("src.llm.interpreter.generate", return_value="explicação gerada") as mock_generate:
        explain_prediction(PATIENT_DATA, prediction=1, probability=0.42)

    mock_generate.assert_called_once_with(expected_prompt, client=None, model=None)


def test_explain_prediction_forwards_client_model_and_generation_config():
    sentinel_client = object()

    with patch("src.llm.interpreter.generate", return_value="explicação gerada") as mock_generate:
        explain_prediction(
            PATIENT_DATA,
            prediction=0,
            probability=0.05,
            client=sentinel_client,
            model="gemini-custom",
            temperature=0.3,
        )

    _, kwargs = mock_generate.call_args
    assert kwargs["client"] is sentinel_client
    assert kwargs["model"] == "gemini-custom"
    assert kwargs["temperature"] == 0.3


def test_summarize_experiment_returns_generate_result():
    with patch("src.llm.interpreter.generate", return_value="resumo gerado") as mock_generate:
        result = summarize_experiment(BASELINE_METRICS, OPTIMIZED_METRICS, BEST_PARAMS)

    assert result == "resumo gerado"
    mock_generate.assert_called_once()


def test_summarize_experiment_builds_the_correct_prompt():
    expected_prompt = build_experiment_prompt(BASELINE_METRICS, OPTIMIZED_METRICS, BEST_PARAMS)

    with patch("src.llm.interpreter.generate", return_value="resumo gerado") as mock_generate:
        summarize_experiment(BASELINE_METRICS, OPTIMIZED_METRICS, BEST_PARAMS)

    mock_generate.assert_called_once_with(expected_prompt, client=None, model=None)


def test_summarize_experiment_forwards_client_model_and_generation_config():
    sentinel_client = object()

    with patch("src.llm.interpreter.generate", return_value="resumo gerado") as mock_generate:
        summarize_experiment(
            BASELINE_METRICS,
            OPTIMIZED_METRICS,
            BEST_PARAMS,
            client=sentinel_client,
            model="gemini-custom",
            temperature=0.3,
        )

    _, kwargs = mock_generate.call_args
    assert kwargs["client"] is sentinel_client
    assert kwargs["model"] == "gemini-custom"
    assert kwargs["temperature"] == 0.3
