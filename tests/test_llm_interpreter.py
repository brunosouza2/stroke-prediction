from unittest.mock import patch

from src.llm.interpreter import explain_prediction
from src.llm.prompts import build_prediction_prompt

PATIENT_DATA = {
    "age": 67,
    "hypertension": 1,
    "heart_disease": 0,
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
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
