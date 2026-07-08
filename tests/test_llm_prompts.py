from src.llm.prompts import PREDICTION_LABELS, build_prediction_prompt

PATIENT_DATA = {
    "age": 67,
    "hypertension": 1,
    "heart_disease": 0,
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "ever_married_Yes": True,
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
