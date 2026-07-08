from src.llm.evaluation import MAX_LENGTH, MIN_LENGTH, evaluate_explanation

PATIENT_DATA = {
    "age": 67,
    "hypertension": 1,
    "avg_glucose_level": 228.69,
}

EXPERIMENT_DATA = {
    "recall": 0.44,
    "f1": 0.2056,
    "confusion_matrix": [[830, 142], [28, 22]],
}

GOOD_EXPERIMENT_TEXT = (
    "O modelo otimizado manteve um recall de 0.44, ou seja, a probabilidade "
    "de identificar corretamente pacientes com risco de AVC segue estável. "
    "Isso sugere que a otimização não trouxe ganhos relevantes neste caso, "
    "mas também não piorou a capacidade de detecção."
)


def test_evaluate_explanation_all_checks_pass():
    result = evaluate_explanation(GOOD_EXPERIMENT_TEXT, EXPERIMENT_DATA)

    assert result["score"] == 1.0
    assert all(result["checks"].values())


def test_evaluate_explanation_fails_mentions_recall_when_missing():
    text = "O modelo manteve desempenho estável, com f1 de 0.2056, o que sugere baixo risco de mudança."
    result = evaluate_explanation(text, EXPERIMENT_DATA)

    assert result["checks"]["mentions_recall"] is False
    assert result["score"] < 1.0


def test_evaluate_explanation_mentions_recall_is_vacuous_without_recall_key():
    text = "Paciente com idade avançada e glicemia elevada apresenta risco a ser monitorado."
    result = evaluate_explanation(text, PATIENT_DATA)

    assert result["checks"]["mentions_recall"] is True


def test_evaluate_explanation_fails_grounded_in_data_when_no_numbers_cited():
    text = "O paciente apresenta fatores de risco relevantes e deve ser monitorado com cautela."
    result = evaluate_explanation(text, PATIENT_DATA)

    assert result["checks"]["grounded_in_data"] is False


def test_evaluate_explanation_passes_grounded_in_data_when_a_real_number_is_cited():
    text = "Aos 67 anos, o paciente apresenta risco a ser monitorado com cautela clínica."
    result = evaluate_explanation(text, PATIENT_DATA)

    assert result["checks"]["grounded_in_data"] is True


def test_evaluate_explanation_grounded_in_data_reaches_nested_confusion_matrix():
    text = "Dos 830 casos negativos corretamente identificados, o modelo indica estabilidade."
    result = evaluate_explanation(text, EXPERIMENT_DATA)

    assert result["checks"]["grounded_in_data"] is True


def test_evaluate_explanation_fails_uses_risk_language_for_categorical_claim():
    text = "O paciente tem AVC. Idade 67, glicemia 228.69, hipertensão 1. Encaminhar imediatamente ao hospital."
    result = evaluate_explanation(text, PATIENT_DATA)

    assert result["checks"]["uses_risk_language"] is False


def test_evaluate_explanation_fails_reasonable_length_when_too_short():
    text = "Ok."
    assert len(text) < MIN_LENGTH

    result = evaluate_explanation(text, PATIENT_DATA)

    assert result["checks"]["reasonable_length"] is False


def test_evaluate_explanation_fails_reasonable_length_when_too_long():
    text = "risco " * (MAX_LENGTH // len("risco ") + 10)
    assert len(text) > MAX_LENGTH

    result = evaluate_explanation(text, PATIENT_DATA)

    assert result["checks"]["reasonable_length"] is False


def test_evaluate_explanation_score_is_fraction_of_passed_checks():
    # Grounded + reasonable length, but no recall mention and no risk language.
    text = "O paciente tem 67 anos e f1 registrado foi de 0.2056 no ultimo experimento realizado ontem."
    result = evaluate_explanation(text, EXPERIMENT_DATA)

    passed = sum(result["checks"].values())
    assert result["score"] == passed / len(result["checks"])
    assert 0.0 < result["score"] < 1.0
