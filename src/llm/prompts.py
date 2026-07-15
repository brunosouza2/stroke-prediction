"""
src/llm/prompts.py

Pure prompt-building functions consumed by src/llm/interpreter.py. None of
these functions call the Gemini API — they only assemble text, which keeps
them cheap and deterministic to test.
"""

from __future__ import annotations

PREDICTION_LABELS = {0: "baixo risco de AVC", 1: "alto risco de AVC"}


def _format_dict_block(data: dict) -> str:
    return "\n".join(f"- {name}: {value}" for name, value in data.items())


def build_prediction_prompt(patient_data: dict, prediction: int, probability: float) -> str:
    """
    Build a prompt asking the LLM to explain one patient's stroke risk
    prediction in plain language for a clinician.

    Args:
        patient_data: mapping of feature name -> value for one patient
                      (e.g. a dict, or a pandas Series via .to_dict()).
        prediction:   predicted class (0 = no stroke, 1 = stroke risk).
        probability:  predicted probability of the positive (stroke) class.

    Returns the prompt text.
    """
    label = PREDICTION_LABELS[prediction]
    features_block = _format_dict_block(patient_data)

    return (
        "Você é um assistente clínico que ajuda médicos a interpretar a saída "
        "de um modelo de machine learning para previsão de risco de AVC "
        "(Acidente Vascular Cerebral).\n\n"
        "Dados do paciente:\n"
        f"{features_block}\n\n"
        f"Predição do modelo: {label} (classe {prediction}).\n"
        f"Probabilidade estimada de AVC: {probability:.1%}.\n\n"
        "Explique este resultado em linguagem natural, em português, para um "
        "profissional de saúde. Destaque quais fatores do paciente mais "
        "provavelmente influenciaram essa predição e sugira, de forma "
        "cautelosa e sem substituir o julgamento clínico, quais próximos "
        "passos poderiam ser considerados. Use apenas os dados fornecidos "
        "acima — não invente exames, históricos ou informações que não "
        "foram passados. Seja objetivo e evite jargão técnico desnecessário."
    )


def build_experiment_prompt(baseline_metrics: dict, optimized_metrics: dict, best_params: dict) -> str:
    """
    Build a prompt asking the LLM to interpret one genetic algorithm
    optimization experiment: how the optimized model compares to the
    original baseline (see results/ga_summary.json for the expected shape
    of baseline_metrics / optimized_metrics / best_params).

    Args:
        baseline_metrics:  metrics dict for the original model (accuracy,
                            precision, recall, f1, confusion_matrix, ...).
        optimized_metrics: metrics dict for the GA-optimized model, same shape.
        best_params:       hyperparameters found by the genetic algorithm.

    Returns the prompt text.
    """
    baseline_block = _format_dict_block(baseline_metrics)
    optimized_block = _format_dict_block(optimized_metrics)
    params_block = _format_dict_block(best_params)

    recall_delta = optimized_metrics["recall"] - baseline_metrics["recall"]
    f1_delta = optimized_metrics["f1"] - baseline_metrics["f1"]

    return (
        "Você é um assistente que ajuda uma equipe médica e de dados a "
        "interpretar o resultado de uma otimização de hiperparâmetros feita "
        "com algoritmos genéticos, para um modelo de previsão de risco de "
        "AVC (Acidente Vascular Cerebral).\n\n"
        "Métricas do modelo original (baseline):\n"
        f"{baseline_block}\n\n"
        "Métricas do modelo otimizado pelo algoritmo genético:\n"
        f"{optimized_block}\n\n"
        "Hiperparâmetros encontrados pela otimização:\n"
        f"{params_block}\n\n"
        f"Variação de recall: {recall_delta:+.4f}.\n"
        f"Variação de F1-score: {f1_delta:+.4f}.\n\n"
        "Nesse contexto, recall é a métrica mais importante: representa a "
        "capacidade do modelo de identificar corretamente pacientes com "
        "risco real de AVC, minimizando falsos negativos. Escreva, em "
        "português, um resumo executivo para profissionais de saúde e "
        "gestores explicando se a otimização trouxe um ganho relevante, o "
        "que mudou nos hiperparâmetros e qual o impacto prático esperado. "
        "Use apenas os números fornecidos acima — não invente estatísticas "
        "adicionais."
    )
