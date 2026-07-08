"""
src/llm/prompts.py

Pure prompt-building functions consumed by src/llm/interpreter.py. None of
these functions call the Gemini API — they only assemble text, which keeps
them cheap and deterministic to test.
"""

from __future__ import annotations

PREDICTION_LABELS = {0: "baixo risco de AVC", 1: "alto risco de AVC"}


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
    features_block = "\n".join(f"- {name}: {value}" for name, value in patient_data.items())

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
