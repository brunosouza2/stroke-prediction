"""
src/llm/interpreter.py

High-level orchestration: combines src/llm/prompts.py (prompt construction)
with src/llm/client.py (Gemini API calls) to turn model outputs into
natural language text.
"""

from __future__ import annotations

from src.llm.client import generate
from src.llm.prompts import build_prediction_prompt


def explain_prediction(
    patient_data: dict,
    prediction: int,
    probability: float,
    client=None,
    model: str | None = None,
    **generation_config,
) -> str:
    """
    Generate a natural language explanation of one patient's stroke risk
    prediction, for a clinician.

    Args:
        patient_data: mapping of feature name -> value for one patient.
        prediction:   predicted class (0 = no stroke, 1 = stroke risk).
        probability:  predicted probability of the positive (stroke) class.
        client:       an already-configured genai.Client (forwarded to
                      client.generate). If None, one is built automatically.
        model:        Gemini model name override (forwarded to client.generate).
        **generation_config: forwarded to client.generate (e.g. temperature).

    Returns the generated explanation text.
    """
    prompt = build_prediction_prompt(patient_data, prediction, probability)
    return generate(prompt, client=client, model=model, **generation_config)
