"""
src/llm/client.py

Thin wrapper around the Gemini API (google-genai) used by the rest of
src/llm to turn prompts into natural language text.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai

DEFAULT_MODEL = "gemini-2.5-flash"

load_dotenv()


def configure_client() -> genai.Client:
    """
    Build a Gemini API client from the GOOGLE_API_KEY environment variable.

    Raises RuntimeError if GOOGLE_API_KEY is not set.
    (Copy .env.example to .env and add your Gemini API key.)
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set.\n"
            "Copy .env.example to .env and add your Gemini API key."
        )
    return genai.Client(api_key=api_key)


def generate(
    prompt: str,
    client: genai.Client | None = None,
    model: str | None = None,
    **generation_config,
) -> str:
    """
    Generate text from a prompt using the Gemini API.

    Args:
        client:  A configured genai.Client. If None, one is built via
                 configure_client().
        model:   Model name to use. Defaults to the GEMINI_MODEL env var,
                 falling back to DEFAULT_MODEL.
        **generation_config: Forwarded as generation config (e.g.
                 temperature=0.2, max_output_tokens=512).

    Returns the generated text.
    """
    if client is None:
        client = configure_client()
    if model is None:
        model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=generation_config or None,
    )
    return response.text
