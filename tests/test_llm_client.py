from unittest.mock import MagicMock, patch

import pytest

from src.llm.client import DEFAULT_MODEL, configure_client, generate


def _fake_client(response_text: str = "generated text") -> MagicMock:
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(text=response_text)
    return client


def test_configure_client_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        configure_client()


def test_configure_client_builds_client_with_api_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    with patch("src.llm.client.genai.Client") as mock_client_cls:
        mock_client_cls.return_value = "the-client"
        result = configure_client()

    mock_client_cls.assert_called_once_with(api_key="fake-key")
    assert result == "the-client"


def test_generate_uses_default_model(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    client = _fake_client("hello")

    result = generate("prompt", client=client)

    assert result == "hello"
    client.models.generate_content.assert_called_once_with(
        model=DEFAULT_MODEL, contents="prompt", config=None
    )


def test_generate_uses_model_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom")
    client = _fake_client()

    generate("prompt", client=client)

    client.models.generate_content.assert_called_once_with(
        model="gemini-custom", contents="prompt", config=None
    )


def test_generate_forwards_generation_config(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    client = _fake_client()

    generate("prompt", client=client, temperature=0.2, max_output_tokens=512)

    client.models.generate_content.assert_called_once_with(
        model=DEFAULT_MODEL,
        contents="prompt",
        config={"temperature": 0.2, "max_output_tokens": 512},
    )


def test_generate_builds_client_when_none_passed(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    client = _fake_client("from default client")

    with patch("src.llm.client.configure_client", return_value=client) as mock_configure:
        result = generate("prompt")

    mock_configure.assert_called_once()
    assert result == "from default client"
