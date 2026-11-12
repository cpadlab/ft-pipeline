
from __future__ import annotations

import pytest

from ft.config import Settings, get_settings


def test_settings_loads_defaults_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_API_KEY", "secret-key-123456")
    monkeypatch.setenv("AI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("AI_MODEL", "custom-model")
    monkeypatch.setenv("AI_VERIFY_SSL", "false")
    monkeypatch.setenv("AI_MAX_CONCURRENCY", "8")
    monkeypatch.setenv("HF_TOKEN", "hf_valid_token_123456")

    settings = Settings(_env_file=None)

    assert settings.AI_API_KEY.get_secret_value() == "secret-key-123456"
    assert str(settings.AI_BASE_URL) == "https://example.com/v1"
    assert settings.AI_MODEL == "custom-model"
    assert settings.AI_VERIFY_SSL is False
    assert settings.AI_MAX_CONCURRENCY == 8
    assert settings.HF_TOKEN is not None
    assert settings.HF_TOKEN.get_secret_value() == "hf_valid_token_123456"


@pytest.mark.parametrize("raw_token", ["", "   ", "\t\n"])
def test_empty_hf_token_is_converted_to_none(
    monkeypatch: pytest.MonkeyPatch, raw_token: str
) -> None:
    monkeypatch.setenv("HF_TOKEN", raw_token)

    settings = Settings(_env_file=None)

    assert settings.HF_TOKEN is None


def test_get_settings_returns_cached_singleton() -> None:
    first = get_settings()
    second = get_settings()

    assert first is second
