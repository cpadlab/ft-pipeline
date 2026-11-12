
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from huggingface_hub.errors import GatedRepoError

from ft.models.registry import BILLION, SLMRegistry


def _make_model_info(
    repo_id: str,
    *,
    total_params: int | None = int(1.5 * BILLION),
    model_type: str = "qwen2",
    gated: bool = False,
    downloads: int = 500,
    likes: int = 10,
    tags: list[str] | None = None,
) -> SimpleNamespace:
    safetensors = SimpleNamespace(total=total_params) if total_params is not None else None
    return SimpleNamespace(
        id=repo_id,
        safetensors=safetensors,
        config={"model_type": model_type},
        gated=gated,
        downloads=downloads,
        likes=likes,
        tags=tags or ["transformers", "safetensors"],
    )


def test_registry_init_validation_and_token_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="max_params_b"):
        SLMRegistry(max_params_b=0.0)

    monkeypatch.setenv("HF_TOKEN", "hf_from_env_123456")
    reg_env = SLMRegistry()
    assert reg_env._hf_token == "hf_from_env_123456"

    reg_explicit = SLMRegistry(hf_token="hf_explicit")
    assert reg_explicit._hf_token == "hf_explicit"


def test_iter_models_and_search_models(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = SLMRegistry(max_params_b=3.0)
    fake_api = MagicMock()
    fake_api.list_models.return_value = [
        _make_model_info("org/no-meta", total_params=None),
        _make_model_info("org/too-big", total_params=int(5.0 * BILLION)),
        _make_model_info("org/slm-1", total_params=int(1.0 * BILLION)),
        _make_model_info("org/slm-2", total_params=int(2.0 * BILLION)),
    ]
    monkeypatch.setattr(reg, "_api", fake_api)

    page = reg.search_models(query="slm", tags=["es"], page=1, page_size=1)
    assert len(page.items) == 1
    assert page.items[0].source == "org/slm-1"
    assert page.has_more is True

    all_models = list(
        reg.iter_models(tags=["transformers", "safetensors"], require_safetensors=False)
    )
    assert len(all_models) == 2

    with pytest.raises(ValueError, match="page must be"):
        reg.search_models(page=0)
    with pytest.raises(ValueError, match="page_size"):
        reg.search_models(page_size=101)
    with pytest.raises(ValueError, match="Invalid parameter range"):
        list(reg.iter_models(min_params_b=2.0, max_params_b=1.0))
    with pytest.raises(ValueError, match="exceeds registry limit"):
        list(reg.iter_models(max_params_b=4.0))


def test_validate_model_hub_and_autoconfig_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reg = SLMRegistry(max_params_b=3.0)
    fake_api = MagicMock()
    monkeypatch.setattr(reg, "_api", fake_api)

    fake_api.model_info.return_value = _make_model_info("org/ok", total_params=int(1.5 * BILLION))
    profile = reg.validate_model("org/ok")
    assert profile.estimated_params_b == 1.5

    fake_api.model_info.return_value = _make_model_info("org/huge", total_params=int(7.0 * BILLION))
    with pytest.raises(ValueError, match="exceeding the SLM ceiling"):
        reg.validate_model("org/huge")

    fake_api.model_info.side_effect = RuntimeError("hub down")
    small_cfg = SimpleNamespace(
        model_type="llama",
        hidden_size=512,
        num_hidden_layers=4,
        vocab_size=32000,
        intermediate_size=1024,
        num_attention_heads=8,
        num_key_value_heads=4,
        tie_word_embeddings=True,
    )
    monkeypatch.setattr("ft.models.registry.AutoConfig.from_pretrained", lambda *a, **kw: small_cfg)
    local_profile = reg.validate_model("org/fallback")
    assert local_profile.model_type == "llama"

    local_dir = tmp_path / "local_slm"
    local_dir.mkdir()
    assert reg.validate_model(local_dir).source == str(local_dir)

    huge_cfg = SimpleNamespace(
        model_type="llama",
        hidden_size=8192,
        num_hidden_layers=80,
        vocab_size=128000,
        intermediate_size=28672,
        num_attention_heads=64,
        tie_word_embeddings=False,
    )
    monkeypatch.setattr("ft.models.registry.AutoConfig.from_pretrained", lambda *a, **kw: huge_cfg)
    with pytest.raises(ValueError, match="exceeding the"):
        reg.validate_model(local_dir)


def test_validate_via_autoconfig_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = SLMRegistry()

    def _raise_gated(*_a: object, **_kw: object) -> None:
        raise OSError("401 Client Error: Gated repo")

    monkeypatch.setattr("ft.models.registry.AutoConfig.from_pretrained", _raise_gated)
    with pytest.raises(PermissionError, match="gated or private"):
        reg._validate_via_autoconfig("org/gated")

    def _raise_other(*_a: object, **_kw: object) -> None:
        raise OSError("network timeout")

    monkeypatch.setattr("ft.models.registry.AutoConfig.from_pretrained", _raise_other)
    with pytest.raises(OSError, match="network timeout"):
        reg._validate_via_autoconfig("org/broken")


def test_download_model_success_and_gated_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reg = SLMRegistry()
    monkeypatch.setattr(
        reg,
        "validate_model",
        lambda _m: _make_model_info("org/ok") and SLMRegistry._profile_from_hub_info(_make_model_info("org/ok"), 0.0, 10.0),
    )

    monkeypatch.setattr(
        "ft.models.registry.snapshot_download",
        lambda **_kw: str(tmp_path / "snap"),
    )
    assert reg.download_model("org/ok", local_dir=tmp_path / "snap") == tmp_path / "snap"

    def _raise_gated(**_kw: object) -> str:
        raise GatedRepoError("denied", response=MagicMock(status_code=403, headers={}))

    monkeypatch.setattr("ft.models.registry.snapshot_download", _raise_gated)
    with pytest.raises(PermissionError, match="Access denied for gated model"):
        reg.download_model("org/gated")


def test_estimate_params_from_config_edge_cases() -> None:
    assert SLMRegistry._estimate_params_from_config(SimpleNamespace()) == 0.0

    mm_cfg = SimpleNamespace(
        text_config=SimpleNamespace(
            hidden_size=256,
            num_hidden_layers=2,
            vocab_size=1000,
            tie_word_embeddings=False,
        )
    )
    assert SLMRegistry._estimate_params_from_config(mm_cfg) > 0.0
