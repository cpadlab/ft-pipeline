
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import bitsandbytes as bnb
import pytest
import torch

from ft.models.loader import SLMModelLoader
from ft.models.registry import SLMProfile


def test_loader_init_token_and_dtype_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_env_token_123456")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    loader = SLMModelLoader(quantization="4bit", compute_dtype="bfloat16")
    assert loader.compute_dtype == torch.float16
    assert loader._hf_token == "hf_env_token_123456"

    loader_explicit = SLMModelLoader(quantization="none", compute_dtype="float32", hf_token="hf_custom")
    assert loader_explicit.compute_dtype == torch.float32
    assert loader_explicit._hf_token == "hf_custom"


def test_validate_runtime_environment_raises_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loader = SLMModelLoader(quantization="4bit")

    with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA GPU"):
        loader.load("org/slm")


def test_build_quantization_config_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    cfg_4bit = SLMModelLoader(quantization="4bit")._build_quantization_config()
    assert cfg_4bit is not None and cfg_4bit.load_in_4bit is True

    cfg_8bit = SLMModelLoader(quantization="8bit")._build_quantization_config()
    assert cfg_8bit is not None and cfg_8bit.load_in_8bit is True

    cfg_none = SLMModelLoader(quantization="none")._build_quantization_config()
    assert cfg_none is None


def test_load_tokenizer_pad_token_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = SLMModelLoader(quantization="none")

    tok_no_pad = SimpleNamespace(pad_token=None, pad_token_id=None, eos_token="<eos>", eos_token_id=2, padding_side="left")
    monkeypatch.setattr("ft.models.loader.AutoTokenizer.from_pretrained", lambda *a, **kw: tok_no_pad)
    loaded = loader._load_tokenizer("org/m")
    assert loaded.pad_token == "<eos>"
    assert loaded.pad_token_id == 2
    assert loaded.padding_side == "right"

    tok_neither = SimpleNamespace(pad_token=None, eos_token=None)
    monkeypatch.setattr("ft.models.loader.AutoTokenizer.from_pretrained", lambda *a, **kw: tok_neither)
    with pytest.raises(ValueError, match="neither pad_token nor eos_token"):
        loader._load_tokenizer("org/m")


def test_load_end_to_end_kbit_and_unquantized(
    monkeypatch: pytest.MonkeyPatch,
    dummy_causal_model: torch.nn.Module,
    dummy_tokenizer: object,
    sample_slm_profile: SLMProfile,
) -> None:
    fake_registry = MagicMock()
    fake_registry.validate_model.return_value = sample_slm_profile

    monkeypatch.setattr("ft.models.loader.AutoTokenizer.from_pretrained", lambda *a, **kw: dummy_tokenizer)
    monkeypatch.setattr("ft.models.loader.AutoModelForCausalLM.from_pretrained", lambda *a, **kw: dummy_causal_model)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loader_none = SLMModelLoader(quantization="none", registry=fake_registry)
    bundle = loader_none.load("org/slm", prepare_for_training=False)
    assert bundle.target_modules == ("q_proj", "v_proj")
    assert dummy_causal_model.config.use_cache is False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    prep_mock = MagicMock(return_value=dummy_causal_model)
    monkeypatch.setattr("ft.models.loader.prepare_model_for_kbit_training", prep_mock)

    loader_4bit = SLMModelLoader(quantization="4bit", registry=fake_registry)
    bundle_4bit = loader_4bit.load("org/slm", prepare_for_training=True)
    assert bundle_4bit.quantization == "4bit"
    prep_mock.assert_called_once_with(dummy_causal_model, use_gradient_checkpointing=True)


def test_discover_target_modules_includes_bnb_layers() -> None:
    class MixedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = torch.nn.Linear(4, 4)
            self.up_proj = bnb.nn.Linear4bit(4, 4, bias=False)
            self.down_proj = bnb.nn.Linear8bitLt(4, 4, bias=False)
            self.lm_head = torch.nn.Linear(4, 8)

    modules = SLMModelLoader.discover_target_modules(MixedModel())
    assert modules == ("down_proj", "gate_proj", "up_proj")
