
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from ft.training.exporter import SLMAdapterExporter


def test_exporter_init_token_and_dtype_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_env_export_123456")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    exporter = SLMAdapterExporter(dtype="bfloat16")
    assert exporter.dtype == torch.float16
    assert exporter.dtype_name == "float16"
    assert exporter._hf_token == "hf_env_export_123456"

    explicit = SLMAdapterExporter(dtype="float32", hf_token="hf_custom")
    assert explicit.dtype == torch.float32
    assert explicit._hf_token == "hf_custom"


def test_merge_and_export_missing_adapter_raises(tmp_path: Path) -> None:
    exporter = SLMAdapterExporter()
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        exporter.merge_and_export(
            base_model_id_or_path="org/base",
            adapter_dir=tmp_path / "missing_adapter",
            output_dir=tmp_path / "merged",
        )


def test_merge_and_export_success_and_no_modelfile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dummy_causal_model: torch.nn.Module,
    dummy_tokenizer: object,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    fake_peft_model = MagicMock()
    fake_peft_model.merge_and_unload.return_value = dummy_causal_model

    tok_sources: list[str] = []

    def _fake_tok(src: str, **_kw: object) -> object:
        tok_sources.append(src)
        return dummy_tokenizer

    monkeypatch.setattr("ft.training.exporter.AutoTokenizer.from_pretrained", _fake_tok)
    monkeypatch.setattr(
        "ft.training.exporter.AutoModelForCausalLM.from_pretrained",
        lambda *_a, **_kw: dummy_causal_model,
    )
    monkeypatch.setattr(
        "ft.training.exporter.PeftModel.from_pretrained",
        lambda *_a, **_kw: fake_peft_model,
    )

    empty_cache = MagicMock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    exporter = SLMAdapterExporter(dtype="float16", device_map="cpu")
    out_dir = tmp_path / "merged"
    res = exporter.merge_and_export(
        base_model_id_or_path="org/base",
        adapter_dir=adapter_dir,
        output_dir=out_dir,
        generate_ollama_modelfile=True,
        system_prompt='You are "helpful".',
        temperature=0.3,
    )

    assert tok_sources == [str(adapter_dir)]
    assert res.merged_model_dir == out_dir
    assert res.modelfile_path == out_dir / "Modelfile"
    modelfile_text = res.modelfile_path.read_text(encoding="utf-8")
    assert 'SYSTEM "You are \\"helpful\\"."' in modelfile_text
    assert "PARAMETER temperature 0.3" in modelfile_text
    empty_cache.assert_called_once()

    (adapter_dir / "tokenizer_config.json").unlink()
    out_dir2 = tmp_path / "merged2"
    res2 = exporter.merge_and_export(
        base_model_id_or_path="org/base",
        adapter_dir=adapter_dir,
        output_dir=out_dir2,
        generate_ollama_modelfile=False,
    )
    assert tok_sources[-1] == "org/base"
    assert res2.modelfile_path is None

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out_dir3 = tmp_path / "merged3"
    res3 = exporter.merge_and_export(
        base_model_id_or_path="org/base",
        adapter_dir=adapter_dir,
        output_dir=out_dir3,
        generate_ollama_modelfile=True,
        system_prompt=None,
    )
    assert res3.modelfile_path is not None
    assert "SYSTEM" not in res3.modelfile_path.read_text(encoding="utf-8")
