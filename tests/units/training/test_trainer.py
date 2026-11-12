
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from ft.training.schemas import LoRASettings, SFTTrainingSettings
from ft.training.trainer import SLMTrainer


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_trainer_train_and_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dummy_causal_model: torch.nn.Module,
    dummy_tokenizer: object,
) -> None:
    ds_path = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
            for i in range(12)
        ],
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    fake_trl_trainer = MagicMock()
    fake_trl_trainer.train.return_value = SimpleNamespace(
        global_step=6,
        training_loss=0.42,
        metrics={"train_loss": 0.42, "note": "ignored"},
    )
    captured_init: dict[str, object] = {}

    def _fake_sft_trainer(**kwargs: object) -> MagicMock:
        captured_init.update(kwargs)
        return fake_trl_trainer

    monkeypatch.setattr("ft.training.trainer.SFTTrainer", _fake_sft_trainer)

    bundle = SimpleNamespace(
        model=dummy_causal_model,
        tokenizer=dummy_tokenizer,
        target_modules=("q_proj", "v_proj"),
    )
    out_dir = tmp_path / "adapter_out"
    trainer = SLMTrainer(
        training_settings=SFTTrainingSettings(output_dir=out_dir, eval_split_ratio=0.2),
        lora_settings=LoRASettings(),
    )
    result = trainer.train(bundle=bundle, dataset_path=ds_path, resume_from_checkpoint=tmp_path / "ckpt")

    assert result.adapter_dir == out_dir
    assert result.train_samples + result.eval_samples == 12
    assert result.eval_samples > 0
    assert result.global_steps == 6
    assert result.train_loss == 0.42
    assert result.metrics == {"train_loss": 0.42}
    assert captured_init["args"].optim == "adamw_torch"
    fake_trl_trainer.save_model.assert_called_once_with(str(out_dir))


def test_load_and_split_dataset_edge_cases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trainer = SLMTrainer(training_settings=SFTTrainingSettings(eval_split_ratio=0.1))

    with pytest.raises(FileNotFoundError):
        trainer._load_and_split_dataset(tmp_path / "missing.jsonl")

    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    from datasets import Dataset, load_dataset as real_load_dataset

    monkeypatch.setattr(
        "ft.training.trainer.load_dataset",
        lambda *_a, **_kw: Dataset.from_dict({"messages": []}),
    )
    with pytest.raises(ValueError, match="is empty"):
        trainer._load_and_split_dataset(empty_file)

    monkeypatch.setattr("ft.training.trainer.load_dataset", real_load_dataset)

    wrong_col = _write_jsonl(tmp_path / "wrong.jsonl", [{"text": "hello"}])
    with pytest.raises(ValueError, match="must contain a 'messages' column"):
        trainer._load_and_split_dataset(wrong_col)

    all_invalid = _write_jsonl(tmp_path / "invalid.jsonl", [{"messages": []}, {"messages": None}])
    with pytest.raises(ValueError, match="No valid conversational rows"):
        trainer._load_and_split_dataset(all_invalid)

    small_ds = _write_jsonl(
        tmp_path / "small.jsonl",
        [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]} for _ in range(3)],
    )
    train_ds, eval_ds = trainer._load_and_split_dataset(small_ds)
    assert len(train_ds) == 3
    assert eval_ds is None

    trainer_no_eval = SLMTrainer(training_settings=SFTTrainingSettings(eval_split_ratio=0.0))
    train_ds2, eval_ds2 = trainer_no_eval._load_and_split_dataset(small_ds)
    assert len(train_ds2) == 3
    assert eval_ds2 is None


def test_build_sft_config_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr("transformers.training_args.is_torch_bf16_gpu_available", lambda: True)

    cfg = SLMTrainer()._build_sft_config(has_eval_dataset=True)
    assert cfg.bf16 is True
    assert cfg.fp16 is False
