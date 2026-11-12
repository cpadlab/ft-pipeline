

from __future__ import annotations

import pytest
from peft import TaskType

from ft.training.schemas import LoRASettings, SFTTrainingSettings


def test_lora_settings_to_peft_config() -> None:
    cfg_fallback = LoRASettings().to_peft_config(["q_proj", "v_proj"])
    assert cfg_fallback.task_type == TaskType.CAUSAL_LM
    assert set(cfg_fallback.target_modules) == {"q_proj", "v_proj"}

    cfg_explicit = LoRASettings(target_modules=("k_proj",)).to_peft_config(["q_proj"])
    assert set(cfg_explicit.target_modules) == {"k_proj"}

    with pytest.raises(ValueError, match="At least one LoRA target module"):
        LoRASettings().to_peft_config([])


def test_sft_training_settings_to_trl_sft_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_no_eval = SFTTrainingSettings(eval_split_ratio=0.0, save_strategy="no")
    sft_cfg = settings_no_eval.to_trl_sft_config(use_bf16=False, has_eval_dataset=False)
    assert sft_cfg.eval_strategy == "no"
    assert sft_cfg.warmup_steps == settings_no_eval.warmup_ratio
    assert sft_cfg.bf16 is False
    assert sft_cfg.fp16 is True

    monkeypatch.setattr("transformers.training_args.is_torch_bf16_gpu_available", lambda: True)
    settings_eval = SFTTrainingSettings(warmup_ratio=0.1, save_strategy="epoch")
    sft_cfg_eval = settings_eval.to_trl_sft_config(use_bf16=True, has_eval_dataset=True)
    assert sft_cfg_eval.eval_strategy == "epoch"
    assert sft_cfg_eval.warmup_steps == 0.1
    assert sft_cfg_eval.bf16 is True
    assert sft_cfg_eval.fp16 is False
