
from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from peft import LoraConfig, TaskType
from pydantic import BaseModel, ConfigDict, Field, model_validator
from trl import SFTConfig


OptimizerName = Literal["paged_adamw_8bit", "adamw_torch", "adamw_8bit"]
SchedulerName = Literal["cosine", "linear", "constant_with_warmup"]


class LoRASettings(BaseModel):
    """
    Validated configuration for Low-Rank Adaptation (LoRA) adapters.

    Converts cleanly into a `peft.LoraConfig` while allowing automatic inheritance
    of introspected target modules when `target_modules` is omitted.
    """

    r: int = Field(default=16, ge=2, le=128)
    lora_alpha: int = Field(default=32, ge=1, le=256)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: tuple[str, ...] | None = Field(default=None)
    bias: Literal["none", "all", "lora_only"] = Field(default="none")

    def to_peft_config(self, fallback_target_modules: Sequence[str]) -> LoraConfig:
        """
        Builds a `peft.LoraConfig` using explicit target modules or loader-discovered defaults.

        Args:
            fallback_target_modules (Sequence[str]): Linear leaf module names introspected
                from the base model when `self.target_modules` is None.

        Returns:
            LoraConfig: Ready-to-use PEFT configuration for causal language modeling.

        Raises:
            ValueError: If neither `self.target_modules` nor `fallback_target_modules` is non-empty.
        """
        resolved_targets = list(self.target_modules or fallback_target_modules)
        
        if not resolved_targets:
            raise ValueError("At least one LoRA target module must be specified or discovered.")

        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=resolved_targets,
            bias=self.bias,
        )

    model_config = ConfigDict(
        frozen=True
    )


class SFTTrainingSettings(BaseModel):
    """
    Validated hyperparameters for Supervised Fine-Tuning (SFT) of SLMs on consumer GPUs.

    Defaults are tuned for QLoRA stability (4-bit base + paged 8-bit AdamW optimizer,
    cosine learning rate decay, and gradient accumulation).
    """

    output_dir: Path = Field(default=Path("outputs/slm-lora-adapter"))
    num_train_epochs: float = Field(default=3.0, gt=0.0, le=50.0)
    per_device_train_batch_size: int = Field(default=2, ge=1, le=64)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=128)
    learning_rate: float = Field(default=2e-4, gt=0.0, lt=1.0)
    warmup_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    max_seq_length: int = Field(default=2048, ge=128, le=32768)
    optim: OptimizerName = Field(default="paged_adamw_8bit")
    lr_scheduler_type: SchedulerName = Field(default="cosine")
    logging_steps: int = Field(default=5, ge=1)
    save_strategy: Literal["no", "steps", "epoch"] = Field(default="epoch")
    save_total_limit: int = Field(default=2, ge=1, le=20)
    packing: bool = Field(default=False)
    seed: int = Field(default=42, ge=0)
    eval_split_ratio: float = Field(default=0.1, ge=0.0, lt=0.5)

    @model_validator(mode="after")
    def _validate_eval_save_alignment(self) -> SFTTrainingSettings:
        """
        Ensures save_strategy remains valid when evaluation split is disabled.
        """
        if self.eval_split_ratio == 0.0 and self.save_strategy == "no":
            return self
        
        return self


    def to_trl_sft_config(self, *, use_bf16: bool, has_eval_dataset: bool) -> SFTConfig:
        """
        Translates validated settings into a `trl.SFTConfig` instance.

        Args:
            use_bf16 (bool): Whether hardware supports native `bfloat16` mixed precision.
            has_eval_dataset (bool): Whether a validation split is active for periodic eval.

        Returns:
            SFTConfig: Configured training arguments ready for `trl.SFTTrainer`.
        """
        eval_strategy = self.save_strategy if (has_eval_dataset and self.save_strategy != "no") else "no"

        return SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_ratio,
            weight_decay=self.weight_decay,
            max_length=self.max_seq_length,
            optim=self.optim,
            lr_scheduler_type=self.lr_scheduler_type,
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            eval_strategy=eval_strategy,
            save_total_limit=self.save_total_limit,
            packing=self.packing,
            bf16=use_bf16,
            fp16=not use_bf16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
            seed=self.seed,
            dataset_text_field=None,
        )

    model_config = ConfigDict(
        frozen=True
    )