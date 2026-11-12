
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable
import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTTrainer

from ft.training.schemas import LoRASettings, SFTTrainingSettings


logger: Final[logging.Logger] = logging.getLogger(__name__)

REQUIRED_DATASET_COLUMN: Final[str] = "messages"
MIN_SAMPLES_FOR_EVAL_SPLIT: Final[int] = 10


@runtime_checkable
class TrainableModelBundle(Protocol):
    """
    Structural contract expected from a loaded SLM bundle.
    """

    @property
    def model(self) -> PreTrainedModel: ...

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase: ...

    @property
    def target_modules(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """
    Summary of a completed QLoRA Supervised Fine-Tuning run.
    """

    adapter_dir: Path
    train_samples: int
    eval_samples: int
    global_steps: int
    train_loss: float
    metrics: dict[str, float]


class SLMTrainer:
    """
    Orchestrates QLoRA Supervised Fine-Tuning (SFT) over conversational JSONL datasets.

    Connects a prepared `TrainableModelBundle` with `LoRASettings` and `SFTTrainingSettings`,
    validates dataset structure, executes training via `trl.SFTTrainer`, and persists
    the resulting PEFT adapter weights and tokenizer configuration.
    """

    def __init__(
        self,
        *,
        training_settings: SFTTrainingSettings | None = None,
        lora_settings: LoRASettings | None = None,
    ) -> None:
        """
        Initializes the SLMTrainer.

        Args:
            training_settings (SFTTrainingSettings | None, optional): Hyperparameters for
                `SFTTrainer`. Uses validated defaults if omitted. Defaults to None.
            lora_settings (LoRASettings | None, optional): LoRA adapter configuration.
                Uses validated defaults if omitted. Defaults to None.
        """
        self.training_settings: Final[SFTTrainingSettings] = (
            training_settings or SFTTrainingSettings()
        )
        self.lora_settings: Final[LoRASettings] = lora_settings or LoRASettings()


    def train(
        self,
        bundle: TrainableModelBundle,
        dataset_path: str | Path,
        *,
        resume_from_checkpoint: str | Path | None = None,
    ) -> TrainingResult:
        """
        Executes QLoRA fine-tuning on the provided model bundle and JSONL dataset.

        Args:
            bundle (TrainableModelBundle): Prepared quantized base model, tokenizer, and
                introspected linear target modules.
            dataset_path (str | Path): Path to the conversational `.jsonl` file containing
                a `"messages"` column.
            resume_from_checkpoint (str | Path | None, optional): Optional checkpoint directory
                from which to resume interrupted training. Defaults to None.

        Returns:
            TrainingResult: Output adapter directory path, split sizes, and training metrics.

        Raises:
            FileNotFoundError: If `dataset_path` does not exist.
            ValueError: If the dataset is empty, malformed, or lacks `"messages"`.
        """
        train_dataset, eval_dataset = self._load_and_split_dataset(Path(dataset_path))
        has_eval = eval_dataset is not None

        peft_config = self.lora_settings.to_peft_config(bundle.target_modules)
        sft_config = self._build_sft_config(has_eval_dataset=has_eval)

        self.training_settings.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting QLoRA SFT run (train_samples=%d, eval_samples=%d, r=%d, alpha=%d, targets=%s)...",
            len(train_dataset),
            len(eval_dataset) if eval_dataset is not None else 0,
            self.lora_settings.r,
            self.lora_settings.lora_alpha,
            peft_config.target_modules,
        )

        trainer = SFTTrainer(
            model=bundle.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=bundle.tokenizer,
            peft_config=peft_config,
        )

        checkpoint_arg = str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        train_output = trainer.train(resume_from_checkpoint=checkpoint_arg)

        adapter_dir = self.training_settings.output_dir
        trainer.save_model(str(adapter_dir))
        bundle.tokenizer.save_pretrained(str(adapter_dir))

        raw_metrics: dict[str, Any] = dict(train_output.metrics or {})
        clean_metrics = {
            k: float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))
        }

        logger.info(
            "QLoRA SFT completed in %d steps (train_loss=%.4f). Adapter saved to '%s'.",
            train_output.global_step,
            train_output.training_loss,
            adapter_dir,
        )

        return TrainingResult(
            adapter_dir=adapter_dir,
            train_samples=len(train_dataset),
            eval_samples=len(eval_dataset) if eval_dataset is not None else 0,
            global_steps=int(train_output.global_step),
            train_loss=float(train_output.training_loss),
            metrics=clean_metrics,
        )


    def _build_sft_config(self, *, has_eval_dataset: bool):
        """
        Builds `SFTConfig` adapting mixed-precision flags to the active device.

        Enables `bf16` automatically on CUDA devices with native bfloat16 support,
        and disables mixed-precision flags while falling back to `adamw_torch`
        when executing on CPU.

        Args:
            has_eval_dataset (bool): Whether a validation split is available to
                enable periodic evaluation during training.

        Returns:
            SFTConfig: Hardware-adapted TRL training configuration object.
        """
        cuda_available = torch.cuda.is_available()
        use_bf16 = cuda_available and torch.cuda.is_bf16_supported()

        sft_config = self.training_settings.to_trl_sft_config(
            use_bf16=use_bf16,
            has_eval_dataset=has_eval_dataset,
        )

        if not cuda_available:
            sft_config.bf16 = False
            sft_config.fp16 = False
            sft_config.optim = "adamw_torch"

        return sft_config


    def _load_and_split_dataset(
        self,
        dataset_path: Path,
    ) -> tuple[Dataset, Dataset | None]:
        """
        Loads a JSONL dataset, validates schema, and splits into train/eval sets.

        Verifies file existence, enforces the presence of the `"messages"` column,
        filters out empty or malformed conversational rows, and performs a seeded
        train/validation split when sample count meets `MIN_SAMPLES_FOR_EVAL_SPLIT`.

        Args:
            dataset_path (Path): Filesystem path to the source `.jsonl` dataset.

        Returns:
            tuple[Dataset, Dataset | None]: A tuple containing `(train_dataset, eval_dataset)`,
                where `eval_dataset` is None if splitting is disabled or the dataset
                contains fewer than `MIN_SAMPLES_FOR_EVAL_SPLIT` valid rows.

        Raises:
            FileNotFoundError: If `dataset_path` does not point to an existing file.
            ValueError: If the dataset is empty, lacks the `"messages"` column, or
                contains zero valid non-empty message lists after filtering.
        """
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Training dataset file not found: '{dataset_path}'")

        dataset: Dataset = load_dataset(
            "json",
            data_files=str(dataset_path),
            split="train",
        )

        if len(dataset) == 0:
            raise ValueError(f"Training dataset '{dataset_path}' is empty.")

        if REQUIRED_DATASET_COLUMN not in dataset.column_names:
            raise ValueError(
                f"Dataset '{dataset_path}' must contain a '{REQUIRED_DATASET_COLUMN}' column. "
                f"Found columns: {dataset.column_names}"
            )

        dataset = dataset.filter(
            lambda row: isinstance(row[REQUIRED_DATASET_COLUMN], list)
            and len(row[REQUIRED_DATASET_COLUMN]) > 0
        )
        if len(dataset) == 0:
            raise ValueError(f"No valid conversational rows found in '{dataset_path}'.")

        eval_ratio = self.training_settings.eval_split_ratio
        if eval_ratio <= 0.0:
            return dataset, None

        if len(dataset) < MIN_SAMPLES_FOR_EVAL_SPLIT:
            logger.warning(
                "Dataset has only %d samples (< %d); skipping validation split.",
                len(dataset),
                MIN_SAMPLES_FOR_EVAL_SPLIT,
            )
            return dataset, None

        split = dataset.train_test_split(
            test_size=eval_ratio,
            seed=self.training_settings.seed,
            shuffle=True,
        )
        
        return split["train"], split["test"]