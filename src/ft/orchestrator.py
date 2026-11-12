
from __future__ import annotations
import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final
import torch
from pydantic import BaseModel, ConfigDict, Field

from ft.models import (
    DEFAULT_MAX_SLM_PARAMS_B,
    ComputeDType,
    QuantizationMode,
    SLMModelLoader,
    SLMProfile,
    SLMRegistry,
)
from ft.training import (
    ExportDType,
    ExportResult,
    LoRASettings,
    SFTTrainingSettings,
    SLMAdapterExporter,
    SLMTrainer,
    TrainingResult,
)


logger: Final[logging.Logger] = logging.getLogger(__name__)


class PipelineConfig(BaseModel):
    """
    End-to-end declarative configuration for an SLM QLoRA fine-tuning and export workflow.
    """

    model_id_or_path: str | Path = Field(...)
    dataset_path: Path = Field(...)
    workspace_dir: Path = Field(default=Path("outputs/run"))
    cache_model_dir: Path | None = Field(default=None)
    max_params_b: float = Field(default=DEFAULT_MAX_SLM_PARAMS_B, gt=0.0)
    quantization: QuantizationMode = Field(default="4bit")
    compute_dtype: ComputeDType = Field(default="bfloat16")
    device_map: str = Field(default="auto")
    use_gradient_checkpointing: bool = Field(default=True)
    lora: LoRASettings = Field(default_factory=LoRASettings)
    training: SFTTrainingSettings | None = Field(default=None)
    merge_and_export: bool = Field(default=True)
    export_dtype: ExportDType = Field(default="bfloat16")
    export_device_map: str = Field(default="auto")
    generate_ollama_modelfile: bool = Field(default=True)
    ollama_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    system_prompt: str | None = Field(default=None)

    model_config = ConfigDict(
        frozen=True
    )


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    """
    Consolidated artifacts and metrics produced by a full `SLMPipelineOrchestrator` run.
    """

    profile: SLMProfile
    resolved_model_path: str | Path
    training: TrainingResult
    export: ExportResult | None


class SLMPipelineOrchestrator:
    """
    High-level library orchestrator coordinating the end-to-end SLM fine-tuning lifecycle.

    Executes validation, optional local caching, quantized model loading, QLoRA training,
    explicit VRAM reclamation, and standalone adapter merging without coupling to any CLI
    or entrypoint script.
    """

    def __init__(
        self,
        *,
        hf_token: str | None = None,
        registry: SLMRegistry | None = None,
        loader: SLMModelLoader | None = None,
        trainer: SLMTrainer | None = None,
        exporter: SLMAdapterExporter | None = None,
    ) -> None:
        """
        Initializes the orchestrator with optional pre-built components for dependency injection.

        Args:
            hf_token (str | None, optional): Optional Hugging Face token forwarded to default
                subcomponents. Falls back to `Settings.HF_TOKEN`. Defaults to None.
            registry (SLMRegistry | None, optional): Custom registry instance. Defaults to None.
            loader (SLMModelLoader | None, optional): Custom model loader instance. Defaults to None.
            trainer (SLMTrainer | None, optional): Custom SFT trainer instance. Defaults to None.
            exporter (SLMAdapterExporter | None, optional): Custom adapter exporter. Defaults to None.
        """
        self._hf_token: Final[str | None] = hf_token
        self._registry = registry
        self._loader = loader
        self._trainer = trainer
        self._exporter = exporter


    def run(
        self,
        config: PipelineConfig,
        *,
        resume_from_checkpoint: str | Path | None = None,
    ) -> PipelineRunResult:
        """
        Executes the full fine-tuning pipeline according to `config`.

        Args:
            config (PipelineConfig): Declarative configuration specifying source model,
                dataset, LoRA/SFT hyperparameters, and export options.
            resume_from_checkpoint (str | Path | None, optional): Optional checkpoint directory
                to resume an interrupted training run. Defaults to None.

        Returns:
            PipelineRunResult: Validated model profile, resolved model location, training
                summary, and optional merged export artifacts.
        """
        registry = self._registry or SLMRegistry(
            max_params_b=config.max_params_b,
            hf_token=self._hf_token,
        )

        profile = registry.validate_model(config.model_id_or_path)
        resolved_source = self._resolve_model_source(config, registry)

        loader = self._loader or SLMModelLoader(
            quantization=config.quantization,
            compute_dtype=config.compute_dtype,
            device_map=config.device_map,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            max_params_b=config.max_params_b,
            hf_token=self._hf_token,
            registry=registry,
        )
        bundle = loader.load(resolved_source, prepare_for_training=True)

        effective_training_settings = self._resolve_training_settings(config)
        trainer = self._trainer or SLMTrainer(
            training_settings=effective_training_settings,
            lora_settings=config.lora,
        )
        training_result = trainer.train(
            bundle=bundle,
            dataset_path=config.dataset_path,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        self._release_training_bundle(bundle)

        export_result: ExportResult | None = None
        if config.merge_and_export:
            exporter = self._exporter or SLMAdapterExporter(
                dtype=config.export_dtype,
                device_map=config.export_device_map,
                hf_token=self._hf_token,
            )
            merged_dir = config.workspace_dir / "merged"
            export_result = exporter.merge_and_export(
                base_model_id_or_path=resolved_source,
                adapter_dir=training_result.adapter_dir,
                output_dir=merged_dir,
                generate_ollama_modelfile=config.generate_ollama_modelfile,
                system_prompt=config.system_prompt,
                temperature=config.ollama_temperature,
            )

        logger.info("SLM fine-tuning pipeline completed successfully.")
        
        return PipelineRunResult(
            profile=profile,
            resolved_model_path=resolved_source,
            training=training_result,
            export=export_result,
        )


    @staticmethod
    def _resolve_model_source(
        config: PipelineConfig,
        registry: SLMRegistry,
    ) -> str | Path:
        """
        Downloads Hub weights into `cache_model_dir` if requested and not yet local.

        Returns the local path directly if `config.model_id_or_path` already
        exists on disk; otherwise triggers a validated snapshot download when
        `cache_model_dir` is specified, or returns the remote repository ID.

        Args:
            config (PipelineConfig): Active pipeline configuration.
            registry (SLMRegistry): Validated SLM registry used to perform
                guarded downloads.

        Returns:
            str | Path: Local directory path or remote Hugging Face repository ID.
        """
        candidate_path = Path(config.model_id_or_path)
        if candidate_path.exists():
            return candidate_path

        if config.cache_model_dir is not None:
            return registry.download_model(
                model_id=str(config.model_id_or_path),
                local_dir=config.cache_model_dir,
            )

        return str(config.model_id_or_path)


    @staticmethod
    def _resolve_training_settings(config: PipelineConfig) -> SFTTrainingSettings:
        """
        Ensures adapter outputs land inside `config.workspace_dir / 'adapter'` by default.

        Instantiates default `SFTTrainingSettings` scoped to the workspace when
        `config.training` is omitted, or redirects the default adapter output
        path into `workspace_dir` while preserving custom explicit paths.

        Args:
            config (PipelineConfig): Active pipeline configuration.

        Returns:
            SFTTrainingSettings: Resolved SFT training hyperparameters with
                workspace-aligned `output_dir`.
        """
        default_adapter_dir = config.workspace_dir / "adapter"
        if config.training is None:
            return SFTTrainingSettings(output_dir=default_adapter_dir)

        if config.training.output_dir == Path("outputs/slm-lora-adapter"):
            return config.training.model_copy(update={"output_dir": default_adapter_dir})

        return config.training


    @staticmethod
    def _release_training_bundle(bundle: object) -> None:
        """
        Frees quantized model weights from GPU memory prior to 16-bit adapter merging.

        Dereferences the loaded model bundle, invokes Python garbage collection,
        and clears the PyTorch CUDA allocator cache to prevent out-of-memory
        errors during full/half-precision weight merging.

        Args:
            bundle (object): Loaded model bundle instance to release.
        """
        del bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()