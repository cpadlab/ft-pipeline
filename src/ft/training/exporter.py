
from __future__ import annotations
import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ft.config import get_settings


logger: Final[logging.Logger] = logging.getLogger(__name__)

ExportDType = Literal["bfloat16", "float16", "float32"]

EXPORT_DTYPE_MAP: Final[dict[ExportDType, torch.dtype]] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True, slots=True)
class ExportResult:
    """
    Artifacts generated after merging LoRA adapters into a standalone base model.
    """

    merged_model_dir: Path
    modelfile_path: Path | None
    dtype: str


class SLMAdapterExporter:
    """
    Merges trained LoRA adapters into a standalone SLM and prepares deployment artifacts.

    Loads the unquantized base model in half/full precision (required by PEFT for lossless
    weight merging), applies `merge_and_unload()`, saves standard `safetensors` shards
    alongside the tokenizer, and optionally emits an Ollama `Modelfile`.
    """

    def __init__(
        self,
        *,
        dtype: ExportDType = "bfloat16",
        device_map: str = "auto",
        hf_token: str | None = None,
    ) -> None:
        """
        Initializes the SLMAdapterExporter.

        Args:
            dtype (ExportDType, optional): Target precision for the merged weights
                (`"bfloat16"`, `"float16"`, or `"float32"`). Falls back to `"float16"`
                if `"bfloat16"` is unsupported on active CUDA hardware. Defaults to `"bfloat16"`.
            device_map (str, optional): Device placement strategy during merging (`"auto"`,
                `"cpu"`, `"cuda:0"`). Defaults to `"auto"`.
            hf_token (str | None, optional): Hugging Face token if the base model must be
                fetched from a gated repository. Falls back to `Settings.HF_TOKEN`.
                Defaults to None.
        """
        self.dtype: Final[torch.dtype] = self._resolve_dtype(dtype)
        self.dtype_name: Final[str] = str(self.dtype).removeprefix("torch.")
        self.device_map: Final[str] = device_map
        self._hf_token: Final[str | None] = self._resolve_hf_token(hf_token)


    @staticmethod
    def _resolve_hf_token(hf_token: str | None) -> str | None:
        """
        Resolves the Hugging Face authentication token from explicit input or settings.

        Args:
            hf_token (str | None): Explicit authentication token override, or None
                to read from `Settings.HF_TOKEN`.

        Returns:
            str | None: Unwrapped secret token string if configured, or None for
                unauthenticated public Hub access.
        """
        if hf_token is not None:
            return hf_token
        
        secret_token = get_settings().HF_TOKEN
        
        return secret_token.get_secret_value() if secret_token is not None else None


    @staticmethod
    def _resolve_dtype(requested: ExportDType) -> torch.dtype:
        """
        Maps an export dtype identifier to a `torch.dtype` with hardware fallback.

        Automatically falls back to `torch.float16` if `"bfloat16"` is requested
        on a CUDA device lacking native BF16 support.

        Args:
            requested (ExportDType): Requested floating-point precision string
                (`"bfloat16"`, `"float16"`, or `"float32"`).

        Returns:
            torch.dtype: Resolved PyTorch floating-point data type.
        """
        if requested == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            logger.warning("CUDA device lacks native bfloat16 support; exporting in float16.")
        
            return torch.float16
        
        return EXPORT_DTYPE_MAP[requested]


    def merge_and_export(
        self,
        *,
        base_model_id_or_path: str | Path,
        adapter_dir: str | Path,
        output_dir: str | Path,
        max_shard_size: str = "2GB",
        generate_ollama_modelfile: bool = True,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> ExportResult:
        """
        Fuses LoRA adapter weights into the base model and writes a standalone checkpoint.

        Args:
            base_model_id_or_path (str | Path): Local directory or Hub repository ID of
                the original unquantized base SLM.
            adapter_dir (str | Path): Directory containing `adapter_model.safetensors` and
                `adapter_config.json` produced by `SLMTrainer`.
            output_dir (str | Path): Destination directory for the merged standalone model.
            max_shard_size (str, optional): Maximum size per `.safetensors` shard file.
                Defaults to `"2GB"`.
            generate_ollama_modelfile (bool, optional): If True, creates a ready-to-import
                `Modelfile` inside `output_dir` for `ollama create`. Defaults to True.
            system_prompt (str | None, optional): Optional default `SYSTEM` instruction
                embedded in the generated Ollama `Modelfile`. Defaults to None.
            temperature (float, optional): Default sampling temperature written into the
                Ollama `Modelfile`. Defaults to 0.2.

        Returns:
            ExportResult: Paths to the merged model directory and optional `Modelfile`.

        Raises:
            FileNotFoundError: If `adapter_dir` does not exist or lacks `adapter_config.json`.
        """
        adapter_path = Path(adapter_dir)
        if not adapter_path.is_dir() or not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"Valid PEFT adapter directory not found at '{adapter_path}' "
                "(missing 'adapter_config.json')."
            )

        base_source = str(base_model_id_or_path)
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Loading unquantized base SLM '%s' in %s for LoRA merging...",
            base_source,
            self.dtype_name,
        )

        tokenizer = self._load_tokenizer(adapter_path=adapter_path, base_source=base_source)
        base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            base_source,
            torch_dtype=self.dtype,
            device_map=self.device_map,
            token=self._hf_token,
            trust_remote_code=False,
        )

        logger.info("Attaching LoRA adapter from '%s' and merging weights...", adapter_path)
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        merged_model: PreTrainedModel = peft_model.merge_and_unload()
        merged_model.config.use_cache = True

        logger.info("Saving standalone merged model to '%s'...", target_dir)
        merged_model.save_pretrained(
            str(target_dir),
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
        tokenizer.save_pretrained(str(target_dir))

        modelfile_path: Path | None = None
        if generate_ollama_modelfile:
            modelfile_path = self._write_ollama_modelfile(
                output_dir=target_dir,
                system_prompt=system_prompt,
                temperature=temperature,
            )

        self._release_memory(merged_model, peft_model, base_model)

        logger.info("Export completed successfully at '%s'.", target_dir)
        return ExportResult(
            merged_model_dir=target_dir,
            modelfile_path=modelfile_path,
            dtype=self.dtype_name,
        )


    def _load_tokenizer(
        self,
        *,
        adapter_path: Path,
        base_source: str,
    ) -> PreTrainedTokenizerBase:
        """
        Loads the tokenizer from the adapter directory or falls back to the base model.

        Prefers `adapter_path` when `tokenizer_config.json` is present to preserve
        any special tokens or chat template updates saved during training.

        Args:
            adapter_path (Path): Directory containing the trained PEFT adapter
                artifacts.
            base_source (str): Hugging Face repository ID or local path of the
                base language model.

        Returns:
            PreTrainedTokenizerBase: Loaded tokenizer instance.
        """
        source = str(adapter_path) if (adapter_path / "tokenizer_config.json").is_file() else base_source
        return AutoTokenizer.from_pretrained(
            source,
            token=self._hf_token,
            trust_remote_code=False,
        )


    @staticmethod
    def _write_ollama_modelfile(
        *,
        output_dir: Path,
        system_prompt: str | None,
        temperature: float,
    ) -> Path:
        """
        Writes an Ollama `Modelfile` pointing to the local safetensors directory.

        Args:
            output_dir (Path): Directory containing the merged SafeTensors model
                where the `Modelfile` will be written.
            system_prompt (str | None): Optional default system instruction
                embedded in the `SYSTEM` directive.
            temperature (float): Default sampling temperature configured via
                `PARAMETER temperature`.

        Returns:
            Path: Filesystem path to the generated `Modelfile`.
        """
        lines: list[str] = [
            f"FROM {output_dir.resolve()}",
            f"PARAMETER temperature {temperature}",
        ]
        if system_prompt:
            escaped_prompt = system_prompt.replace('"', '\\"')
            lines.append(f'SYSTEM "{escaped_prompt}"')

        modelfile_path = output_dir / "Modelfile"
        modelfile_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        
        return modelfile_path


    @staticmethod
    def _release_memory(*models: object) -> None:
        """
        Deletes model references and clears CUDA cache after export.

        Args:
            *models (object): Model and adapter instances to dereference before
                triggering garbage collection and `torch.cuda.empty_cache()`.
        """
        for model in models:
            del model
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()