
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal
import bitsandbytes as bnb
import torch
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ft.config import get_settings
from ft.models.registry import DEFAULT_MAX_SLM_PARAMS_B, SLMProfile, SLMRegistry


logger: Final[logging.Logger] = logging.getLogger(__name__)

QuantizationMode = Literal["4bit", "8bit", "none"]
ComputeDType = Literal["bfloat16", "float16", "float32"]


DTYPE_MAP: Final[dict[ComputeDType, torch.dtype]] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

EXCLUDED_LORA_MODULES: Final[frozenset[str]] = frozenset({"lm_head"})


@dataclass(frozen=True, slots=True)
class LoadedModelBundle:
    """
    Container holding a prepared SLM, its tokenizer, and resolved architectural metadata.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    profile: SLMProfile
    quantization: QuantizationMode
    target_modules: tuple[str, ...]


class SLMModelLoader:
    """
    Loads and prepares Small Language Models (SLMs) for quantized fine-tuning or inference.

    Enforces SLM parameter ceilings via `SLMRegistry`, configures `bitsandbytes` 4-bit/8-bit
    quantization, prepares k-bit gradient flow, initializes the tokenizer safely, and
    dynamically introspects linear layer names for LoRA adaptation.

    Attributes:
        quantization (Final[QuantizationMode]): Active quantization strategy (`"4bit"`, `"8bit"`, `"none"`).
        compute_dtype (Final[torch.dtype]): Torch floating-point type used for matrix multiplications.
        device_map (Final[str]): Device placement strategy passed to `from_pretrained`.
        use_gradient_checkpointing (Final[bool]): Whether to enable gradient checkpointing in k-bit prep.
    """

    def __init__(
        self,
        *,
        quantization: QuantizationMode = "4bit",
        compute_dtype: ComputeDType = "bfloat16",
        device_map: str = "auto",
        use_gradient_checkpointing: bool = True,
        max_params_b: float = DEFAULT_MAX_SLM_PARAMS_B,
        hf_token: str | None = None,
        registry: SLMRegistry | None = None,
    ) -> None:
        """
        Initializes the SLMModelLoader.

        Args:
            quantization (QuantizationMode, optional): Quantization mode applied during
                weight loading (`"4bit"`, `"8bit"`, or `"none"`). Defaults to `"4bit"`.
            compute_dtype (ComputeDType, optional): Computation dtype (`"bfloat16"`,
                `"float16"`, or `"float32"`). Falls back to `"float16"` automatically if
                `"bfloat16"` is requested on hardware without native BF16 support.
                Defaults to `"bfloat16"`.
            device_map (str, optional): Target device mapping (e.g., `"auto"`, `"cuda:0"`,
                `"cpu"`). Defaults to `"auto"`.
            use_gradient_checkpointing (bool, optional): Enables gradient checkpointing
                during `prepare_model_for_kbit_training` to lower VRAM usage. Defaults to True.
            max_params_b (float, optional): Upper parameter limit in billions enforced when
                no custom `registry` is supplied. Defaults to `DEFAULT_MAX_SLM_PARAMS_B`.
            hf_token (str | None, optional): Explicit Hugging Face token. Falls back to
                `Settings.HF_TOKEN` when omitted. Defaults to None.
            registry (SLMRegistry | None, optional): Custom `SLMRegistry` instance used to
                validate models prior to loading. Defaults to None.
        """
        self.quantization: Final[QuantizationMode] = quantization
        self.compute_dtype: Final[torch.dtype] = self._resolve_compute_dtype(compute_dtype)
        self.device_map: Final[str] = device_map
        self.use_gradient_checkpointing: Final[bool] = use_gradient_checkpointing
        self._hf_token: Final[str | None] = self._resolve_hf_token(hf_token)
        self._registry: Final[SLMRegistry] = registry or SLMRegistry(
            max_params_b=max_params_b,
            hf_token=self._hf_token,
        )


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
    def _resolve_compute_dtype(requested: ComputeDType) -> torch.dtype:
        """
        Maps a compute dtype identifier to a `torch.dtype` with hardware fallback.

        Automatically downgrades `"bfloat16"` to `torch.float16` when running on a
        CUDA device lacking native BF16 tensor core support.

        Args:
            requested (ComputeDType): Requested floating-point precision string
                (`"bfloat16"`, `"float16"`, or `"float32"`).

        Returns:
            torch.dtype: Resolved PyTorch floating-point data type.
        """
        if requested == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            logger.warning("CUDA device does not support bfloat16 natively; falling back to float16.")
        
            return torch.float16
        
        return DTYPE_MAP[requested]


    def load(
        self,
        model_id_or_path: str | Path,
        *,
        prepare_for_training: bool = True,
    ) -> LoadedModelBundle:
        """
        Validates, loads, and prepares an SLM and its tokenizer from Hub or local storage.

        Args:
            model_id_or_path (str | Path): Hugging Face repository ID or local model directory.
            prepare_for_training (bool, optional): If True and quantization is active
                (`"4bit"` or `"8bit"`), freezes base weights and prepares normalization
                layers for k-bit LoRA training. Set to False when loading for inference or
                adapter merging. Defaults to True.

        Returns:
            LoadedModelBundle: Ready-to-use model, configured tokenizer, validated profile,
                and discovered linear `target_modules`.

        Raises:
            ValueError: If the model exceeds the SLM parameter ceiling.
            RuntimeError: If quantized loading (`"4bit"` / `"8bit"`) is requested without CUDA.
        """
        self._validate_runtime_environment()
        profile = self._registry.validate_model(model_id_or_path)
        source_str = str(model_id_or_path)

        logger.info(
            "Loading SLM '%s' (~%.2fB params, arch=%s, quant=%s)...",
            profile.source,
            profile.estimated_params_b,
            profile.model_type,
            self.quantization,
        )

        tokenizer = self._load_tokenizer(source_str)
        model = self._load_base_model(source_str, prepare_for_training=prepare_for_training)
        target_modules = self.discover_target_modules(model)

        logger.info(
            "SLM '%s' loaded successfully. Discovered LoRA target modules: %s",
            profile.source,
            target_modules,
        )

        return LoadedModelBundle(
            model=model,
            tokenizer=tokenizer,
            profile=profile,
            quantization=self.quantization,
            target_modules=target_modules,
        )


    def _validate_runtime_environment(self) -> None:
        """
        Verifies hardware availability required by the configured quantization mode.

        Raises:
            RuntimeError: If `"4bit"` or `"8bit"` quantization is configured on a
                system without an available NVIDIA CUDA GPU.
        """
        if self.quantization in ("4bit", "8bit") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Quantization mode '{self.quantization}' requires an NVIDIA CUDA GPU, "
                "but no CUDA device was detected. Use quantization='none' on CPU."
            )


    def _build_quantization_config(self) -> BitsAndBytesConfig | None:
        """Constructs the `bitsandbytes` quantization configuration for model loading.

        Configures NF4 double quantization for `"4bit"` mode, standard INT8
        quantization for `"8bit"` mode, or returns `None` for unquantized loading.

        Returns:
            BitsAndBytesConfig | None: Configured quantization specification, or
                None when `self.quantization == "none"`.
        """
        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
            )
        
        if self.quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        
        return None


    def _load_tokenizer(self, source: str) -> PreTrainedTokenizerBase:
        """
        Loads and normalizes the tokenizer with valid padding configuration.

        Ensures `pad_token` falls back to `eos_token` when unset and configures
        right-side padding for causal SFT training.

        Args:
            source (str): Hugging Face repository ID or local directory path.

        Returns:
            PreTrainedTokenizerBase: Initialized and padding-configured tokenizer.

        Raises:
            ValueError: If the tokenizer defines neither `pad_token` nor `eos_token`.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            token=self._hf_token,
            trust_remote_code=False,
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError(f"Tokenizer for '{source}' defines neither pad_token nor eos_token.")
            
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        tokenizer.padding_side = "right"
        
        return tokenizer


    def _load_base_model(
        self,
        source: str,
        *,
        prepare_for_training: bool,
    ) -> PreTrainedModel:
        """
        Instantiates the causal language model and optionally prepares k-bit training.

        Disables KV cache (`use_cache = False`) for training compatibility and wraps
        quantized models with `prepare_model_for_kbit_training` when requested.

        Args:
            source (str): Hugging Face repository ID or local directory path.
            prepare_for_training (bool): Whether to freeze base weights and cast
                normalization layers for k-bit LoRA gradient flow.

        Returns:
            PreTrainedModel: Loaded causal language model instance.
        """
        bnb_config = self._build_quantization_config()
        load_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "token": self._hf_token,
            "trust_remote_code": False,
        }

        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
        else:
            load_kwargs["torch_dtype"] = self.compute_dtype

        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
        model.config.use_cache = False

        if prepare_for_training and self.quantization in ("4bit", "8bit"):
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
            )

        return model


    @staticmethod
    def discover_target_modules(model: PreTrainedModel) -> tuple[str, ...]:
        """
        Introspects a loaded model graph and extracts all linear projection leaf names
        suitable for LoRA adaptation (excluding output head layers such as `lm_head`).

        Args:
            model (PreTrainedModel): Instantiated causal language model.

        Returns:
            tuple[str, ...]: Sorted unique leaf module names (e.g., `('down_proj', 'gate_proj', ...)`).
        """
        linear_classes: tuple[type[torch.nn.Module], ...] = (
            torch.nn.Linear,
            bnb.nn.Linear4bit,
            bnb.nn.Linear8bitLt,
        )

        discovered: set[str] = set()
        for module_name, module in model.named_modules():
            if isinstance(module, linear_classes):
                leaf_name = module_name.rsplit(".", maxsplit=1)[-1]
                
                if leaf_name not in EXCLUDED_LORA_MODULES:
                    discovered.add(leaf_name)

        return tuple(sorted(discovered))