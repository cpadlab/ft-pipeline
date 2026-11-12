from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from huggingface_hub import HfApi, ModelInfo, snapshot_download
from transformers import AutoConfig, PretrainedConfig

from ft.config import get_settings

logger: Final[logging.Logger] = logging.getLogger(__name__)

DEFAULT_MAX_SLM_PARAMS_B: Final[float] = 4.5
BILLION: Final[float] = 1_000_000_000.0

SortField = Literal["downloads", "likes", "lastModified", "trendingScore"]


@dataclass(frozen=True, slots=True)
class SLMProfile:
    """
    Validated metadata for any Small Language Model discovered locally or on Hub.
    """

    source: str
    model_type: str
    estimated_params_b: float
    gated: bool = False
    downloads: int = 0
    likes: int = 0
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SLMPage:
    """
    Paginated response slice returned by `SLMRegistry.search_models`.
    """

    items: list[SLMProfile]
    page: int
    page_size: int
    has_more: bool


class SLMRegistry:
    """
    Dynamically discovers, filters, validates, and downloads SLMs from Hugging Face Hub or local paths.

    Avoids hardcoded architecture allowlists by querying Hub metadata dynamically with
    streaming pagination and inspecting model configurations on the fly to enforce
    parameter-count ceilings.

    Attributes:
        max_params_b (Final[float]): Upper parameter bound in billions allowed for an SLM.
    """

    def __init__(
        self,
        max_params_b: float = DEFAULT_MAX_SLM_PARAMS_B,
        hf_token: str | None = None,
    ) -> None:
        """
        Initializes the SLMRegistry.

        Args:
            max_params_b (float, optional): Maximum allowed model size in billions
                of parameters. Must be positive. Defaults to `DEFAULT_MAX_SLM_PARAMS_B` (4.5).
            hf_token (str | None, optional): Hugging Face read token for gated or
                private repositories. Falls back to `Settings.HF_TOKEN` if omitted.
                Defaults to None.

        Raises:
            ValueError: If `max_params_b <= 0.0`.
        """
        if max_params_b <= 0.0:
            raise ValueError("max_params_b must be greater than 0.0")

        self.max_params_b: Final[float] = max_params_b
        self._hf_token: Final[str | None] = self._resolve_hf_token(hf_token)
        self._api: Final[HfApi] = HfApi(token=self._hf_token)


    @staticmethod
    def _resolve_hf_token(hf_token: str | None) -> str | None:
        """Resolves the Hugging Face authentication token from explicit input or settings.

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


    def iter_models(
        self,
        *,
        query: str | None = None,
        author: str | None = None,
        tags: Sequence[str] | None = None,
        min_params_b: float = 0.05,
        max_params_b: float | None = None,
        sort: SortField = "downloads",
        require_safetensors: bool = True,
    ) -> Iterator[SLMProfile]:
        """
        Lazily streams validated SLM profiles from Hugging Face Hub matching dynamic filters.

        Automatically paginates over the Hub API without loading unbounded result sets
        into memory.

        Args:
            query (str | None, optional): Free-text search query (e.g., `"Instruct"`,
                `"Coder"`, `"falcon"`). Defaults to None.
            author (str | None, optional): Filter by Hub organization or username.
                Defaults to None.
            tags (Sequence[str] | None, optional): Additional Hub filter tags (e.g.,
                `["conversational"]`, `["es"]`). Defaults to None.
            min_params_b (float, optional): Minimum parameter count in billions.
                Defaults to 0.05.
            max_params_b (float | None, optional): Maximum parameter count in billions;
                cannot exceed `self.max_params_b`. Defaults to `self.max_params_b`.
            sort (SortField, optional): Hub sorting criterion. Defaults to `"downloads"`.
            require_safetensors (bool, optional): If True, filters for models exposing
                `safetensors` metadata for reliable parameter verification. Defaults to True.

        Yields:
            SLMProfile: Next matching and size-verified SLM profile.

        Raises:
            ValueError: If the parameter range is invalid or exceeds `self.max_params_b`.
        """
        upper_bound = self.max_params_b if max_params_b is None else max_params_b
        self._validate_param_bounds(min_params_b, upper_bound)

        filter_tags = list(tags) if tags else []
        if "transformers" not in filter_tags:
            filter_tags.append("transformers")
        
        if require_safetensors and "safetensors" not in filter_tags:
            filter_tags.append("safetensors")

        hub_stream = self._api.list_models(
            search=query,
            author=author,
            filter=filter_tags,
            pipeline_tag="text-generation",
            sort=sort,
            expand=["safetensors", "config", "gated", "downloads", "likes", "tags"],
        )

        for info in hub_stream:
            profile = self._profile_from_hub_info(info, min_params_b, upper_bound)
            
            if profile is not None:
                yield profile


    def search_models(
        self,
        *,
        query: str | None = None,
        author: str | None = None,
        tags: Sequence[str] | None = None,
        min_params_b: float = 0.05,
        max_params_b: float | None = None,
        sort: SortField = "downloads",
        page: int = 1,
        page_size: int = 20,
    ) -> SLMPage:
        """
        Returns a deterministic page of validated SLM profiles matching the given filters.

        Args:
            query (str | None, optional): Free-text search term. Defaults to None.
            author (str | None, optional): Hub organization or user filter. Defaults to None.
            tags (Sequence[str] | None, optional): Optional Hub tags filter. Defaults to None.
            min_params_b (float, optional): Lower parameter bound in billions. Defaults to 0.05.
            max_params_b (float | None, optional): Upper parameter bound in billions.
                Defaults to `self.max_params_b`.
            sort (SortField, optional): Ranking order on Hub. Defaults to `"downloads"`.
            page (int, optional): 1-based page index to retrieve. Defaults to 1.
            page_size (int, optional): Number of items per page (1 to 100). Defaults to 20.

        Returns:
            SLMPage: Paginated container with `items`, `page`, `page_size`, and `has_more`.

        Raises:
            ValueError: If `page < 1`, `page_size` is out of `[1, 100]`, or bounds are invalid.
        """
        if page < 1:
            raise ValueError("page must be greater than or equal to 1")
        
        if not (1 <= page_size <= 100):
            raise ValueError("page_size must be between 1 and 100")

        stream = self.iter_models(
            query=query,
            author=author,
            tags=tags,
            min_params_b=min_params_b,
            max_params_b=max_params_b,
            sort=sort,
        )

        offset = (page - 1) * page_size
        window = list(itertools.islice(stream, offset, offset + page_size + 1))
        has_more = len(window) > page_size
        items = window[:page_size]

        return SLMPage(
            items=items,
            page=page,
            page_size=page_size,
            has_more=has_more,
        )


    def validate_model(self, model_id_or_path: str | Path) -> SLMProfile:
        """
        Inspects model metadata (Hub repo or local folder) and enforces the SLM size ceiling.

        Checks Hub metadata first when a repository ID is provided so gated models exceeding
        `max_params_b` are rejected immediately without requiring authentication.

        Args:
            model_id_or_path (str | Path): Hugging Face repository ID or local directory path.

        Returns:
            SLMProfile: Validated metadata profile for the target model.

        Raises:
            ValueError: If the model exceeds `self.max_params_b`.
            PermissionError: If accessing a gated/private model without valid `HF_TOKEN`.
        """
        local_candidate = Path(model_id_or_path)
        if not local_candidate.exists():
            hub_profile = self._inspect_hub_metadata(str(model_id_or_path))
            if hub_profile is not None:
                
                if hub_profile.estimated_params_b > self.max_params_b:
                    raise ValueError(
                        f"Model '{hub_profile.source}' has ~{hub_profile.estimated_params_b:.2f}B "
                        f"parameters, exceeding the SLM ceiling of {self.max_params_b:.2f}B."
                    )
                
                return hub_profile

        return self._validate_via_autoconfig(str(model_id_or_path))


    def _inspect_hub_metadata(self, repo_id: str) -> SLMProfile | None:
        """
        Fetches lightweight public Hub metadata without downloading repository files.

        Queries `HfApi.model_info` with expanded `safetensors` and `config`
        fields to build an unfiltered `SLMProfile` prior to ceiling enforcement.

        Args:
            repo_id (str): Hugging Face repository identifier (e.g.,
                `"Qwen/Qwen2.5-3B-Instruct"`).

        Returns:
            SLMProfile | None: Extracted metadata profile if the repository exists
                and exposes valid parameter metadata; otherwise None.
        """
        try:
            info = self._api.model_info(
                repo_id=repo_id,
                expand=["safetensors", "config", "gated", "downloads", "likes", "tags"],
            )
        except Exception:
            return None

        return self._profile_from_hub_info(info, min_params_b=0.0, max_params_b=float("inf"))


    def _validate_via_autoconfig(self, source_str: str) -> SLMProfile:
        """Loads `AutoConfig` locally or from Hub and enforces the parameter ceiling.

        Estimates parameter count directly from architectural dimensions when
        Hub `safetensors` metadata is unavailable or the source is a local directory.

        Args:
            source_str (str): Hugging Face repository ID or local directory path.

        Returns:
            SLMProfile: Validated metadata profile with architecture type and
                estimated parameter count.

        Raises:
            PermissionError: If the repository is gated or private and lacks a
                valid `HF_TOKEN` or accepted license.
            ValueError: If the estimated parameter count exceeds `self.max_params_b`.
            OSError: If the configuration cannot be resolved locally or remotely.
        """
        try:
            
            config: PretrainedConfig = AutoConfig.from_pretrained(
                source_str,
                token=self._hf_token,
                trust_remote_code=False,
            )

        except OSError as exc:
            if "gated repo" in str(exc).lower() or "401" in str(exc):
                raise PermissionError(
                    f"Model '{source_str}' is gated or private. Provide a valid HF_TOKEN "
                    "and accept the model license on Hugging Face Hub."
                ) from exc

            raise

        model_type = str(getattr(config, "model_type", "unknown")).lower()
        estimated_b = self._estimate_params_from_config(config)

        if estimated_b > self.max_params_b:
            raise ValueError(
                f"Model '{source_str}' has ~{estimated_b:.2f}B parameters, exceeding the "
                f"SLM ceiling of {self.max_params_b:.2f}B."
            )

        return SLMProfile(
            source=source_str,
            model_type=model_type,
            estimated_params_b=round(estimated_b, 2),
        )
    

    def download_model(
        self,
        model_id: str,
        local_dir: str | Path | None = None,
    ) -> Path:
        """
        Validates that a Hub repository meets the SLM ceiling and downloads its weights.

        Args:
            model_id (str): Hugging Face repository identifier.
            local_dir (str | Path | None, optional): Optional destination directory.
                Defaults to the standard Hugging Face cache when None.

        Returns:
            Path: Local filesystem path to the downloaded snapshot directory.

        Raises:
            ValueError: If the model exceeds `self.max_params_b`.
        """
        profile = self.validate_model(model_id)

        logger.info(
            "Downloading validated SLM '%s' (~%.2fB params, arch=%s)...",
            profile.source,
            profile.estimated_params_b,
            profile.model_type,
        )

        snapshot_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir) if local_dir is not None else None,
            token=self._hf_token,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "original/*"],
        )

        return Path(snapshot_path)


    def _validate_param_bounds(self, min_params_b: float, max_params_b: float) -> None:
        """Validates requested lower and upper parameter bounds against registry limits.

        Args:
            min_params_b (float): Lower parameter threshold in billions.
            max_params_b (float): Upper parameter threshold in billions.

        Raises:
            ValueError: If `min_params_b < 0.0`, `max_params_b < min_params_b`,
                or `max_params_b > self.max_params_b`.
        """
        if min_params_b < 0.0 or max_params_b < min_params_b:
            raise ValueError("Invalid parameter range: ensure 0 <= min_params_b <= max_params_b")
        
        if max_params_b > self.max_params_b:
            raise ValueError(
                f"Requested max_params_b ({max_params_b}B) exceeds registry limit ({self.max_params_b}B)"
            )


    @staticmethod
    def _profile_from_hub_info(
        info: ModelInfo,
        min_params_b: float,
        max_params_b: float,
    ) -> SLMProfile | None:
        """Builds an `SLMProfile` from Hub metadata if parameter count falls within bounds.

        Extracts total parameter count from `safetensors` metadata and filters
        out models lacking parameter counts or falling outside `[min_params_b, max_params_b]`.

        Args:
            info (ModelInfo): Hugging Face Hub model metadata object.
            min_params_b (float): Minimum allowed parameter count in billions.
            max_params_b (float): Maximum allowed parameter count in billions.

        Returns:
            SLMProfile | None: Populated profile if valid and within bounds;
                otherwise None.
        """
        safetensors_meta = getattr(info, "safetensors", None)
        total_params = getattr(safetensors_meta, "total", None) if safetensors_meta else None
        
        if not isinstance(total_params, int) or total_params <= 0:
            return None

        params_b = total_params / BILLION
        if not (min_params_b <= params_b <= max_params_b):
            return None

        config_meta: dict[str, Any] = getattr(info, "config", None) or {}
        model_type = str(config_meta.get("model_type", "unknown")).lower()
        raw_tags = getattr(info, "tags", None) or ()

        return SLMProfile(
            source=info.id,
            model_type=model_type,
            estimated_params_b=round(params_b, 2),
            gated=bool(getattr(info, "gated", False)),
            downloads=int(getattr(info, "downloads", 0) or 0),
            likes=int(getattr(info, "likes", 0) or 0),
            tags=tuple(str(tag) for tag in raw_tags),
        )


    @staticmethod
    def _estimate_params_from_config(config: PretrainedConfig) -> float:
        """
        Estimates parameter count in billions from standard Transformer config attributes.

        Computes total parameter count across token embeddings (accounting for
        weight tying), Grouped-Query/Multi-Head Attention projections, and
        SwiGLU-style feed-forward layers.

        Args:
            config (PretrainedConfig): Loaded Hugging Face model configuration
                (or multimodal configuration exposing `text_config`).

        Returns:
            float: Estimated model size in billions of parameters, or `0.0` if
                core dimensions (`hidden_size` or `num_hidden_layers`) are missing.
        """
        text_cfg = getattr(config, "text_config", config)
        hidden = int(getattr(text_cfg, "hidden_size", 0) or 0)
        layers = int(getattr(text_cfg, "num_hidden_layers", 0) or 0)
        vocab = int(getattr(text_cfg, "vocab_size", 0) or 0)
        intermediate = int(getattr(text_cfg, "intermediate_size", hidden * 4) or 0)
        num_heads = int(getattr(text_cfg, "num_attention_heads", 1) or 1)
        kv_heads = int(getattr(text_cfg, "num_key_value_heads", num_heads) or num_heads)

        if hidden == 0 or layers == 0:
            return 0.0

        head_dim = hidden // max(num_heads, 1)
        attn_params = hidden * (num_heads * head_dim + 2 * kv_heads * head_dim + hidden)
        mlp_params = 3 * hidden * intermediate
        tie_embeddings = bool(getattr(text_cfg, "tie_word_embeddings", True))
        embed_params = vocab * hidden * (1 if tie_embeddings else 2)

        total = embed_params + layers * (attn_params + mlp_params)
        
        return total / BILLION