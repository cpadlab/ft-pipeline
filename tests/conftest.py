
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from ft.config import get_settings
from ft.datasets.ingest import DocumentChunk
from ft.datasets.synthesizer import ChatMessage, SFTRecord
from ft.models.registry import SLMProfile


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """
    Clears the cached `Settings` singleton and sets safe default AI env vars.
    """
    get_settings.cache_clear()
    monkeypatch.setenv("AI_API_KEY", "test-api-key-123456")
    monkeypatch.setenv("AI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("AI_MODEL", "test-synth-model")
    monkeypatch.setenv("AI_VERIFY_SSL", "true")
    monkeypatch.setenv("AI_MAX_CONCURRENCY", "4")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    yield
    get_settings.cache_clear()


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """
    Returns representative `DocumentChunk` instances for synthesis and pipeline tests.
    """
    return [
        DocumentChunk(
            chunk_id="doc_p001",
            source_file="doc.pdf",
            section_title="Introduction",
            content="This section explains the core architecture and design choices in detail.",
            page_start=1,
            page_end=1,
        ),
        DocumentChunk(
            chunk_id="doc_p002",
            source_file="doc.pdf",
            section_title="Debugging",
            content="Common configuration mistakes include mismatched tensor dtypes and OOM errors.",
            page_start=2,
            page_end=2,
        ),
    ]


@pytest.fixture
def sample_sft_records() -> list[SFTRecord]:
    """
    Returns valid `SFTRecord` instances for serializer and dataset tests.
    """
    return [
        SFTRecord(
            chunk_id="doc_p001",
            category="architecture",
            messages=[
                ChatMessage(role="user", content="Explain the core architecture."),
                ChatMessage(role="assistant", content="The architecture separates ingestion, synthesis, and training."),
            ],
        ),
        SFTRecord(
            chunk_id="doc_p002",
            category="debugging",
            messages=[
                ChatMessage(role="user", content="How do I debug dtype mismatches?"),
                ChatMessage(role="assistant", content="Ensure compute_dtype matches hardware capabilities."),
            ],
        ),
    ]


@pytest.fixture
def sample_slm_profile() -> SLMProfile:
    """
    Returns a validated `SLMProfile` fixture.
    """
    return SLMProfile(
        source="Qwen/Qwen2.5-0.5B-Instruct",
        model_type="qwen2",
        estimated_params_b=0.5,
        gated=False,
        downloads=12345,
        likes=100,
        tags=("transformers", "safetensors"),
    )


class _DummyCausalModel(torch.nn.Module):
    """
    Minimal `torch.nn.Module` mimicking a causal LM with linear layers and `lm_head`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.v_proj = torch.nn.Linear(4, 4, bias=False)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.config = SimpleNamespace(use_cache=True)
        self.saved_calls: list[tuple[str, dict[str, Any]]] = []

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        self.saved_calls.append((save_directory, kwargs))
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / "model.safetensors").write_bytes(b"dummy")


class _DummyTokenizer:
    """
    Minimal tokenizer stub supporting padding attributes and `save_pretrained`.
    """

    def __init__(self, *, pad_token: str | None = "<pad>", eos_token: str | None = "</s>") -> None:
        self.pad_token = pad_token
        self.pad_token_id = 0 if pad_token is not None else None
        self.eos_token = eos_token
        self.eos_token_id = 1 if eos_token is not None else None
        self.padding_side = "left"
        self.saved_dirs: list[str] = []

    def save_pretrained(self, save_directory: str) -> None:
        self.saved_dirs.append(save_directory)
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


@pytest.fixture
def dummy_causal_model() -> _DummyCausalModel:
    """
    Provides a minimal causal LM stub for loader and exporter tests.
    """
    return _DummyCausalModel()


@pytest.fixture
def dummy_tokenizer() -> _DummyTokenizer:
    """
    Provides a minimal tokenizer stub for loader, trainer, and exporter tests.
    """
    return _DummyTokenizer()
