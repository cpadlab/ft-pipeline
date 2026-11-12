from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ft.datasets.ingest import DocumentChunk
from ft.datasets.synthesizer import (
    DatasetSynthesizer,
    QAPair,
    _ChunkSynthesisResponse,
)


def test_synthesizer_init_validation() -> None:
    with pytest.raises(ValueError, match="questions_per_chunk"):
        DatasetSynthesizer(questions_per_chunk=0)

    with pytest.raises(ValueError, match="temperature"):
        DatasetSynthesizer(temperature=2.5)


def test_resolve_connection_params_explicit_vs_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ft.datasets.synthesizer.get_settings",
        lambda: (_ for _ in ()).throw(AssertionError("should not call get_settings")),
    )
    resolved = DatasetSynthesizer._resolve_connection_params(
        api_key="k",
        base_url="http://localhost:8000/v1",
        model="m",
        verify_ssl=False,
        max_concurrency=2,
    )
    assert resolved == ("k", "http://localhost:8000/v1", "m", False, 2)


def test_synthesize_chunks_empty_and_populated(
    sample_chunks: list[DocumentChunk],
) -> None:
    async def _run() -> None:
        async with DatasetSynthesizer(
            api_key="test-key-123456",
            base_url="http://localhost:8000/v1",
            model="test-model",
            verify_ssl=True,
            max_concurrency=2,
            questions_per_chunk=1,
        ) as synth:
            assert await synth.synthesize_chunks([]) == []

            parsed_payload = _ChunkSynthesisResponse(
                pairs=[
                    QAPair(
                        category="implementation",
                        instruction="How do you run the pipeline?",
                        output="Call pipeline.run(input_path, output_path).",
                    )
                ]
            )
            fake_completion = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed_payload))]
            )
            synth._client.beta.chat.completions.parse = AsyncMock(return_value=fake_completion)

            records = await synth.synthesize_chunks(sample_chunks)
            assert len(records) == 2
            assert records[0].chunk_id == "doc_p001"
            assert records[0].messages[0].role == "user"
            assert records[0].messages[1].role == "assistant"

    asyncio.run(_run())


def test_request_structured_pairs_raises_when_parsed_is_none(
    sample_chunks: list[DocumentChunk],
) -> None:
    async def _run() -> None:
        async with DatasetSynthesizer() as synth:
            refused = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=None))]
            )
            synth._client.beta.chat.completions.parse = AsyncMock(return_value=refused)

            with pytest.raises(RuntimeError, match="unparseable or refused"):
                await synth.synthesize_single(sample_chunks[0])

    asyncio.run(_run())
