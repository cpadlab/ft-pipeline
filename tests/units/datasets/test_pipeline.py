
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from ft.datasets.ingest import DocumentChunk
from ft.datasets.pipeline import DatasetPipeline
from ft.datasets.synthesizer import SFTRecord


def test_pipeline_default_components_and_owned_synthesizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_chunks: list[DocumentChunk],
    sample_sft_records: list[SFTRecord],
) -> None:
    ingestor = MagicMock()
    ingestor.ingest_path.return_value = sample_chunks

    serializer = MagicMock()
    expected_out = tmp_path / "out.jsonl"
    serializer.save.return_value = expected_out

    synth = MagicMock()
    synth.__aenter__ = AsyncMock(return_value=synth)
    synth.__aexit__ = AsyncMock(return_value=None)
    synth.synthesize_chunks = AsyncMock(return_value=sample_sft_records)

    monkeypatch.setattr("ft.datasets.pipeline.PDFIngestor", lambda: ingestor)
    monkeypatch.setattr("ft.datasets.pipeline.DatasetSynthesizer", lambda: synth)
    monkeypatch.setattr("ft.datasets.pipeline.JSONLSerializer", lambda: serializer)

    pipeline = DatasetPipeline()
    result = asyncio.run(pipeline.run(tmp_path / "in.pdf", expected_out, overwrite=True))

    assert result.chunks_extracted == 2
    assert result.records_synthesized == 2
    assert result.output_path == expected_out
    synth.__aenter__.assert_awaited_once()
    synth.__aexit__.assert_awaited_once()


def test_pipeline_injected_synthesizer_does_not_manage_context(
    tmp_path: Path,
    sample_chunks: list[DocumentChunk],
    sample_sft_records: list[SFTRecord],
) -> None:
    ingestor = MagicMock()
    ingestor.ingest_path.return_value = sample_chunks
    serializer = MagicMock()
    serializer.save.return_value = tmp_path / "out.jsonl"

    synth = MagicMock()
    synth.__aenter__ = AsyncMock()
    synth.synthesize_chunks = AsyncMock(return_value=sample_sft_records)

    pipeline = DatasetPipeline(ingestor=ingestor, synthesizer=synth, serializer=serializer)
    result = asyncio.run(pipeline.run(tmp_path / "in.pdf", tmp_path / "out.jsonl"))

    assert result.records_synthesized == 2
    synth.__aenter__.assert_not_called()
    synth.synthesize_chunks.assert_awaited_once_with(sample_chunks)
