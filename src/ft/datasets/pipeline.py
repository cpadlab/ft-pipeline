
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ft.datasets.ingest import DocumentChunk, PDFIngestor
from ft.datasets.serializer import JSONLSerializer
from ft.datasets.synthesizer import DatasetSynthesizer, SFTRecord


logger: Final[logging.Logger] = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetPipelineResult:

    source_path: Path
    output_path: Path
    chunks_extracted: int
    records_synthesized: int


class DatasetPipeline:
    """
    Orchestrates end-to-end SFT dataset generation from PDF documents.

    Coordinates document ingestion (`PDFIngestor`), asynchronous LLM synthesis
    (`DatasetSynthesizer`), and validated JSONL persistence (`JSONLSerializer`)
    behind a single cohesive entry point.

    Attributes:
        ingestor (Final[PDFIngestor]): Component responsible for PDF text extraction.
        synthesizer (Final[DatasetSynthesizer]): Component responsible for async QA generation.
        serializer (Final[JSONLSerializer]): Component responsible for JSONL validation and export.
    """

    def __init__(
        self,
        ingestor: PDFIngestor | None = None,
        synthesizer: DatasetSynthesizer | None = None,
        serializer: JSONLSerializer | None = None,
    ) -> None:
        """
        Initializes the DatasetPipeline with optional custom stage components.

        Args:
            ingestor (PDFIngestor | None, optional): Custom PDF ingestor instance.
                Instantiates a default `PDFIngestor()` if omitted. Defaults to None.
            synthesizer (DatasetSynthesizer | None, optional): Custom LLM synthesizer
                instance. Instantiates a default `DatasetSynthesizer()` using
                environment settings if omitted. Defaults to None.
            serializer (JSONLSerializer | None, optional): Custom JSONL serializer
                instance. Instantiates a default `JSONLSerializer()` if omitted.
                Defaults to None.
        """
        self.ingestor: Final[PDFIngestor] = ingestor or PDFIngestor()
        self.synthesizer: Final[DatasetSynthesizer] = synthesizer or DatasetSynthesizer()
        self.serializer: Final[JSONLSerializer] = serializer or JSONLSerializer()
        self._owns_synthesizer: Final[bool] = synthesizer is None


    async def run(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        overwrite: bool = False,
        append: bool = False,
        create_dirs: bool = True,
    ) -> DatasetPipelineResult:
        """
        Executes the full ingestion, synthesis, and serialization workflow.

        Args:
            input_path (str | Path): Path to a single `.pdf` file or a directory
                containing PDF documents.
            output_path (str | Path): Destination `.jsonl` file path or target
                output directory.
            overwrite (bool, optional): Whether to replace an existing output file
                when `append=False`. Defaults to False.
            append (bool, optional): Whether to append records to an existing `.jsonl`
                file. Defaults to False.
            create_dirs (bool, optional): Whether to create missing parent directories
                for `output_path`. Defaults to True.

        Returns:
            DatasetPipelineResult: Immutable execution summary containing extracted
                chunk count, synthesized record count, and resolved output path.
        """
        resolved_input = Path(input_path)
        logger.info("Starting dataset pipeline for source: %s", resolved_input)

        chunks = self._extract_chunks(resolved_input)
        records = await self._synthesize_records(chunks)
        saved_path = self._persist_records(
            records,
            output_path=output_path,
            overwrite=overwrite,
            append=append,
            create_dirs=create_dirs,
        )

        logger.info(
            "Pipeline completed: %d chunks -> %d SFT records saved at %s",
            len(chunks),
            len(records),
            saved_path,
        )

        return DatasetPipelineResult(
            source_path=resolved_input,
            output_path=saved_path,
            chunks_extracted=len(chunks),
            records_synthesized=len(records),
        )


    def _extract_chunks(self, source: Path) -> list[DocumentChunk]:
        """
        Extracts cleaned document chunks from the target PDF file or directory.

        Args:
            source (Path): Resolved path to a single PDF document or directory
                containing PDFs.

        Returns:
            list[DocumentChunk]: Validated document chunks extracted by the
                configured `PDFIngestor`.
        """
        chunks = self.ingestor.ingest_path(source)
        logger.debug("Extracted %d valid chunks from %s", len(chunks), source)
        return chunks


    async def _synthesize_records(self, chunks: list[DocumentChunk]) -> list[SFTRecord]:
        """
        Generates synthetic SFT records from extracted document chunks.

        Manages the asynchronous context lifecycle automatically if the pipeline
        instantiated its own internal `DatasetSynthesizer`; otherwise delegates
        directly to the caller-managed synthesizer instance.

        Args:
            chunks (list[DocumentChunk]): Document chunks to convert into
                conversational QA pairs.

        Returns:
            list[SFTRecord]: Synthesized conversational records ready for export.
        """
        if self._owns_synthesizer:
            async with self.synthesizer as active_synthesizer:
                return await active_synthesizer.synthesize_chunks(chunks)

        return await self.synthesizer.synthesize_chunks(chunks)


    def _persist_records(
        self,
        records: list[SFTRecord],
        *,
        output_path: str | Path,
        overwrite: bool,
        append: bool,
        create_dirs: bool,
    ) -> Path:
        """
        Validates, deduplicates, and writes synthesized SFT records to disk.

        Args:
            records (list[SFTRecord]): Synthesized SFT records to serialize.
            output_path (str | Path): Destination `.jsonl` file path or target
                output directory.
            overwrite (bool): Whether to replace an existing output file when
                `append=False`.
            append (bool): Whether to append records to an existing `.jsonl` file.
            create_dirs (bool): Whether to create missing parent directories
                automatically.

        Returns:
            Path: Resolved filesystem path to the written `.jsonl` artifact.
        """
        return self.serializer.save(
            records,
            output_path=output_path,
            overwrite=overwrite,
            append=append,
            create_dirs=create_dirs,
        )