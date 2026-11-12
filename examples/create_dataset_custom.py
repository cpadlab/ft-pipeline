"""Custom SFT dataset generation pipeline with explicit credentials and stage tuning.

Demonstrates injecting customized `PDFIngestor`, `DatasetSynthesizer`, and
`JSONLSerializer` instances into `DatasetPipeline` to process a directory of
PDF documents, manage the synthesizer's async context externally, and verify
the exported JSONL artifact with Hugging Face `datasets`.
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from datasets import load_dataset

from ft.datasets import (
    DatasetPipeline,
    DatasetSynthesizer,
    JSONLSerializer,
    PDFIngestor,
)


async def main() -> None:
    """
    Runs the SFT dataset pipeline with explicit credentials and custom stages.

    Configures custom PDF parsing thresholds, metadata-enriched JSONL export,
    and an externally managed `DatasetSynthesizer` context pointing to a local
    OpenAI-compatible endpoint before validating the output with Hugging Face
    `datasets`.
    """
    input_dir = Path("upload")
    output_dir = Path("data/processed")

    ingestor = PDFIngestor(
        heading_size_ratio=1.10,
        min_content_chars=150,
        filter_diagram_labels=True,
    )

    serializer = JSONLSerializer(
        include_metadata=True,
        deduplicate=True,
        default_filename="falcon.jsonl",
    )

    async with DatasetSynthesizer(
        api_key="<api-key>",
        base_url="http://localhost:8000/v1",
        model="qwen3.8-27b-q4",
        verify_ssl=False,
        max_concurrency=4,
        questions_per_chunk=4,
    ) as synthesizer:

        pipeline = DatasetPipeline(
            ingestor=ingestor,
            synthesizer=synthesizer,
            serializer=serializer,
        )

        result = await pipeline.run(
            input_path=input_dir,
            output_path=output_dir,
            overwrite=True,
        )

    print(
        f"Custom pipeline completed: {result.chunks_extracted} chunks -> "
        f"{result.records_synthesized} records saved at '{result.output_path}'"
    )

    hf_dataset = load_dataset("json", data_files=str(result.output_path), split="train")
    print(f"Dataset verified in Hugging Face: {hf_dataset}")


if __name__ == "__main__":
    asyncio.run(main())