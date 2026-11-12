"""
End-to-end SFT dataset generation and Hugging Face verification script.

Demonstrates executing `DatasetPipeline` on a source PDF document, exporting
synthesized conversational records to JSONL format, and verifying direct
compatibility with Hugging Face `datasets`.
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from datasets import load_dataset

from ft.datasets import DatasetPipeline


async def main() -> None:
    """
    Runs the SFT dataset generation pipeline and verifies the exported JSONL file.

    Ingests a source PDF document, synthesizes conversational question-answer
    pairs using environment-configured LLM settings, persists the output to
    disk, and validates schema loading via Hugging Face `datasets`.
    """
    input_pdf = Path("upload/example.pdf")
    output_jsonl = Path("data/processed/example.jsonl")

    pipeline = DatasetPipeline()
    result = await pipeline.run(
        input_path=input_pdf,
        output_path=output_jsonl,
        overwrite=True,
    )

    print(
        f"Pipeline completed: {result.chunks_extracted} chunks -> "
        f"{result.records_synthesized} records saved at '{result.output_path}'"
    )

    hf_dataset = load_dataset("json", data_files=str(result.output_path), split="train")
    print(f"Dataset verified in Hugging Face: {hf_dataset}")


if __name__ == "__main__":
    asyncio.run(main())