from __future__ import annotations
import argparse
from pathlib import Path

from ft.models import DEFAULT_MAX_SLM_PARAMS_B


def _add_dataset_arguments(parser: argparse.ArgumentParser, *, required_pdf: bool = True) -> None:
    """
    Registers Phase 1 PDF-to-JSONL dataset synthesis options on an argument parser.

    Args:
        parser (argparse.ArgumentParser): Target subcommand parser.
        required_pdf (bool, optional): Whether `--pdf-dir` is mandatory.
            Defaults to True.
    """
    group = parser.add_argument_group("Dataset Generation Options")

    group.add_argument(
        "--pdf-dir",
        type=Path,
        required=required_pdf,
        help="Path to a source PDF file or directory containing PDF documents.",
    )
    group.add_argument(
        "--dataset-out",
        type=Path,
        default=Path("data/dataset.jsonl"),
        help="Output path or directory for the generated JSONL dataset (default: data/dataset.jsonl).",
    )
    group.add_argument(
        "--heading-size-ratio",
        type=float,
        default=1.12,
        help="Minimum font-size ratio over body text to detect section headings (default: 1.12).",
    )
    group.add_argument(
        "--min-content-chars",
        type=int,
        default=100,
        help="Minimum cleaned character count required to keep a page chunk (default: 100).",
    )
    group.add_argument(
        "--keep-diagram-labels",
        action="store_true",
        help="Keep short isolated diagram labels instead of filtering them out.",
    )
    group.add_argument(
        "--pairs-per-chunk",
        type=int,
        default=3,
        help="Number of Q&A pairs synthesized per chunk (default: 3).",
    )
    group.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum concurrent LLM API requests during dataset synthesis (default: 5).",
    )
    group.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include source provenance metadata alongside `messages` in each JSONL record.",
    )
    group.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable deduplication of identical synthesized user prompts.",
    )
    group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output JSONL file if it already exists.",
    )
    group.add_argument(
        "--append",
        action="store_true",
        help="Append synthesized records to an existing JSONL file.",
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Registers base SLM selection, caching, and quantization options.

    Args:
        parser (argparse.ArgumentParser): Target subcommand parser.
    """
    group = parser.add_argument_group("Model & Quantization Options")

    group.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face repo ID (e.g. 'google/gemma-2-2b-it') or local model directory.",
    )
    group.add_argument(
        "--cache-model-dir",
        type=Path,
        default=None,
        help="Optional local folder to download/cache Hub weights before loading.",
    )
    group.add_argument(
        "--max-params-b",
        type=float,
        default=DEFAULT_MAX_SLM_PARAMS_B,
        help=f"Maximum allowed SLM parameter ceiling in billions (default: {DEFAULT_MAX_SLM_PARAMS_B}).",
    )
    group.add_argument(
        "--quantization",
        choices=("4bit", "8bit", "none"),
        default="4bit",
        help="Base model quantization mode (default: 4bit).",
    )
    group.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Computation dtype during training (default: bfloat16).",
    )
    group.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device placement map passed to transformers (default: auto).",
    )
    group.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (faster, higher VRAM usage).",
    )


def _add_lora_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Registers LoRA adapter hyperparameter options on an argument parser.

    Args:
        parser (argparse.ArgumentParser): Target subcommand parser.
    """
    group = parser.add_argument_group("LoRA Adapter Options")
    group.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA attention rank dimension r (default: 16).",
    )
    group.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA scaling factor alpha (default: 32).",
    )
    group.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="Dropout probability for LoRA layers (default: 0.05).",
    )
    group.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help="Explicit linear target modules (default: auto-detected from model architecture).",
    )
    group.add_argument(
        "--lora-bias",
        choices=("none", "all", "lora_only"),
        default="none",
        help="Bias training strategy for PEFT (default: none).",
    )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Registers SFT optimization, scheduling, and checkpointing options.

    Args:
        parser (argparse.ArgumentParser): Target subcommand parser.
    """
    group = parser.add_argument_group("SFT Training Hyperparameters")
    group.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path("outputs/run"),
        help="Root output directory for adapters and merged model (default: outputs/run).",
    )
    group.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Number of training epochs (default: 3.0).",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device training batch size (default: 2).",
    )
    group.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4).",
    )
    group.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Peak learning rate (default: 2e-4).",
    )
    group.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Learning rate warmup fraction (default: 0.05).",
    )
    group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay regularization (default: 0.01).",
    )
    group.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length in tokens (default: 2048).",
    )
    group.add_argument(
        "--optim",
        choices=("paged_adamw_8bit", "adamw_torch", "adamw_8bit"),
        default="paged_adamw_8bit",
        help="Optimizer implementation (default: paged_adamw_8bit).",
    )
    group.add_argument(
        "--lr-scheduler",
        choices=("cosine", "linear", "constant_with_warmup"),
        default="cosine",
        help="Learning rate scheduler type (default: cosine).",
    )
    group.add_argument(
        "--logging-steps",
        type=int,
        default=5,
        help="Log training metrics every N steps (default: 5).",
    )
    group.add_argument(
        "--save-strategy",
        choices=("no", "steps", "epoch"),
        default="epoch",
        help="Checkpoint saving strategy (default: epoch).",
    )
    group.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints kept on disk (default: 2).",
    )
    group.add_argument(
        "--packing",
        action="store_true",
        help="Enable sequence packing for short conversational samples.",
    )
    group.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Validation split fraction between 0.0 and 0.5 (default: 0.1).",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    group.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint directory to resume interrupted training.",
    )


def _add_export_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Registers adapter merging and Ollama artifact export options.

    Args:
        parser (argparse.ArgumentParser): Target subcommand parser.
    """
    group = parser.add_argument_group("Merge & Export Options")

    group.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip merging LoRA weights into a standalone model after training.",
    )
    group.add_argument(
        "--export-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Target precision for merged standalone weights (default: bfloat16).",
    )
    group.add_argument(
        "--export-device-map",
        type=str,
        default="auto",
        help="Device map used during 16-bit weight merging (default: auto).",
    )
    group.add_argument(
        "--no-ollama-modelfile",
        action="store_true",
        help="Do not generate an Ollama Modelfile alongside the merged model.",
    )
    group.add_argument(
        "--ollama-temperature",
        type=float,
        default=0.2,
        help="Default sampling temperature in the generated Ollama Modelfile (default: 0.2).",
    )
    group.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional SYSTEM prompt written into the Ollama Modelfile.",
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Constructs the top-level CLI argument parser with all `ft` subcommands.

    Returns:
        argparse.ArgumentParser: Configured parser supporting `dataset`,
            `train`, `run`, `export`, and `models` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="ft",
        description="End-to-end Small Language Model (SLM) dataset synthesis, QLoRA fine-tuning, and export CLI.",
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face read token (overrides HF_TOKEN in .env).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Generate a conversational SFT JSONL dataset from PDF documents.",
    )
    _add_dataset_arguments(dataset_parser, required_pdf=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Run QLoRA fine-tuning (and optional merge/export) using an existing JSONL dataset.",
    )
    train_parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the input conversational `.jsonl` training dataset.",
    )

    _add_model_arguments(train_parser)
    _add_lora_arguments(train_parser)
    _add_training_arguments(train_parser)
    _add_export_arguments(train_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="Execute the full end-to-end pipeline: PDF synthesis -> QLoRA training -> merge & export.",
    )

    _add_dataset_arguments(run_parser, required_pdf=True)
    _add_model_arguments(run_parser)
    _add_lora_arguments(run_parser)
    _add_training_arguments(run_parser)
    _add_export_arguments(run_parser)

    export_parser = subparsers.add_parser(
        "export",
        help="Merge an existing LoRA adapter with its base SLM and generate deployment artifacts.",
    )
    export_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base SLM repository ID or local directory.",
    )
    export_parser.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
        help="Directory containing trained PEFT adapter weights.",
    )
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the standalone merged model.",
    )

    _add_export_arguments(export_parser)

    models_parser = subparsers.add_parser(
        "models",
        help="Discover and validate candidate SLMs on Hugging Face Hub.",
    )
    models_parser.add_argument("--query", type=str, default=None, help="Search query string.")
    models_parser.add_argument("--author", type=str, default=None, help="Filter by Hub author/org.")
    models_parser.add_argument("--min-params-b", type=float, default=0.5, help="Minimum params (B).")
    models_parser.add_argument(
        "--max-params-b",
        type=float,
        default=DEFAULT_MAX_SLM_PARAMS_B,
        help="Maximum params (B).",
    )
    models_parser.add_argument("--page", type=int, default=1, help="Page number (default: 1).")
    models_parser.add_argument("--page-size", type=int, default=10, help="Results per page (default: 10).")

    return parser
