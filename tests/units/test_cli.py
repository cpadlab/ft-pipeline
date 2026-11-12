
from __future__ import annotations

from pathlib import Path

import pytest

from ft.cli import build_parser


def test_build_parser_dataset_subcommand_defaults_and_overrides() -> None:
    parser = build_parser()

    args = parser.parse_args(["dataset", "--pdf-dir", "pdfs"])
    assert args.command == "dataset"
    assert args.pdf_dir == Path("pdfs")
    assert args.dataset_out == Path("data/dataset.jsonl")
    assert args.heading_size_ratio == 1.12
    assert args.min_content_chars == 100
    assert args.keep_diagram_labels is False
    assert args.pairs_per_chunk == 3
    assert args.max_concurrency is None
    assert args.include_metadata is False
    assert args.no_deduplicate is False
    assert args.overwrite is False
    assert args.append is False
    assert args.ai_base_url is None
    assert args.ai_api_key is None
    assert args.ai_model is None
    assert args.ai_temperature == 0.2
    assert args.ai_no_verify_ssl is False

    custom = parser.parse_args(
        [
            "--hf-token",
            "hf_abc",
            "-v",
            "dataset",
            "--pdf-dir",
            "upload",
            "--dataset-out",
            "out/data.jsonl",
            "--heading-size-ratio",
            "1.25",
            "--min-content-chars",
            "50",
            "--keep-diagram-labels",
            "--pairs-per-chunk",
            "5",
            "--max-concurrency",
            "10",
            "--include-metadata",
            "--no-deduplicate",
            "--overwrite",
            "--ai-base-url",
            "http://127.0.0.1:9000/v1",
            "--ai-api-key",
            "key-override",
            "--ai-model",
            "qwen-72b",
            "--ai-temperature",
            "0.7",
            "--ai-no-verify-ssl",
        ]
    )
    assert custom.hf_token == "hf_abc"
    assert custom.verbose is True
    assert custom.keep_diagram_labels is True
    assert custom.max_concurrency == 10
    assert custom.include_metadata is True
    assert custom.no_deduplicate is True
    assert custom.overwrite is True
    assert custom.ai_base_url == "http://127.0.0.1:9000/v1"
    assert custom.ai_api_key == "key-override"
    assert custom.ai_model == "qwen-72b"
    assert custom.ai_temperature == 0.7
    assert custom.ai_no_verify_ssl is True


def test_build_parser_train_and_run_subcommands() -> None:
    parser = build_parser()

    train_args = parser.parse_args(
        [
            "train",
            "--dataset",
            "data/train.jsonl",
            "--model",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--target-modules",
            "q_proj",
            "v_proj",
            "--packing",
            "--no-gradient-checkpointing",
            "--no-merge",
            "--no-ollama-modelfile",
        ]
    )
    assert train_args.command == "train"
    assert train_args.dataset == Path("data/train.jsonl")
    assert train_args.target_modules == ["q_proj", "v_proj"]
    assert train_args.packing is True
    assert train_args.no_gradient_checkpointing is True
    assert train_args.no_merge is True
    assert train_args.no_ollama_modelfile is True

    run_args = parser.parse_args(
        ["run", "--pdf-dir", "pdfs", "--model", "models/slm", "--system-prompt", "Be concise."]
    )
    assert run_args.command == "run"
    assert run_args.pdf_dir == Path("pdfs")
    assert run_args.system_prompt == "Be concise."


def test_build_parser_export_and_models_subcommands() -> None:
    parser = build_parser()

    export_args = parser.parse_args(
        [
            "export",
            "--model",
            "models/base",
            "--adapter-dir",
            "outputs/adapter",
            "--output-dir",
            "outputs/merged",
            "--export-dtype",
            "float16",
            "--ollama-temperature",
            "0.4",
        ]
    )
    assert export_args.command == "export"
    assert export_args.export_dtype == "float16"
    assert export_args.ollama_temperature == 0.4

    models_args = parser.parse_args(
        ["models", "--query", "coder", "--author", "Qwen", "--page", "2", "--page-size", "15"]
    )
    assert models_args.command == "models"
    assert models_args.query == "coder"
    assert models_args.author == "Qwen"
    assert models_args.page == 2
    assert models_args.page_size == 15


def test_build_parser_requires_subcommand() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
