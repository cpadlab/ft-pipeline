
from __future__ import annotations
import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from ft.cli import build_parser
from ft.datasets import (
    DatasetPipeline,
    DatasetSynthesizer,
    JSONLSerializer,
    PDFIngestor,
)
from ft.models import SLMRegistry
from ft.orchestrator import PipelineConfig, SLMPipelineOrchestrator
from ft.training import LoRASettings, SFTTrainingSettings, SLMAdapterExporter


def _configure_logging(verbose: bool) -> None:
    """
    Configures root logging format and verbosity level for CLI execution.

    Args:
        verbose (bool): If True, sets the logging level to `DEBUG`; otherwise
            uses `INFO`.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


async def _run_dataset_step(args: argparse.Namespace) -> Path:
    """
    Executes the Phase 1 PDF-to-JSONL dataset synthesis step.

    Initializes `PDFIngestor`, `DatasetSynthesizer`, and `JSONLSerializer` from
    parsed CLI arguments, runs `DatasetPipeline`, and returns the saved JSONL
    artifact path.

    Args:
        args (argparse.Namespace): Parsed CLI arguments containing dataset
            ingestion, synthesis, and serialization options.

    Returns:
        Path: Resolved filesystem path to the generated `.jsonl` dataset.
    """
    ingestor = PDFIngestor(
        heading_size_ratio=args.heading_size_ratio,
        min_content_chars=args.min_content_chars,
        filter_diagram_labels=not args.keep_diagram_labels,
    )
    serializer = JSONLSerializer(
        include_metadata=args.include_metadata,
        deduplicate=not args.no_deduplicate,
    )

    async with DatasetSynthesizer(
        api_key=args.ai_api_key,
        base_url=args.ai_base_url,
        model=args.ai_model,
        verify_ssl=False if args.ai_no_verify_ssl else None,
        max_concurrency=args.max_concurrency,
        questions_per_chunk=args.pairs_per_chunk,
        temperature=args.ai_temperature,
    ) as synthesizer:
        pipeline = DatasetPipeline(
            ingestor=ingestor,
            synthesizer=synthesizer,
            serializer=serializer,
        )

        result = await pipeline.run(
            input_path=args.pdf_dir,
            output_path=args.dataset_out,
            overwrite=args.overwrite,
            append=args.append,
        )

    return result.output_path


def _build_pipeline_config(args: argparse.Namespace, *, dataset_path: Path) -> PipelineConfig:
    """
    Builds a validated `PipelineConfig` from parsed CLI arguments.

    Args:
        args (argparse.Namespace): Parsed CLI arguments from `train` or `run`.
        dataset_path (Path): Path to the conversational `.jsonl` training file.

    Returns:
        PipelineConfig: Complete configuration for `SLMPipelineOrchestrator`.
    """
    lora_settings = LoRASettings(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=tuple(args.target_modules) if args.target_modules else None,
        bias=args.lora_bias,
    )

    training_settings = SFTTrainingSettings(
        output_dir=args.workspace_dir / "adapter",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_seq_length=args.max_seq_length,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        packing=args.packing,
        eval_split_ratio=args.eval_split,
        seed=args.seed,
    )

    return PipelineConfig(
        model_id_or_path=args.model,
        dataset_path=dataset_path,
        workspace_dir=args.workspace_dir,
        cache_model_dir=args.cache_model_dir,
        max_params_b=args.max_params_b,
        quantization=args.quantization,
        compute_dtype=args.compute_dtype,
        device_map=args.device_map,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        lora=lora_settings,
        training=training_settings,
        merge_and_export=not args.no_merge,
        export_dtype=args.export_dtype,
        export_device_map=args.export_device_map,
        generate_ollama_modelfile=not args.no_ollama_modelfile,
        ollama_temperature=args.ollama_temperature,
        system_prompt=args.system_prompt,
    )


def _handle_train_or_run(args: argparse.Namespace, *, dataset_path: Path) -> None:
    """
    Executes the model loading, QLoRA training, and optional export stages.

    Args:
        args (argparse.Namespace): Parsed CLI arguments from `train` or `run`.
        dataset_path (Path): Path to the conversational `.jsonl` training file.
    """
    config = _build_pipeline_config(args, dataset_path=dataset_path)
    orchestrator = SLMPipelineOrchestrator(hf_token=args.hf_token)
    result = orchestrator.run(
        config,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    print("\n=== PIPELINE EXECUTION SUMMARY ===")
    print(f"Model:          {result.profile.source} (~{result.profile.estimated_params_b}B, {result.profile.model_type})")
    print(f"Train / Eval:   {result.training.train_samples} / {result.training.eval_samples} samples")
    print(f"Global Steps:   {result.training.global_steps}")
    print(f"Train Loss:     {result.training.train_loss:.4f}")
    print(f"LoRA Adapter:   {result.training.adapter_dir.resolve()}")

    if result.export is not None:
        print(f"Merged Model:   {result.export.merged_model_dir.resolve()} ({result.export.dtype})")
        if result.export.modelfile_path is not None:
            print(f"Ollama File:    {result.export.modelfile_path.resolve()}")


def _handle_export(args: argparse.Namespace) -> None:
    """
    Merges a trained LoRA adapter into a standalone model and prints export paths.

    Args:
        args (argparse.Namespace): Parsed CLI arguments from the `export` subcommand.
    """
    exporter = SLMAdapterExporter(
        dtype=args.export_dtype,
        device_map=args.export_device_map,
        hf_token=args.hf_token,
    )

    result = exporter.merge_and_export(
        base_model_id_or_path=args.model,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        generate_ollama_modelfile=not args.no_ollama_modelfile,
        system_prompt=args.system_prompt,
        temperature=args.ollama_temperature,
    )
    
    print("\n=== EXPORT SUMMARY ===")
    print(f"Merged Model:   {result.merged_model_dir.resolve()} ({result.dtype})")
    if result.modelfile_path is not None:
        print(f"Ollama File:    {result.modelfile_path.resolve()}")


def _handle_models(args: argparse.Namespace) -> None:
    """
    Searches Hugging Face Hub for candidate SLMs within the parameter budget.

    Args:
        args (argparse.Namespace): Parsed CLI arguments from the `models` subcommand.
    """
    registry = SLMRegistry(max_params_b=args.max_params_b, hf_token=args.hf_token)
    page = registry.search_models(
        query=args.query,
        author=args.author,
        min_params_b=args.min_params_b,
        max_params_b=args.max_params_b,
        page=args.page,
        page_size=args.page_size,
    )
    print(f"\n=== SLMs ON HUB (Page {page.page}, has_more={page.has_more}) ===")
    for item in page.items:
        print(
            f"- {item.source:<38} | {item.estimated_params_b:>4.2f}B | "
            f"arch={item.model_type:<8} | downloads={item.downloads:>9,}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entry point dispatching `dataset`, `train`, `run`, `export`, and `models`.

    Args:
        argv (Sequence[str] | None, optional): Optional argument vector; defaults
            to `sys.argv[1:]` when None. Defaults to None.

    Returns:
        int: Process exit code (`0` on success, `130` on keyboard interrupt,
            `1` on unhandled error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    try:
        if args.command == "dataset":
            out_path = asyncio.run(_run_dataset_step(args))
            print(f"Dataset generated at: {out_path.resolve()}")
        elif args.command == "train":
            _handle_train_or_run(args, dataset_path=args.dataset)
        elif args.command == "run":
            dataset_path = asyncio.run(_run_dataset_step(args))
            _handle_train_or_run(args, dataset_path=dataset_path)
        elif args.command == "export":
            _handle_export(args)
        elif args.command == "models":
            _handle_models(args)

        return 0
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Execution interrupted by user.")
        return 130
    except Exception as exc:
        logging.getLogger(__name__).error("Execution failed: %s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
