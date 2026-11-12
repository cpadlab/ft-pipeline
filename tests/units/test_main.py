

from __future__ import annotations

import logging
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import ft.__main__ as ft_main
from ft.cli import build_parser
from ft.datasets.pipeline import DatasetPipelineResult
from ft.models.registry import SLMPage, SLMProfile
from ft.orchestrator import PipelineRunResult
from ft.training.exporter import ExportResult
from ft.training.trainer import TrainingResult


def test_configure_logging_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[int] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: captured.append(kwargs["level"]))

    ft_main._configure_logging(False)
    ft_main._configure_logging(True)

    assert captured == [logging.INFO, logging.DEBUG]


def test_run_dataset_step_wires_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected_out = tmp_path / "out.jsonl"
    captured: dict[str, object] = {}

    class FakeSynthesizer:
        def __init__(self, **kwargs: object) -> None:
            captured["synth_kwargs"] = kwargs

        async def __aenter__(self) -> FakeSynthesizer:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    class FakePipeline:
        def __init__(self, **kwargs: object) -> None:
            captured["pipeline_kwargs"] = kwargs

        async def run(self, **kwargs: object) -> DatasetPipelineResult:
            captured["run_kwargs"] = kwargs
            return DatasetPipelineResult(
                source_path=tmp_path,
                output_path=expected_out,
                chunks_extracted=2,
                records_synthesized=4,
            )

    monkeypatch.setattr(ft_main, "DatasetSynthesizer", FakeSynthesizer)
    monkeypatch.setattr(ft_main, "DatasetPipeline", FakePipeline)

    args = build_parser().parse_args(
        [
            "dataset",
            "--pdf-dir",
            str(tmp_path),
            "--dataset-out",
            str(expected_out),
            "--ai-api-key",
            "k-123",
            "--ai-base-url",
            "http://localhost:9000/v1",
            "--ai-model",
            "m1",
            "--ai-no-verify-ssl",
            "--max-concurrency",
            "3",
            "--pairs-per-chunk",
            "2",
            "--ai-temperature",
            "0.5",
            "--overwrite",
        ]
    )

    import asyncio

    result_path = asyncio.run(ft_main._run_dataset_step(args))

    assert result_path == expected_out
    assert captured["synth_kwargs"] == {
        "api_key": "k-123",
        "base_url": "http://localhost:9000/v1",
        "model": "m1",
        "verify_ssl": False,
        "max_concurrency": 3,
        "questions_per_chunk": 2,
        "temperature": 0.5,
    }
    assert captured["run_kwargs"] == {
        "input_path": tmp_path,
        "output_path": expected_out,
        "overwrite": True,
        "append": False,
    }


def test_build_pipeline_config(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--dataset",
            str(tmp_path / "data.jsonl"),
            "--model",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--workspace-dir",
            str(tmp_path / "ws"),
            "--target-modules",
            "q_proj",
            "v_proj",
            "--no-merge",
            "--no-ollama-modelfile",
            "--no-gradient-checkpointing",
        ]
    )
    cfg = ft_main._build_pipeline_config(args, dataset_path=tmp_path / "data.jsonl")

    assert cfg.model_id_or_path == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.lora.target_modules == ("q_proj", "v_proj")
    assert cfg.training is not None
    assert cfg.training.output_dir == tmp_path / "ws" / "adapter"
    assert cfg.use_gradient_checkpointing is False
    assert cfg.merge_and_export is False
    assert cfg.generate_ollama_modelfile is False


def test_handle_train_or_run_prints_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    profile = SLMProfile(source="m", model_type="qwen2", estimated_params_b=0.5)
    training_res = TrainingResult(
        adapter_dir=tmp_path / "adapter",
        train_samples=10,
        eval_samples=2,
        global_steps=5,
        train_loss=0.1234,
        metrics={},
    )
    export_res = ExportResult(
        merged_model_dir=tmp_path / "merged",
        modelfile_path=tmp_path / "merged" / "Modelfile",
        dtype="bfloat16",
    )
    fake_orch = MagicMock()
    fake_orch.run.return_value = PipelineRunResult(
        profile=profile,
        resolved_model_path="m",
        training=training_res,
        export=export_res,
    )
    monkeypatch.setattr(ft_main, "SLMPipelineOrchestrator", lambda hf_token=None: fake_orch)

    args = build_parser().parse_args(
        ["train", "--dataset", str(tmp_path / "d.jsonl"), "--model", "m"]
    )
    ft_main._handle_train_or_run(args, dataset_path=tmp_path / "d.jsonl")

    out = capsys.readouterr().out
    assert "PIPELINE EXECUTION SUMMARY" in out
    assert "Merged Model:" in out
    assert "Ollama File:" in out

    fake_orch.run.return_value = PipelineRunResult(
        profile=profile,
        resolved_model_path="m",
        training=training_res,
        export=ExportResult(
            merged_model_dir=tmp_path / "merged",
            modelfile_path=None,
            dtype="bfloat16",
        ),
    )
    ft_main._handle_train_or_run(args, dataset_path=tmp_path / "d.jsonl")
    out_no_modelfile = capsys.readouterr().out
    assert "Merged Model:" in out_no_modelfile
    assert "Ollama File:" not in out_no_modelfile

    fake_orch.run.return_value = PipelineRunResult(
        profile=profile,
        resolved_model_path="m",
        training=training_res,
        export=None,
    )
    ft_main._handle_train_or_run(args, dataset_path=tmp_path / "d.jsonl")
    assert "PIPELINE EXECUTION SUMMARY" in capsys.readouterr().out


def test_handle_export_and_models(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake_exporter = MagicMock()
    fake_exporter.merge_and_export.return_value = ExportResult(
        merged_model_dir=tmp_path / "merged",
        modelfile_path=tmp_path / "merged" / "Modelfile",
        dtype="float16",
    )
    monkeypatch.setattr(ft_main, "SLMAdapterExporter", lambda **kwargs: fake_exporter)

    export_args = build_parser().parse_args(
        [
            "export",
            "--model",
            "m",
            "--adapter-dir",
            str(tmp_path / "a"),
            "--output-dir",
            str(tmp_path / "merged"),
        ]
    )
    ft_main._handle_export(export_args)
    out = capsys.readouterr().out
    assert "EXPORT SUMMARY" in out
    assert "Ollama File:" in out

    fake_exporter.merge_and_export.return_value = ExportResult(
        merged_model_dir=tmp_path / "merged",
        modelfile_path=None,
        dtype="float16",
    )
    ft_main._handle_export(export_args)
    assert "Ollama File:" not in capsys.readouterr().out

    fake_registry = MagicMock()
    fake_registry.search_models.return_value = SLMPage(
        items=[SLMProfile(source="org/m", model_type="qwen2", estimated_params_b=0.5, downloads=100)],
        page=1,
        page_size=10,
        has_more=False,
    )
    monkeypatch.setattr(ft_main, "SLMRegistry", lambda **kwargs: fake_registry)

    models_args = build_parser().parse_args(["models", "--query", "qwen"])
    ft_main._handle_models(models_args)
    assert "SLMs ON HUB" in capsys.readouterr().out


def test_main_dispatch_and_error_handling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ft_main, "_configure_logging", lambda _: None)
    monkeypatch.setattr(ft_main, "_run_dataset_step", AsyncMock(return_value=tmp_path / "ds.jsonl"))
    monkeypatch.setattr(ft_main, "_handle_train_or_run", MagicMock())
    monkeypatch.setattr(ft_main, "_handle_export", MagicMock())
    monkeypatch.setattr(ft_main, "_handle_models", MagicMock())

    assert ft_main.main(["dataset", "--pdf-dir", str(tmp_path)]) == 0
    assert ft_main.main(["train", "--dataset", str(tmp_path / "ds.jsonl"), "--model", "m"]) == 0
    assert ft_main.main(["run", "--pdf-dir", str(tmp_path), "--model", "m"]) == 0
    assert (
        ft_main.main(
            [
                "export",
                "--model",
                "m",
                "--adapter-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert ft_main.main(["models"]) == 0

    monkeypatch.setattr(ft_main, "_handle_models", MagicMock(side_effect=KeyboardInterrupt))
    assert ft_main.main(["models"]) == 130

    monkeypatch.setattr(ft_main, "_handle_models", MagicMock(side_effect=RuntimeError("boom")))
    assert ft_main.main(["models"]) == 1

    fake_parser = MagicMock()
    fake_parser.parse_args.return_value = SimpleNamespace(command="unknown", verbose=False)
    monkeypatch.setattr(ft_main, "build_parser", lambda: fake_parser)
    assert ft_main.main([]) == 0


def test_main_module_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["ft", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("ft.__main__", run_name="__main__")
    assert exc_info.value.code == 0
