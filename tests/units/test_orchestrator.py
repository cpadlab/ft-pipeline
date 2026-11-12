
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from ft.models.registry import SLMProfile
from ft.orchestrator import PipelineConfig, SLMPipelineOrchestrator
from ft.training.exporter import ExportResult
from ft.training.schemas import SFTTrainingSettings
from ft.training.trainer import TrainingResult


def test_orchestrator_run_with_injected_components(tmp_path: Path) -> None:
    model_dir = tmp_path / "local_model"
    model_dir.mkdir()
    dataset_path = tmp_path / "data.jsonl"

    profile = SLMProfile(source=str(model_dir), model_type="qwen2", estimated_params_b=0.5)
    registry = MagicMock()
    registry.validate_model.return_value = profile

    loader = MagicMock()
    bundle = MagicMock()
    loader.load.return_value = bundle

    trainer = MagicMock()
    training_res = TrainingResult(
        adapter_dir=tmp_path / "ws" / "adapter",
        train_samples=10,
        eval_samples=2,
        global_steps=5,
        train_loss=0.2,
        metrics={},
    )
    trainer.train.return_value = training_res

    exporter = MagicMock()
    export_res = ExportResult(
        merged_model_dir=tmp_path / "ws" / "merged",
        modelfile_path=tmp_path / "ws" / "merged" / "Modelfile",
        dtype="bfloat16",
    )
    exporter.merge_and_export.return_value = export_res

    orch = SLMPipelineOrchestrator(
        hf_token="tok",
        registry=registry,
        loader=loader,
        trainer=trainer,
        exporter=exporter,
    )
    cfg = PipelineConfig(
        model_id_or_path=model_dir,
        dataset_path=dataset_path,
        workspace_dir=tmp_path / "ws",
    )
    result = orch.run(cfg, resume_from_checkpoint=tmp_path / "ckpt")

    assert result.profile == profile
    assert result.resolved_model_path == model_dir
    assert result.training == training_res
    assert result.export == export_res
    loader.load.assert_called_once_with(model_dir, prepare_for_training=True)
    trainer.train.assert_called_once_with(
        bundle=bundle,
        dataset_path=dataset_path,
        resume_from_checkpoint=tmp_path / "ckpt",
    )
    exporter.merge_and_export.assert_called_once()


def test_orchestrator_run_instantiates_defaults_and_skips_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = SLMProfile(source="org/slm", model_type="llama", estimated_params_b=1.0)
    mock_registry = MagicMock()
    mock_registry.validate_model.return_value = profile
    mock_registry.download_model.return_value = tmp_path / "cached"

    mock_loader = MagicMock()
    mock_loader.load.return_value = MagicMock()

    mock_trainer = MagicMock()
    mock_trainer.train.return_value = TrainingResult(
        adapter_dir=tmp_path / "adapter",
        train_samples=5,
        eval_samples=0,
        global_steps=2,
        train_loss=0.5,
        metrics={},
    )

    monkeypatch.setattr("ft.orchestrator.SLMRegistry", lambda **kwargs: mock_registry)
    monkeypatch.setattr("ft.orchestrator.SLMModelLoader", lambda **kwargs: mock_loader)
    monkeypatch.setattr("ft.orchestrator.SLMTrainer", lambda **kwargs: mock_trainer)

    cfg = PipelineConfig(
        model_id_or_path="org/slm",
        dataset_path=tmp_path / "d.jsonl",
        workspace_dir=tmp_path / "ws",
        cache_model_dir=tmp_path / "cache",
        merge_and_export=False,
    )
    result = SLMPipelineOrchestrator().run(cfg)

    assert result.resolved_model_path == tmp_path / "cached"
    assert result.export is None


def test_orchestrator_run_creates_default_exporter_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mock_registry = MagicMock()
    mock_registry.validate_model.return_value = SLMProfile(
        source="org/slm", model_type="llama", estimated_params_b=1.0
    )
    mock_loader = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = TrainingResult(
        adapter_dir=tmp_path / "adapter",
        train_samples=5,
        eval_samples=0,
        global_steps=2,
        train_loss=0.5,
        metrics={},
    )
    mock_exporter = MagicMock()
    mock_exporter.merge_and_export.return_value = ExportResult(
        merged_model_dir=tmp_path / "ws" / "merged",
        modelfile_path=None,
        dtype="bfloat16",
    )

    monkeypatch.setattr("ft.orchestrator.SLMAdapterExporter", lambda **kwargs: mock_exporter)

    orch = SLMPipelineOrchestrator(
        registry=mock_registry,
        loader=mock_loader,
        trainer=mock_trainer,
    )
    cfg = PipelineConfig(
        model_id_or_path="org/slm",
        dataset_path=tmp_path / "d.jsonl",
        workspace_dir=tmp_path / "ws",
        merge_and_export=True,
    )
    result = orch.run(cfg)

    assert result.resolved_model_path == "org/slm"
    assert result.export is not None


def test_resolve_training_settings_variants(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    cfg_none = PipelineConfig(model_id_or_path="m", dataset_path=tmp_path / "d.jsonl", workspace_dir=ws)
    assert SLMPipelineOrchestrator._resolve_training_settings(cfg_none).output_dir == ws / "adapter"

    cfg_default_out = PipelineConfig(
        model_id_or_path="m",
        dataset_path=tmp_path / "d.jsonl",
        workspace_dir=ws,
        training=SFTTrainingSettings(),
    )
    assert SLMPipelineOrchestrator._resolve_training_settings(cfg_default_out).output_dir == ws / "adapter"

    custom_dir = tmp_path / "custom_adapter"
    cfg_custom = PipelineConfig(
        model_id_or_path="m",
        dataset_path=tmp_path / "d.jsonl",
        workspace_dir=ws,
        training=SFTTrainingSettings(output_dir=custom_dir),
    )
    assert SLMPipelineOrchestrator._resolve_training_settings(cfg_custom).output_dir == custom_dir


def test_release_training_bundle_clears_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_cache = MagicMock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    SLMPipelineOrchestrator._release_training_bundle(object())
    empty_cache.assert_called_once()
