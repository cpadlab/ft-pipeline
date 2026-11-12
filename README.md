# Fine Tuning Pipeline

An end-to-end Python pipeline and CLI for **Small Language Model (SLM)** adaptation—taking raw PDF documents through synthetic conversational dataset generation, QLoRA Supervised Fine-Tuning (SFT), and lossless LoRA merging for local deployment with Ollama.

---

## Overview

**Fine Tuning Pipeline (`ft`)** provides a modular workflow to specialize compact language models (default ceiling `<= 4.5B` parameters) on domain-specific technical documentation without manual dataset curation or brittle glue scripts.

The pipeline separates document ingestion, asynchronous OpenAI-compatible QA synthesis, parameter-bounded model loading, QLoRA training, and full-precision adapter merging behind a single Python API (`SLMPipelineOrchestrator`) and CLI (`ft` / `python -m ft`).

---

## Pipeline Modules & Capabilities

- **PDF Ingestion (`ft.datasets.ingest`)**: Layout-aware PDF parsing via PyMuPDF (`PDFIngestor`) using font-size heuristics for heading detection, diagram-label filtering, and broken-paragraph reconstruction.
- **Synthetic QA Generation (`ft.datasets.synthesizer`)**: Asynchronous structured-output synthesis (`DatasetSynthesizer`) over any OpenAI-compatible endpoint (`AI_BASE_URL`, `AI_MODEL`, `AI_API_KEY`) with bounded concurrency and Pydantic schema enforcement.
- **Validated Serialization (`ft.datasets.serializer`)**: Safe JSONL persistence (`JSONLSerializer`) with prompt-level deduplication, optional provenance metadata, and append/overwrite guards.
- **Dataset Orchestration (`ft.datasets.pipeline`)**: Coordinates ingestion, synthesis, and serialization behind `DatasetPipeline`.
- **Dynamic SLM Registry (`ft.models.registry`)**: Streams and filters Hugging Face Hub models (`SLMRegistry`) and validates parameter budgets (`max_params_b`) via Hub metadata or `AutoConfig` before downloading.
- **Quantized Model Loader (`ft.models.loader`)**: Configures `bitsandbytes` 4-bit NF4 / 8-bit / unquantized loading (`SLMModelLoader`), k-bit gradient preparation, and automatic LoRA linear target-module discovery.
- **QLoRA SFT & Export (`ft.training`)**: Pydantic-validated `LoRASettings` and `SFTTrainingSettings`, `trl.SFTTrainer` execution (`SLMTrainer`), and unquantized `merge_and_unload()` SafeTensors + Ollama `Modelfile` export (`SLMAdapterExporter`).
- **End-to-End Orchestrator & CLI (`ft.orchestrator`, `ft.cli`, `ft.__main__`)**: Unifies the full lifecycle via `SLMPipelineOrchestrator` and the `ft` CLI (`dataset`, `train`, `run`, `export`, `models`).

---

## Quickstart

Requires Python `>= 3.14` and [Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/cpadlab/ft-pipeline.git
cd ft-pipeline
poetry install
```

Configure your `.env` file for Phase 1 dataset synthesis (and optional Hugging Face Hub access for gated models):

```env
AI_BASE_URL=http://localhost:8000/v1
AI_API_KEY=your-api-key
AI_MODEL=your-synthesis-model
AI_VERIFY_SSL=true
AI_MAX_CONCURRENCY=4
HF_TOKEN=
```

### CLI Usage

```bash
# 1. Synthesize a conversational JSONL dataset from PDFs
poetry run ft dataset --pdf-dir upload --dataset-out data/dataset.jsonl --overwrite

# 2. Fine-tune a validated SLM with QLoRA and export merged weights + Ollama Modelfile
poetry run ft train --dataset data/dataset.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --workspace-dir outputs/run

# 3. Run the full end-to-end pipeline (PDF -> JSONL -> QLoRA -> Merge & Export)
poetry run ft run --pdf-dir upload --model Qwen/Qwen2.5-0.5B-Instruct --workspace-dir outputs/run

# 4. Merge an existing LoRA adapter separately
poetry run ft export --model models/base --adapter-dir outputs/run/adapter --output-dir outputs/run/merged

# 5. Discover candidate SLMs on Hugging Face Hub
poetry run ft models --query instruct --max-params-b 3.5
```

Standalone Python scripts demonstrating each stage are also available in `examples/`.

---

## Development

Run the unit test suite and coverage report with Poetry:

```bash
poetry run pytest
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
