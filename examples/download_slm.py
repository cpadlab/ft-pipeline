"""
One-time SLM validation and local snapshot download for offline workflows.

Demonstrates verifying a Hugging Face Hub model against the parameter-count
ceiling via `SLMRegistry` and downloading its SafeTensors weights to a local
directory for air-gapped or offline fine-tuning pipelines.
"""

from __future__ import annotations
from pathlib import Path

from ft.models import SLMRegistry


def main() -> None:
    """
    Validates and downloads a target SLM once to local storage for offline workflows.

    Initializes an `SLMRegistry` with a 4.5B parameter ceiling, verifies that
    the requested repository complies with the size limit, and downloads the
    model snapshot to a local folder.
    """
    model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
    destination_dir = Path("models/qwen2.5-coder-3b-instruct")

    registry = SLMRegistry(max_params_b=4.5)

    print(f"Verifying and starting download for '{model_id}'...")
    saved_path = registry.download_model(
        model_id=model_id,
        local_dir=destination_dir,
    )

    print(f"Download completed. Model weights saved at: {saved_path.resolve()}")


if __name__ == "__main__":
    main()