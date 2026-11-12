"""
LoRA adapter merging and standalone SLM export script for Ollama deployment.

Demonstrates fusing a fine-tuned PEFT LoRA adapter back into its unquantized
base Small Language Model via `SLMAdapterExporter`, saving sharded SafeTensors
weights, and generating a ready-to-import Ollama `Modelfile`.
"""

from __future__ import annotations
from pathlib import Path

from ft.training import SLMAdapterExporter


def main() -> None:
    """
    Merges a trained LoRA adapter with its base SLM and generates an Ollama Modelfile.

    Instantiates `SLMAdapterExporter` in `bfloat16` precision, merges the trained
    adapter weights into the base model checkpoint, exports the standalone model
    artifacts, and prints the CLI command required to register the model in Ollama.
    """
    exporter = SLMAdapterExporter(
        dtype="bfloat16",
        device_map="auto",
    )

    result = exporter.merge_and_export(
        base_model_id_or_path=Path("models/gemma-2-2b-it"),
        adapter_dir=Path("outputs/gemma-2-2b-lora"),
        output_dir=Path("outputs/gemma-2-2b-merged"),
    )

    print("=== EXPORT COMPLETED ===")
    print(f"Merged model:     {result.merged_model_dir.resolve()}")
    print(f"Precision:        {result.dtype}")
    if result.modelfile_path is not None:
        print(f"Modelfile ready:  {result.modelfile_path.resolve()}")
        print("\nTo import directly into Ollama, run:")
        print(f'  ollama create my-slm-finetuned -f "{result.modelfile_path}"')


if __name__ == "__main__":
    main()