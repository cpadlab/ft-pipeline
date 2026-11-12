"""
4-bit quantized SLM loading, LoRA target discovery, and chat template verification.

Demonstrates loading a local or Hub-hosted Small Language Model via
`SLMModelLoader` in 4-bit NF4 quantization, inspecting discovered linear
projection modules for LoRA adaptation, reporting GPU VRAM allocation, and
verifying conversational chat template formatting.
"""

from __future__ import annotations
from pathlib import Path
import torch

from ft.models import SLMModelLoader


def main() -> None:
    """
    Loads a local SLM in 4-bit quantization and verifies tokenizer and LoRA targets.

    Instantiates `SLMModelLoader` with 4-bit quantization and `bfloat16` compute
    precision, loads the target model prepared for k-bit training, logs memory
    consumption and introspected LoRA layers, and renders a sample conversational
    turn using the tokenizer's chat template.
    """
    model_source = Path("models/qwen2.5-coder-3b-instruct")

    loader = SLMModelLoader(
        quantization="none",
        compute_dtype="bfloat16",
        device_map="auto",
        use_gradient_checkpointing=True,
    )

    bundle = loader.load(model_source, prepare_for_training=True)

    print("=== LOADED MODEL SUMMARY ===")
    print(f"Source:             {bundle.profile.source}")
    print(f"Architecture:       {bundle.profile.model_type}")
    print(f"Estimated params:   ~{bundle.profile.estimated_params_b}B")
    print(f"Quantization:       {bundle.quantization}")
    print(f"LoRA layers (auto): {bundle.target_modules}")

    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        print(f"Allocated VRAM:     {allocated_gb:.2f} GB")

    sample_messages = [
        {"role": "user", "content": "What is a REST architecture?"},
        {
            "role": "assistant",
            "content": "It is an architectural style for distributed hypermedia systems based on HTTP.",
        },
    ]
    
    formatted_prompt = bundle.tokenizer.apply_chat_template(
        sample_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    print("\n=== APPLIED CHAT TEMPLATE SAMPLE ===")
    print(formatted_prompt)


if __name__ == "__main__":
    main()