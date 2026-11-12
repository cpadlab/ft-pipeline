"""
End-to-end QLoRA Supervised Fine-Tuning (SFT) execution script.

Demonstrates loading a 4-bit quantized Small Language Model via
`SLMModelLoader`, configuring LoRA adapter and SFT training hyperparameters,
executing fine-tuning on a conversational JSONL dataset via `SLMTrainer`, and
persisting the resulting PEFT adapter weights to disk.
"""

from __future__ import annotations
from pathlib import Path

from ft.models import SLMModelLoader
from ft.training import LoRASettings, SFTTrainingSettings, SLMTrainer


def main() -> None:
    """
    Runs an end-to-end QLoRA fine-tuning session on a local SLM and JSONL dataset.

    Loads a local SLM in 4-bit NF4 quantization with gradient checkpointing,
    initializes LoRA rank/alpha settings and SFT optimization hyperparameters,
    trains the adapter over a conversational dataset, and logs summary metrics.
    """
    model_path = Path("models/qwen2.5-coder-3b-instruct")
    dataset_path = Path("data/processed/tema_1.jsonl")
    output_adapter_dir = Path("outputs/qwen2.5-coder-3b-instruct")

    loader = SLMModelLoader(
        quantization="none",
        compute_dtype="bfloat16",
        device_map="auto",
        use_gradient_checkpointing=True,
    )
    bundle = loader.load(model_path, prepare_for_training=True)

    lora_cfg = LoRASettings(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    train_cfg = SFTTrainingSettings(
        output_dir=output_adapter_dir,
        num_train_epochs=3.0,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_seq_length=2048,
        eval_split_ratio=0.1,
    )

    trainer = SLMTrainer(
        training_settings=train_cfg,
        lora_settings=lora_cfg,
    )
    result = trainer.train(bundle=bundle, dataset_path=dataset_path)

    print("=== TRAINING COMPLETED ===")
    print(f"Adapter saved at:      {result.adapter_dir.resolve()}")
    print(f"Train/eval samples:    {result.train_samples} / {result.eval_samples}")
    print(f"Total global steps:    {result.global_steps}")
    print(f"Final training loss:   {result.train_loss:.4f}")


if __name__ == "__main__":
    main()