import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .calibration import calibrate_model, collect_calibration_data
from .fake_quant import apply_fake_quant_to_model


def parse_args():
    parser = argparse.ArgumentParser(description="QAT Training for LLMs")
    parser.add_argument("--model-id", required=True, help="HuggingFace model ID")
    parser.add_argument("--dataset", default="wikitext", help="Dataset name for training")
    parser.add_argument("--output-dir", type=Path, default=Path("./qat-output"), help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--quantization-mode", choices=["none", "ptq", "qat"], default="qat", help="Quantization mode")
    parser.add_argument("--bits", type=int, default=8, help="Quantization bits")
    parser.add_argument("--calibration-samples", type=int, default=50, help="Calibration dataset size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("TRANSFORMERS_NO_TORCH_LOAD_REQUIREMENT", "1")

    print(f"Loading model {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, device_map="auto")

    # Apply fake quantization if QAT mode
    if args.quantization_mode == "qat":
        print(f"Applying fake quantization with {args.bits} bits...")
        model = apply_fake_quant_to_model(model, bits=args.bits)

        # Calibration step
        print("Running calibration...")
        calib_data = collect_calibration_data(args.dataset, max_samples=args.calibration_samples)
        calibrate_model(model, tokenizer, calib_data, device=str(model.device))

    # Load dataset
    print(f"Loading training dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, "wikitext-2-raw-v1" if "wikitext" in args.dataset else None, split="train")
    dataset = dataset.shuffle(seed=42).select(range(min(1000, len(dataset))))  # Limit for demo

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,  # Disable for QAT to avoid double quantization
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
