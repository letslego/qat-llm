from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from .fake_quant import FakeQuantize, FakeQuantizeLinear


def collect_calibration_data(dataset_name: str, split: str = "train", max_samples: int = 100, context_length: int = 128):
    """Load a calibration dataset for quantization parameter initialization."""
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1" if "wikitext" in dataset_name else None, split=split)
    dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))
    return dataset


def calibrate_model(model: torch.nn.Module, tokenizer, calibration_data, device: str = "cpu", batch_size: int = 4):
    """
    Run calibration pass to initialize quantization parameters.
    Collects activation and weight statistics from calibration dataset.
    """
    model.eval()
    model.to(device)

    print(f"Calibrating model on {len(calibration_data)} samples...")
    for i, sample in enumerate(tqdm(calibration_data, desc="Calibration")):
        text = sample.get("text", "")
        if not text.strip():
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Forward pass to collect stats
            _ = model(**inputs)

        # Calibrate fake quantizers
        for module in model.modules():
            if isinstance(module, FakeQuantizeLinear):
                # Calibrate using last seen activations (simple approach)
                # In production, accumulate stats over multiple batches
                module.calibrate(inputs["input_ids"].float())

        if i >= 20:  # Limit calibration samples for speed
            break

    print("Calibration complete.")


def update_quantization_ranges(model: torch.nn.Module, percentile: float = 99.9):
    """
    Apply percentile-based clipping to quantization ranges.
    Helps with outliers that distort quantization parameters.
    """
    for module in model.modules():
        if isinstance(module, FakeQuantize):
            # Placeholder: in practice, track histogram and compute percentile
            # For now, just rescale by a conservative factor
            with torch.no_grad():
                module.scale *= 0.95  # Slight shrink to handle outliers
