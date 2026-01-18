# QAT-LLM: Quantization-Aware Training for LLMs

Train LLMs with simulated quantization to minimize accuracy loss during deployment. Compare Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT).

## What it does
- Implements fake quantization nodes during training to simulate INT8 inference.
- Weight and activation clipping strategies for better quantization ranges.
- Calibration dataset support for optimal quantization parameters.
- Side-by-side PTQ vs QAT accuracy comparison on benchmark tasks.
- PEFT-compatible (LoRA/QLoRA integration).

## Why QAT matters
- PTQ is fast but can lose accuracy; QAT trains the model to be robust under quantization.
- Rare for LLMs due to compute cost, but yields superior results for deployment.
- Learn quantization-aware fine-tuning strategies that stand out in production ML.

## Setup
```bash
cd ~/projects/qat-llm
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Notes:
- `numpy<2` pinned for macOS torch compatibility.
- bitsandbytes skipped on macOS; use Linux/Windows CUDA for INT8 training.
- Requires GPU for realistic QAT experiments.

## How to run QAT training
```bash
qat-train \
  --model-id openai-community/gpt2 \
  --dataset wikitext \
  --output-dir ./qat-output \
  --num-epochs 1 \
  --quantization-mode qat \
  --bits 8
```

## How to compare PTQ vs QAT
```bash
qat-compare \
  --model-id openai-community/gpt2 \
  --dataset wikitext \
  --ptq-checkpoint ./ptq-output \
  --qat-checkpoint ./qat-output \
  --metrics perplexity accuracy
```

## How to test
```bash
cd ~/projects/qat-llm
source .venv/bin/activate
pytest -q
```

## Repository layout
- `src/qat_llm/fake_quant.py` — Fake quantization ops for weights/activations.
- `src/qat_llm/trainer.py` — QAT-aware training loop with calibration.
- `src/qat_llm/calibration.py` — Calibration dataset management and stats.
- `src/qat_llm/cli.py` — Training CLI entry point.
- `src/qat_llm/compare.py` — PTQ vs QAT comparison harness.
- `src/qat_llm/utils.py` — Clipping strategies and quantization helpers.
- `tests/` — Unit tests and fixtures.

## Features
- **Fake Quantization**: Simulate INT8 quantization during forward pass with learnable scale/zero-point.
- **Weight Clipping**: Min-max or percentile-based clipping for better dynamic range.
- **Activation Clipping**: Running stats for activation quantization ranges.
- **Calibration**: Use calibration dataset to initialize quantization parameters before QAT.
- **PTQ Baseline**: Train normally, then apply PTQ for comparison.
- **QAT Pipeline**: Train with fake quant from scratch or after calibration.

## Roadmap
- Add mixed-precision QAT (W8A16, W4A16).
- Support for grouped quantization (channel-wise, layer-wise).
- Integration with TensorRT/ONNX export for deployment.
- Benchmarks on standard LLM tasks (GLUE, SuperGLUE).
