import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_perplexity(model, tokenizer, dataset_name: str, max_samples: int = 50):
    """Compute perplexity on a test dataset."""
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1" if "wikitext" in dataset_name else None, split="test")
    dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))

    model.eval()
    nlls = []
    for sample in dataset:
        text = sample.get("text", "")
        if not text.strip():
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        if inputs["input_ids"].numel() == 0:
            continue
        with torch.inference_mode():
            outputs = model(**inputs, labels=inputs["input_ids"])
            nlls.append(outputs.loss.float().item())

    import math

    if not nlls:
        return float("nan")
    mean_nll = sum(nlls) / len(nlls)
    return math.exp(mean_nll)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PTQ vs QAT models")
    parser.add_argument("--model-id", required=True, help="Base model ID")
    parser.add_argument("--dataset", default="wikitext", help="Evaluation dataset")
    parser.add_argument("--ptq-checkpoint", type=Path, help="PTQ checkpoint directory")
    parser.add_argument("--qat-checkpoint", type=Path, help="QAT checkpoint directory")
    parser.add_argument("--metrics", nargs="+", default=["perplexity"], help="Metrics to compare")
    parser.add_argument("--output", type=Path, default=Path("comparison_results.json"), help="Output file")
    return parser.parse_args()


def main():
    args = parse_args()
    results = {}

    print(f"Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Evaluate baseline
    print("Evaluating baseline model...")
    baseline_model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")
    baseline_ppl = compute_perplexity(baseline_model, tokenizer, args.dataset)
    results["baseline"] = {"perplexity": baseline_ppl}
    print(f"Baseline perplexity: {baseline_ppl:.2f}")
    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Evaluate PTQ
    if args.ptq_checkpoint and args.ptq_checkpoint.exists():
        print(f"Evaluating PTQ model from {args.ptq_checkpoint}...")
        ptq_model = AutoModelForCausalLM.from_pretrained(str(args.ptq_checkpoint), device_map="auto")
        ptq_ppl = compute_perplexity(ptq_model, tokenizer, args.dataset)
        results["ptq"] = {"perplexity": ptq_ppl}
        print(f"PTQ perplexity: {ptq_ppl:.2f}")
        del ptq_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Evaluate QAT
    if args.qat_checkpoint and args.qat_checkpoint.exists():
        print(f"Evaluating QAT model from {args.qat_checkpoint}...")
        qat_model = AutoModelForCausalLM.from_pretrained(str(args.qat_checkpoint), device_map="auto")
        qat_ppl = compute_perplexity(qat_model, tokenizer, args.dataset)
        results["qat"] = {"perplexity": qat_ppl}
        print(f"QAT perplexity: {qat_ppl:.2f}")
        del qat_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Comparison results saved to {args.output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
