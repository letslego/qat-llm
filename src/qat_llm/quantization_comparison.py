import torch
import torch.nn as nn
import time
from typing import Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
import json


@dataclass
class QuantizationMetrics:
    """Container for quantization comparison metrics."""
    method: str
    model_size_mb: float
    latency_ms: float
    throughput_samples_per_sec: float
    peak_memory_mb: float
    accuracy: Optional[float] = None
    quantization_error: Optional[float] = None
    

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QuantizationComparison:
    """
    Compare fake quantization vs real quantization in terms of:
    - Model size
    - Inference latency
    - Memory usage
    - Accuracy
    - Quantization error
    """

    def __init__(self, original_model: nn.Module):
        self.original_model = original_model
        self.fake_quant_model = None
        self.real_quant_model = None
        self.metrics = {}

    def setup_models(self, fake_quant_model: nn.Module, real_quant_model: nn.Module) -> None:
        """Set up models for comparison."""
        self.fake_quant_model = fake_quant_model.eval()
        self.real_quant_model = real_quant_model.eval()

    def _get_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        total_params = sum(p.numel() for p in model.parameters())
        # Assume float32 = 4 bytes per parameter
        size_mb = (total_params * 4) / (1024 * 1024)
        return size_mb

    def _benchmark_latency(
        self,
        model: nn.Module,
        input_shape: tuple = (10, 10),
        num_iterations: int = 100,
        device: str = "cpu",
        warmup: int = 10,
    ) -> Tuple[float, float]:
        """
        Benchmark model latency and throughput.
        
        Returns:
            Tuple of (latency_ms, throughput_samples_per_sec)
        """
        model = model.to(device)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(torch.randn(input_shape, device=device))
        
        torch.cuda.synchronize() if device != "cpu" else None
        start = time.perf_counter()
        
        # Benchmark
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(torch.randn(input_shape, device=device))
        
        torch.cuda.synchronize() if device != "cpu" else None
        end = time.perf_counter()
        
        total_time = (end - start) * 1000  # Convert to ms
        latency_ms = total_time / num_iterations
        batch_size = input_shape[0]
        throughput = (batch_size * num_iterations) / (total_time / 1000)
        
        return latency_ms, throughput

    def _measure_memory(
        self,
        model: nn.Module,
        input_shape: tuple = (10, 10),
        device: str = "cpu",
    ) -> float:
        """Measure peak memory usage in MB."""
        if device == "cpu":
            return 0.0  # Memory measurement not reliable on CPU
        
        model = model.to(device)
        model.eval()
        
        torch.cuda.reset_peak_memory_stats(device)
        
        with torch.no_grad():
            _ = model(torch.randn(input_shape, device=device))
        
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        return peak_memory

    def _compute_quantization_error(
        self,
        original_outputs: torch.Tensor,
        quantized_outputs: torch.Tensor,
    ) -> float:
        """Compute MSE between original and quantized outputs."""
        mse = torch.mean((original_outputs - quantized_outputs) ** 2).item()
        return mse

    def compare(
        self,
        test_loader,
        input_shape: tuple = (10, 10),
        device: str = "cpu",
        eval_metric_fn: Optional[Callable] = None,
        num_eval_batches: Optional[int] = None,
    ) -> Dict[str, QuantizationMetrics]:
        """
        Run comprehensive comparison of quantization methods.
        
        Args:
            test_loader: DataLoader for evaluation
            input_shape: Shape of input tensors for benchmarking
            device: Device to run benchmarks on
            eval_metric_fn: Optional accuracy metric function
            num_eval_batches: Number of batches for evaluation
            
        Returns:
            Dictionary mapping method names to QuantizationMetrics
        """
        results = {}
        
        # Original model metrics
        print("\n📊 Benchmarking Original Model...")
        original_size = self._get_model_size(self.original_model)
        original_latency, original_throughput = self._benchmark_latency(
            self.original_model, input_shape, device=device
        )
        original_memory = self._measure_memory(self.original_model, input_shape, device=device)
        
        results["original"] = QuantizationMetrics(
            method="original_float32",
            model_size_mb=original_size,
            latency_ms=original_latency,
            throughput_samples_per_sec=original_throughput,
            peak_memory_mb=original_memory,
        )
        
        # Fake quantization metrics
        if self.fake_quant_model is not None:
            print("📊 Benchmarking Fake Quantization...")
            fake_size = self._get_model_size(self.fake_quant_model)
            fake_latency, fake_throughput = self._benchmark_latency(
                self.fake_quant_model, input_shape, device=device
            )
            fake_memory = self._measure_memory(self.fake_quant_model, input_shape, device=device)
            
            # Compute quantization error
            fake_error = self._compute_quantization_error_on_loader(
                self.original_model, self.fake_quant_model, test_loader,
                device=device, num_batches=num_eval_batches
            )
            
            # Compute accuracy if metric provided
            fake_accuracy = None
            if eval_metric_fn:
                fake_accuracy = self._evaluate_accuracy(
                    self.fake_quant_model, test_loader, eval_metric_fn, 
                    device=device, num_batches=num_eval_batches
                )
            
            results["fake_quant"] = QuantizationMetrics(
                method="fake_quantization",
                model_size_mb=fake_size,
                latency_ms=fake_latency,
                throughput_samples_per_sec=fake_throughput,
                peak_memory_mb=fake_memory,
                accuracy=fake_accuracy,
                quantization_error=fake_error,
            )
        
        # Real quantization metrics
        if self.real_quant_model is not None:
            print("📊 Benchmarking Real Quantization...")
            real_size = self._get_model_size(self.real_quant_model)
            real_latency, real_throughput = self._benchmark_latency(
                self.real_quant_model, input_shape, device=device
            )
            real_memory = self._measure_memory(self.real_quant_model, input_shape, device=device)
            
            # Compute quantization error
            real_error = self._compute_quantization_error_on_loader(
                self.original_model, self.real_quant_model, test_loader,
                device=device, num_batches=num_eval_batches
            )
            
            # Compute accuracy if metric provided
            real_accuracy = None
            if eval_metric_fn:
                real_accuracy = self._evaluate_accuracy(
                    self.real_quant_model, test_loader, eval_metric_fn,
                    device=device, num_batches=num_eval_batches
                )
            
            results["real_quant"] = QuantizationMetrics(
                method="real_int8_quantization",
                model_size_mb=real_size,
                latency_ms=real_latency,
                throughput_samples_per_sec=real_throughput,
                peak_memory_mb=real_memory,
                accuracy=real_accuracy,
                quantization_error=real_error,
            )
        
        self.metrics = results
        return results

    def _compute_quantization_error_on_loader(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_loader,
        device: str = "cpu",
        num_batches: Optional[int] = None,
    ) -> float:
        """Compute average quantization error across batches."""
        original_model.eval()
        quantized_model.eval()
        original_model.to(device)
        quantized_model.to(device)
        
        total_error = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                x = x.to(device)
                
                original_output = original_model(x)
                quantized_output = quantized_model(x)
                
                error = self._compute_quantization_error(original_output, quantized_output)
                total_error += error
                batch_count += 1
                
                if num_batches and batch_count >= num_batches:
                    break
        
        return total_error / batch_count if batch_count > 0 else 0.0

    def _evaluate_accuracy(
        self,
        model: nn.Module,
        test_loader,
        metric_fn: Callable,
        device: str = "cpu",
        num_batches: Optional[int] = None,
    ) -> float:
        """Evaluate model accuracy."""
        model.eval()
        model.to(device)
        
        all_predictions = []
        all_targets = []
        batch_count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    x = batch
                    y = None
                
                x = x.to(device)
                predictions = model(x)
                all_predictions.append(predictions.cpu())
                
                if y is not None:
                    all_targets.append(y.cpu() if isinstance(y, torch.Tensor) else y)
                
                batch_count += 1
                if num_batches and batch_count >= num_batches:
                    break
        
        if all_targets:
            predictions = torch.cat(all_predictions, dim=0)
            targets = torch.cat(all_targets, dim=0) if isinstance(all_targets[0], torch.Tensor) else all_targets
            return metric_fn(predictions, targets)
        
        return 0.0

    def print_comparison(self) -> None:
        """Pretty print comparison results."""
        if not self.metrics:
            print("No metrics available. Run compare() first.")
            return
        
        print("\n" + "=" * 100)
        print("QUANTIZATION COMPARISON RESULTS".center(100))
        print("=" * 100)
        
        # Print table header
        print(f"{'Method':<25} {'Size (MB)':<15} {'Latency (ms)':<15} {'Throughput (s/s)':<18} {'Memory (MB)':<12}")
        print("-" * 100)
        
        # Print metrics for each method
        for method_name, metrics in self.metrics.items():
            print(
                f"{metrics.method:<25} {metrics.model_size_mb:>13.2f} {metrics.latency_ms:>13.4f} "
                f"{metrics.throughput_samples_per_sec:>16.0f} {metrics.peak_memory_mb:>10.2f}"
            )
        
        # Print detailed metrics
        print("\n" + "-" * 100)
        print("DETAILED METRICS")
        print("-" * 100)
        
        for method_name, metrics in self.metrics.items():
            print(f"\n{metrics.method}:")
            print(f"  Model Size:              {metrics.model_size_mb:.2f} MB")
            print(f"  Latency:                 {metrics.latency_ms:.4f} ms")
            print(f"  Throughput:              {metrics.throughput_samples_per_sec:.0f} samples/sec")
            print(f"  Peak Memory:             {metrics.peak_memory_mb:.2f} MB")
            if metrics.accuracy is not None:
                print(f"  Accuracy:                {metrics.accuracy:.4f}")
            if metrics.quantization_error is not None:
                print(f"  Quantization Error (MSE):{metrics.quantization_error:.6f}")
        
        # Print speedup/compression ratios
        print("\n" + "-" * 100)
        print("COMPRESSION & SPEEDUP RATIOS (vs Original)")
        print("-" * 100)
        
        if "original" in self.metrics:
            original = self.metrics["original"]
            
            for method_name in ["fake_quant", "real_quant"]:
                if method_name in self.metrics:
                    method = self.metrics[method_name]
                    size_compression = original.model_size_mb / method.model_size_mb
                    latency_speedup = original.latency_ms / method.latency_ms
                    throughput_speedup = method.throughput_samples_per_sec / original.throughput_samples_per_sec
                    
                    print(f"\n{method.method}:")
                    print(f"  Model Compression:       {size_compression:.2f}x")
                    print(f"  Latency Speedup:         {latency_speedup:.2f}x")
                    print(f"  Throughput Speedup:      {throughput_speedup:.2f}x")

    def save_comparison_report(self, output_path: str) -> None:
        """Save comparison results to JSON file."""
        report = {
            method_name: metrics.to_dict()
            for method_name, metrics in self.metrics.items()
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Comparison report saved to {output_path}")
