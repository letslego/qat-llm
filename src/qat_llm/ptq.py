import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple
from qat_llm.real_quant import RealQuantizeLinear, convert_to_real_quant


class PTQPipeline:
    """
    Post-Training Quantization (PTQ) Pipeline.
    Quantizes a trained model without retraining using calibration data.
    """

    def __init__(self, model: nn.Module, bits: int = 8):
        self.model = model
        self.bits = bits
        self.quantization_config = {"bits": bits, "method": "ptq"}

    def calibrate(self, calibration_loader, num_batches: Optional[int] = None) -> None:
        """
        Calibrate quantization parameters using calibration data.
        
        Args:
            calibration_loader: DataLoader with calibration samples
            num_batches: Number of batches to use for calibration (None = all)
        """
        self.model.eval()
        
        with torch.no_grad():
            batch_count = 0
            for batch in calibration_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                # Forward pass to collect statistics
                _ = self.model(x)
                batch_count += 1
                
                if num_batches and batch_count >= num_batches:
                    break
        
        print(f"Calibration complete using {batch_count} batches")

    def convert_to_real_quant(self) -> nn.Module:
        """Convert model to use RealQuantizeLinear layers."""
        quantized_model = convert_to_real_quant(self.model, bits=self.bits)
        return quantized_model

    def quantize_weights(self) -> None:
        """Quantize all weights in the model to int8."""
        for module in self.model.modules():
            if isinstance(module, RealQuantizeLinear):
                module.quantize_weights()

    def get_model_size(self) -> Tuple[float, float]:
        """
        Calculate model sizes in MB (original float32 vs quantized int8).
        
        Returns:
            Tuple of (original_size_mb, quantized_size_mb)
        """
        # Calculate original size (assuming float32 = 4 bytes per param)
        total_params = sum(p.numel() for p in self.model.parameters())
        original_size_mb = (total_params * 4) / (1024 * 1024)
        
        # Calculate quantized size (int8 = 1 byte per weight, scales are float32)
        quantized_params = 0
        scale_params = 0
        for module in self.model.modules():
            if isinstance(module, RealQuantizeLinear):
                quantized_params += module.weight_int8.numel()  # int8
                scale_params += 1  # float32 scale per layer
        
        scale_size_mb = (scale_params * 4) / (1024 * 1024)
        quantized_size_mb = (quantized_params * 1 + scale_size_mb) / (1024 * 1024)
        
        return original_size_mb, quantized_size_mb

    def benchmark_inference(
        self, 
        test_loader, 
        device: str = "cpu",
        num_batches: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Benchmark inference speed (throughput).
        
        Args:
            test_loader: DataLoader for testing
            device: Device to run on
            num_batches: Number of batches to benchmark
            
        Returns:
            Tuple of (throughput_samples_per_sec, avg_latency_ms)
        """
        import time
        
        self.model.eval()
        self.model.to(device)
        
        total_samples = 0
        total_time = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                x = x.to(device)
                batch_size = x.shape[0]
                
                torch.cuda.synchronize() if device != "cpu" else None
                start = time.perf_counter()
                
                _ = self.model(x)
                
                torch.cuda.synchronize() if device != "cpu" else None
                end = time.perf_counter()
                
                batch_time = (end - start) * 1000  # Convert to ms
                total_time += batch_time
                total_samples += batch_size
                batch_count += 1
                
                if num_batches and batch_count >= num_batches:
                    break
        
        avg_latency_ms = total_time / batch_count
        throughput = total_samples / (total_time / 1000)  # samples per second
        
        return throughput, avg_latency_ms

    def evaluate_accuracy(
        self,
        test_loader,
        metric_fn: Callable,
        device: str = "cpu"
    ) -> float:
        """
        Evaluate model accuracy using provided metric function.
        
        Args:
            test_loader: DataLoader for testing
            metric_fn: Function that takes (predictions, targets) and returns score
            device: Device to run on
            
        Returns:
            Accuracy score from metric_fn
        """
        self.model.eval()
        self.model.to(device)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    x = batch
                    y = None
                
                x = x.to(device)
                predictions = self.model(x)
                all_predictions.append(predictions.cpu())
                
                if y is not None:
                    all_targets.append(y.cpu() if isinstance(y, torch.Tensor) else y)
        
        if all_targets:
            predictions = torch.cat(all_predictions, dim=0)
            targets = torch.cat(all_targets, dim=0) if isinstance(all_targets[0], torch.Tensor) else all_targets
            return metric_fn(predictions, targets)
        
        return 0.0
