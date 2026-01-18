import torch
import pytest
from qat_llm.ptq import PTQPipeline
from qat_llm.real_quant import RealQuantizeLinear


class SimpleDataLoader:
    """Simple test data loader."""
    def __init__(self, num_batches=5, batch_size=10, input_size=10):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.input_size = input_size
    
    def __iter__(self):
        for _ in range(self.num_batches):
            yield torch.randn(self.batch_size, self.input_size)


def test_ptq_pipeline_creation():
    """Test PTQ pipeline initialization."""
    model = torch.nn.Linear(10, 5)
    pipeline = PTQPipeline(model, bits=8)
    
    assert pipeline.model == model
    assert pipeline.bits == 8
    assert pipeline.quantization_config["bits"] == 8


def test_ptq_calibrate():
    """Test PTQ calibration."""
    model = torch.nn.Linear(10, 5)
    pipeline = PTQPipeline(model, bits=8)
    
    data_loader = SimpleDataLoader(num_batches=3)
    pipeline.calibrate(data_loader, num_batches=2)
    
    # Should complete without error


def test_ptq_convert_to_real_quant():
    """Test conversion to real quantization."""
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5)
    )
    pipeline = PTQPipeline(model, bits=8)
    
    quantized_model = pipeline.convert_to_real_quant()
    
    # Check that it's now using RealQuantizeLinear
    has_real_quant = False
    for module in quantized_model.modules():
        if isinstance(module, RealQuantizeLinear):
            has_real_quant = True
            break
    
    # Since we have a Sequential with a Linear, should be converted
    assert has_real_quant or any(isinstance(m, RealQuantizeLinear) for m in quantized_model.modules())


def test_ptq_quantize_weights():
    """Test weight quantization."""
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.Linear(20, 5)
    )
    pipeline = PTQPipeline(model, bits=8)
    
    # Convert to real quant first
    quantized_model = pipeline.convert_to_real_quant()
    pipeline.model = quantized_model
    
    # Quantize weights
    pipeline.quantize_weights()
    
    # Check that weights are quantized
    for module in quantized_model.modules():
        if isinstance(module, RealQuantizeLinear):
            assert module.is_quantized.item() or not module.training


def test_ptq_get_model_size():
    """Test model size calculation."""
    model = torch.nn.Linear(10, 5)
    pipeline = PTQPipeline(model)
    
    original_size, _ = pipeline.get_model_size()
    
    # Should be approximately (10*5 + 5) * 4 bytes / 1024^2
    expected_size = (50 + 5) * 4 / (1024 * 1024)
    assert abs(original_size - expected_size) < 0.001


def test_ptq_benchmark_inference():
    """Test inference benchmarking."""
    model = torch.nn.Linear(10, 5)
    model.eval()
    pipeline = PTQPipeline(model)
    
    data_loader = SimpleDataLoader(num_batches=5)
    throughput, latency = pipeline.benchmark_inference(data_loader, num_batches=3)
    
    assert throughput > 0
    assert latency > 0


def test_ptq_evaluate_accuracy():
    """Test accuracy evaluation."""
    model = torch.nn.Linear(10, 5)
    model.eval()
    pipeline = PTQPipeline(model)
    
    data_loader = SimpleDataLoader(num_batches=3)
    
    def dummy_metric(pred, target):
        return 0.5
    
    # Need to wrap data in tuples for accuracy evaluation
    class TupleDataLoader:
        def __init__(self, num_batches=3):
            self.num_batches = num_batches
        
        def __iter__(self):
            for _ in range(self.num_batches):
                yield (torch.randn(10, 10), torch.randn(10, 5))
    
    tuple_loader = TupleDataLoader(num_batches=3)
    accuracy = pipeline.evaluate_accuracy(tuple_loader, dummy_metric)
    assert accuracy == 0.5


def test_ptq_pipeline_full_workflow():
    """Test complete PTQ workflow."""
    # Create and train a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    
    # Create pipeline
    pipeline = PTQPipeline(model, bits=8)
    
    # Calibrate
    calibration_loader = SimpleDataLoader(num_batches=2)
    pipeline.calibrate(calibration_loader)
    
    # Convert to real quantization
    quantized_model = pipeline.convert_to_real_quant()
    
    # Quantize weights
    pipeline.model = quantized_model
    pipeline.quantize_weights()
    
    # Get size
    original_size, quantized_size = pipeline.get_model_size()
    assert quantized_size > 0
    
    # Benchmark
    test_loader = SimpleDataLoader(num_batches=2)
    throughput, latency = pipeline.benchmark_inference(test_loader, num_batches=1)
    assert throughput > 0
    assert latency > 0
