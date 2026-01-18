import torch
import pytest
from qat_llm.real_quant import RealQuantize, RealQuantizeLinear, convert_to_real_quant


def test_real_quantize_forward():
    """Test RealQuantize forward pass."""
    rq = RealQuantize(bits=8, symmetric=True)
    rq.train()
    x = torch.randn(10, 10)
    rq.calibrate(x)
    y = rq(x)
    
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_real_quantize_quantize_method():
    """Test explicit quantize method returns int8."""
    rq = RealQuantize(bits=8, symmetric=True)
    x = torch.randn(10, 10)
    rq.calibrate(x)
    
    q = rq.quantize(x)
    assert q.dtype == torch.int8
    assert q.shape == x.shape


def test_real_quantize_dequantize():
    """Test dequantize method."""
    rq = RealQuantize(bits=8, symmetric=True)
    x = torch.randn(10, 10)
    rq.calibrate(x)
    
    q = rq.quantize(x)
    dq = rq.dequantize(q)
    
    assert dq.dtype == torch.float32
    assert dq.shape == x.shape


def test_real_quantize_linear():
    """Test RealQuantizeLinear layer."""
    layer = RealQuantizeLinear(10, 5, bits=8)
    layer.train()
    x = torch.randn(2, 10)
    layer.calibrate(x)
    y = layer(x)
    
    assert y.shape == (2, 5)
    assert y.dtype == torch.float32


def test_real_quantize_linear_weight_quantization():
    """Test weight quantization in RealQuantizeLinear."""
    layer = RealQuantizeLinear(10, 5, bits=8)
    layer.eval()
    x = torch.randn(2, 10)
    
    # Initially not quantized
    assert not layer.is_quantized.item()
    
    # Quantize weights
    layer.quantize_weights()
    assert layer.is_quantized.item()
    assert layer.weight_int8.dtype == torch.int8


def test_convert_to_real_quant():
    """Test converting model to real quantization."""
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    
    quantized_model = convert_to_real_quant(model, bits=8)
    
    # Check that Linear layers were replaced
    for module in quantized_model.modules():
        if isinstance(module, torch.nn.Linear):
            pytest.fail("Linear layer was not replaced")
    
    # Test forward pass
    x = torch.randn(2, 10)
    y = quantized_model(x)
    assert y.shape == (2, 5)


def test_real_quantize_asymmetric():
    """Test asymmetric quantization."""
    rq = RealQuantize(bits=8, symmetric=False)
    x = torch.randn(10, 10) + 10  # Shift to positive range
    rq.calibrate(x)
    
    y = rq(x)
    assert y.shape == x.shape
    assert rq.zero_point.item() != 0


def test_real_quantize_different_bitwidths():
    """Test quantization with different bit widths."""
    for bits in [4, 8, 16]:
        rq = RealQuantize(bits=bits, symmetric=True)
        x = torch.randn(10, 10)
        rq.calibrate(x)
        y = rq(x)
        
        assert y.shape == x.shape
        
        # Check quantization range
        q = rq.quantize(x)
        assert q.min().item() >= -(2 ** (bits - 1))
        assert q.max().item() <= 2 ** (bits - 1) - 1
