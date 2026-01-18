import torch

from qat_llm.fake_quant import FakeQuantize, FakeQuantizeLinear


def test_fake_quantize_forward():
    fq = FakeQuantize(bits=8, symmetric=True)
    fq.train()
    x = torch.randn(10, 10)
    fq.calibrate(x)
    y = fq(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_fake_quantize_linear():
    layer = FakeQuantizeLinear(10, 5, bits=8)
    layer.train()
    x = torch.randn(2, 10)
    layer.calibrate(x)
    y = layer(x)
    assert y.shape == (2, 5)
