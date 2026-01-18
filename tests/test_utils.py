from qat_llm.utils import asymmetric_quantize, percentile_clip, symmetric_quantize
import torch


def test_symmetric_quantize():
    x = torch.randn(10, 10)
    q, scale = symmetric_quantize(x, bits=8)
    assert q.shape == x.shape
    assert scale > 0


def test_asymmetric_quantize():
    x = torch.randn(10, 10)
    q, scale, zp = asymmetric_quantize(x, bits=8)
    assert q.shape == x.shape
    assert scale > 0


def test_percentile_clip():
    x = torch.randn(100)
    clipped = percentile_clip(x, percentile=95.0)
    assert clipped.shape == x.shape
