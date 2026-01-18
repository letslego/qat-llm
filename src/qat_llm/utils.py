import torch


def symmetric_quantize(tensor: torch.Tensor, bits: int = 8):
    """Apply symmetric uniform quantization."""
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    scale = tensor.abs().max() / qmax
    q_tensor = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    dq_tensor = q_tensor * scale
    return dq_tensor, scale


def asymmetric_quantize(tensor: torch.Tensor, bits: int = 8):
    """Apply asymmetric uniform quantization."""
    qmin = 0
    qmax = 2**bits - 1
    t_min, t_max = tensor.min(), tensor.max()
    scale = (t_max - t_min) / (qmax - qmin)
    zero_point = qmin - torch.round(t_min / scale)
    q_tensor = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax)
    dq_tensor = (q_tensor - zero_point) * scale
    return dq_tensor, scale, zero_point


def percentile_clip(tensor: torch.Tensor, percentile: float = 99.5):
    """Clip tensor values at specified percentile to handle outliers."""
    threshold = torch.quantile(tensor.abs(), percentile / 100.0)
    return torch.clamp(tensor, -threshold, threshold)


def estimate_quantization_error(original: torch.Tensor, quantized: torch.Tensor):
    """Compute mean squared error between original and quantized tensors."""
    mse = torch.mean((original - quantized) ** 2)
    return mse.item()
