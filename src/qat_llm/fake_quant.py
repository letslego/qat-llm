import torch
import torch.nn as nn


class FakeQuantize(nn.Module):
    """
    Simulates quantization during training by quantizing and dequantizing tensors.
    Implements symmetric uniform quantization with learnable scale.
    """

    def __init__(self, bits: int = 8, symmetric: bool = True):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.qmin = -(2 ** (bits - 1)) if symmetric else 0
        self.qmax = 2 ** (bits - 1) - 1 if symmetric else 2**bits - 1
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zero_point", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        # Quantize
        x_q = torch.clamp(torch.round(x / self.scale) + self.zero_point, self.qmin, self.qmax)
        # Dequantize
        x_dq = (x_q - self.zero_point) * self.scale
        return x_dq

    def calibrate(self, x: torch.Tensor) -> None:
        """Update scale and zero_point based on observed tensor statistics."""
        with torch.no_grad():
            if self.symmetric:
                abs_max = x.abs().max()
                self.scale.copy_(abs_max / (2 ** (self.bits - 1) - 1))
                self.zero_point.fill_(0)
            else:
                x_min, x_max = x.min(), x.max()
                self.scale.copy_((x_max - x_min) / (2**self.bits - 1))
                self.zero_point.copy_(-torch.round(x_min / self.scale))


class FakeQuantizeLinear(nn.Module):
    """Linear layer with fake quantization on weights and activations."""

    def __init__(self, in_features: int, out_features: int, bits: int = 8, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight_quantizer = FakeQuantize(bits=bits, symmetric=True)
        self.activation_quantizer = FakeQuantize(bits=bits, symmetric=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize activations
        x_q = self.activation_quantizer(x)
        # Quantize weights
        w_q = self.weight_quantizer(self.linear.weight)
        # Forward pass with quantized weights
        output = nn.functional.linear(x_q, w_q, self.linear.bias)
        return output

    def calibrate(self, x: torch.Tensor) -> None:
        """Run calibration on weights and activations."""
        with torch.no_grad():
            self.weight_quantizer.calibrate(self.linear.weight)
            self.activation_quantizer.calibrate(x)


def apply_fake_quant_to_model(model: nn.Module, bits: int = 8) -> nn.Module:
    """
    Replace all Linear layers in a model with FakeQuantizeLinear.
    Returns the modified model.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            fake_quant_layer = FakeQuantizeLinear(
                module.in_features, module.out_features, bits=bits, bias=module.bias is not None
            )
            fake_quant_layer.linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                fake_quant_layer.linear.bias.data = module.bias.data.clone()
            setattr(model, name, fake_quant_layer)
        else:
            apply_fake_quant_to_model(module, bits)
    return model
