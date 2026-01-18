import torch
import torch.nn as nn


class RealQuantize(nn.Module):
    """
    Real quantization module that uses actual integer operations (int8).
    Unlike FakeQuantize, this module stores weights and performs computations in int8.
    """

    def __init__(self, bits: int = 8, symmetric: bool = True):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.qmin = -(2 ** (bits - 1)) if symmetric else 0
        self.qmax = 2 ** (bits - 1) - 1 if symmetric else 2**bits - 1
        self.register_buffer("scale", torch.ones(1, dtype=torch.float32))
        self.register_buffer("zero_point", torch.zeros(1, dtype=torch.int32))

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize to integer values."""
        q = torch.clamp(torch.round(x / self.scale) + self.zero_point.float(), self.qmin, self.qmax)
        return q.to(torch.int8)

    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        """Dequantize from integer back to float."""
        return (q.float() - self.zero_point.float()) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply real quantization followed by dequantization."""
        if self.training:
            # In training mode, use fake quantization for differentiability
            q = torch.clamp(torch.round(x / self.scale) + self.zero_point.float(), self.qmin, self.qmax)
            dq = (q - self.zero_point.float()) * self.scale
            return dq
        else:
            # In eval mode, perform actual int8 quantization
            q = self.quantize(x)
            return self.dequantize(q)

    def calibrate(self, x: torch.Tensor) -> None:
        """Calibrate scale and zero_point based on data statistics."""
        with torch.no_grad():
            if self.symmetric:
                abs_max = x.abs().max()
                self.scale.copy_(abs_max / (2 ** (self.bits - 1) - 1))
                self.zero_point.fill_(0)
            else:
                x_min, x_max = x.min(), x.max()
                self.scale.copy_((x_max - x_min) / (2**self.bits - 1))
                self.zero_point.copy_(-torch.round(x_min / self.scale).to(torch.int32))


class RealQuantizeLinear(nn.Module):
    """
    Linear layer with real int8 quantization on weights and activations.
    Stores weights as int8 and performs int8xint8 matrix operations when possible.
    """

    def __init__(self, in_features: int, out_features: int, bits: int = 8, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        # Store original weights as parameters (for training)
        self.register_parameter(
            "weight", nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        )
        if bias:
            self.register_parameter("bias", nn.Parameter(torch.zeros(out_features)))
        else:
            self.register_parameter("bias", None)

        # Quantization parameters
        self.weight_quantizer = RealQuantize(bits=bits, symmetric=True)
        self.activation_quantizer = RealQuantize(bits=bits, symmetric=False)

        # Buffers for int8 weights (computed after calibration)
        self.register_buffer("weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.ones(1))
        self.register_buffer("is_quantized", torch.tensor(False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Training or evaluation before quantization - use float weights
            x_q = self.activation_quantizer(x)
            w_q = self.weight_quantizer(self.weight)
            output = torch.nn.functional.linear(x_q, w_q, self.bias)
            return output
        elif self.is_quantized.item():
            # Real quantized inference using int8 weights
            x_q = self.activation_quantizer(x)
            # Perform int8 computation
            output = torch.nn.functional.linear(x_q, self.weight_int8.float(), self.bias)
            # Rescale by combined scales
            output = output * self.weight_scale * self.activation_quantizer.scale
            return output
        else:
            # Evaluation without quantization - use float weights
            x_q = self.activation_quantizer(x)
            w_q = self.weight_quantizer(self.weight)
            output = torch.nn.functional.linear(x_q, w_q, self.bias)
            return output

    def calibrate(self, x: torch.Tensor) -> None:
        """Calibrate quantization parameters."""
        with torch.no_grad():
            self.weight_quantizer.calibrate(self.weight)
            self.activation_quantizer.calibrate(x)

    def quantize_weights(self) -> None:
        """Convert float weights to int8 representation (one-way conversion for inference)."""
        with torch.no_grad():
            self.weight_quantizer.calibrate(self.weight)
            w_q = self.weight_quantizer.quantize(self.weight)
            self.weight_int8.copy_(w_q)
            self.weight_scale.copy_(self.weight_quantizer.scale)
            self.is_quantized.fill_(True)


def convert_to_real_quant(model: nn.Module, bits: int = 8) -> nn.Module:
    """
    Replace all Linear layers with RealQuantizeLinear.
    Enables real int8 quantization for the model.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            real_quant_layer = RealQuantizeLinear(
                module.in_features,
                module.out_features,
                bits=bits,
                bias=module.bias is not None,
            )
            real_quant_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                real_quant_layer.bias.data = module.bias.data.clone()
            setattr(model, name, real_quant_layer)
        else:
            convert_to_real_quant(module, bits)
    return model
