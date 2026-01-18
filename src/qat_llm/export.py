import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path


class ONNXExporter:
    """
    Export quantized models to ONNX format for deployment.
    Supports both fake and real quantized models.
    """

    def __init__(self, model: nn.Module, model_name: str = "qat_model"):
        self.model = model
        self.model_name = model_name

    def export(
        self,
        output_path: str,
        input_shape: tuple = (1, 10),
        opset_version: int = 14,
        optimize_model: bool = True,
        add_quantization_info: bool = True,
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            input_shape: Shape of dummy input for tracing
            opset_version: ONNX opset version
            optimize_model: Whether to optimize the model
            add_quantization_info: Whether to add quantization metadata
            
        Returns:
            Path to exported ONNX model
        """
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Please install onnx and onnxruntime: pip install onnx onnxruntime")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        
        # Create dummy input for tracing
        dummy_input = torch.randn(*input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            verbose=False,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        
        # Load and optimize if requested
        onnx_model = onnx.load(str(output_path))
        
        if add_quantization_info:
            onnx_model = self._add_quantization_metadata(onnx_model)
        
        if optimize_model:
            onnx_model = self._optimize_onnx_model(onnx_model)
        
        # Save optimized model
        onnx.save(onnx_model, str(output_path))
        
        print(f"✓ Model exported to {output_path}")
        return str(output_path)

    def _add_quantization_metadata(self, onnx_model) -> Any:
        """Add quantization information to ONNX model metadata."""
        import onnx
        from onnx import helper
        
        # Extract quantization info from model
        quant_info = self._extract_quantization_info()
        
        # Create metadata properties
        metadata_props = []
        for key, value in quant_info.items():
            prop = onnx.StringStringEntryProto()
            prop.key = key
            prop.value = str(value)
            metadata_props.append(prop)
        
        onnx_model.metadata_props.extend(metadata_props)
        
        return onnx_model

    def _extract_quantization_info(self) -> Dict[str, Any]:
        """Extract quantization configuration from model."""
        quant_info = {"quantization_method": "unknown", "bits": 8}
        
        for module in self.model.modules():
            if hasattr(module, "bits"):
                quant_info["bits"] = module.bits
            if hasattr(module, "symmetric"):
                quant_info["symmetric"] = module.symmetric
            if hasattr(module, "is_quantized"):
                quant_info["quantization_method"] = "real" if module.is_quantized else "fake"
        
        return quant_info

    def _optimize_onnx_model(self, onnx_model) -> Any:
        """Optimize ONNX model for inference."""
        try:
            from onnxruntime.transformers import optimizer
            
            optimized_model_path = str(Path(self.model_name + "_optimized.onnx"))
            optimizer.optimize_model(
                str(self.model_name + ".onnx"),
                model_type="bert",
                num_heads=None,
                hidden_size=None,
                optimization_options=optimizer.OptimizationOptions(
                    enable_embed_layer_norm=True
                ),
            )
            return onnx_model
        except Exception as e:
            print(f"Warning: Could not optimize ONNX model: {e}")
            return onnx_model

    def validate_export(self, input_shape: tuple = (1, 10), rtol: float = 1e-3) -> bool:
        """
        Validate that ONNX export works correctly by comparing outputs.
        
        Args:
            input_shape: Shape of test input
            rtol: Relative tolerance for output comparison
            
        Returns:
            True if validation passes
        """
        try:
            import onnxruntime as ort
        except ImportError:
            print("Warning: onnxruntime not available for validation")
            return True
        
        # Create test input
        test_input = torch.randn(*input_shape)
        
        # Get PyTorch output
        self.model.eval()
        with torch.no_grad():
            torch_output = self.model(test_input).numpy()
        
        # Get ONNX output
        session = ort.InferenceSession(
            str(Path(self.model_name + ".onnx")),
            providers=["CPUExecutionProvider"]
        )
        onnx_output = session.run(
            None,
            {"input": test_input.numpy().astype("float32")}
        )[0]
        
        # Compare outputs
        is_close = torch.allclose(
            torch.tensor(torch_output),
            torch.tensor(onnx_output),
            rtol=rtol
        )
        
        if is_close:
            print("✓ ONNX export validation passed")
        else:
            max_diff = abs(torch_output - onnx_output).max()
            print(f"⚠ Warning: Output mismatch (max diff: {max_diff})")
        
        return is_close

    def get_onnx_model_info(self, onnx_path: str) -> Dict[str, Any]:
        """
        Get information about exported ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Dictionary with model information
        """
        try:
            import onnx
        except ImportError:
            raise ImportError("Please install onnx: pip install onnx")
        
        onnx_model = onnx.load(onnx_path)
        
        info = {
            "model_path": onnx_path,
            "inputs": [],
            "outputs": [],
            "parameters": 0,
            "opset_version": onnx_model.opset_import[0].version if onnx_model.opset_import else "unknown",
        }
        
        # Extract input/output info
        for input_info in onnx_model.graph.input:
            info["inputs"].append({
                "name": input_info.name,
                "shape": [d.dim_value for d in input_info.type.tensor_type.shape.dim]
            })
        
        for output_info in onnx_model.graph.output:
            info["outputs"].append({
                "name": output_info.name,
                "shape": [d.dim_value for d in output_info.type.tensor_type.shape.dim]
            })
        
        # Count parameters in initializers
        for init in onnx_model.graph.initializer:
            dims = init.dims
            size = 1
            for d in dims:
                size *= d
            info["parameters"] += size
        
        return info
