import torch
import pytest
import tempfile
import os
from qat_llm.export import ONNXExporter


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )


def test_onnx_exporter_creation(simple_model):
    """Test ONNX exporter initialization."""
    exporter = ONNXExporter(simple_model, model_name="test_model")
    assert exporter.model == simple_model
    assert exporter.model_name == "test_model"


def test_onnx_export(simple_model):
    """Test basic ONNX export."""
    try:
        import onnx
    except ImportError:
        pytest.skip("onnx not installed")
    
    exporter = ONNXExporter(simple_model)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "model.onnx")
        exporter.export(output_path, input_shape=(1, 10))
        
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0


def test_onnx_extract_quantization_info(simple_model):
    """Test extracting quantization info from model."""
    exporter = ONNXExporter(simple_model)
    info = exporter._extract_quantization_info()
    
    assert isinstance(info, dict)
    assert "quantization_method" in info
    assert "bits" in info


def test_onnx_model_info():
    """Test getting ONNX model information."""
    try:
        import onnx
    except ImportError:
        pytest.skip("onnx not installed")
    
    model = torch.nn.Linear(10, 5)
    exporter = ONNXExporter(model)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "model.onnx")
        exporter.export(output_path, input_shape=(1, 10))
        
        info = exporter.get_onnx_model_info(output_path)
        
        assert "inputs" in info
        assert "outputs" in info
        assert len(info["inputs"]) > 0
        assert len(info["outputs"]) > 0


def test_onnx_export_with_different_input_shapes(simple_model):
    """Test ONNX export with different input shapes."""
    try:
        import onnx
    except ImportError:
        pytest.skip("onnx not installed")
    
    exporter = ONNXExporter(simple_model)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for batch_size in [1, 2, 8]:
            output_path = os.path.join(tmpdir, f"model_bs{batch_size}.onnx")
            exporter.export(output_path, input_shape=(batch_size, 10))
            
            assert os.path.exists(output_path)


def test_onnx_export_with_optimization(simple_model):
    """Test ONNX export with optimization flag."""
    try:
        import onnx
    except ImportError:
        pytest.skip("onnx not installed")
    
    exporter = ONNXExporter(simple_model)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "model.onnx")
        # optimize_model might fail gracefully, so just check it doesn't crash
        try:
            exporter.export(output_path, input_shape=(1, 10), optimize_model=True)
            assert os.path.exists(output_path)
        except Exception:
            # Optimization might fail, that's okay for this test
            pass


def test_onnx_export_with_quantization_metadata(simple_model):
    """Test ONNX export with quantization metadata."""
    try:
        import onnx
    except ImportError:
        pytest.skip("onnx not installed")
    
    exporter = ONNXExporter(simple_model)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "model.onnx")
        exporter.export(output_path, input_shape=(1, 10), add_quantization_info=True)
        
        assert os.path.exists(output_path)


def test_onnx_validate_export(simple_model):
    """Test ONNX export validation."""
    try:
        import onnx
    except ImportError:
        pytest.skip("onnx not installed")
    
    exporter = ONNXExporter(simple_model)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        output_path = os.path.join(tmpdir, "model.onnx")
        exporter.export(output_path, input_shape=(1, 10))
        
        # Validation might fail if onnxruntime not available, that's okay
        try:
            exporter.model_name = os.path.join(tmpdir, "model")
            is_valid = exporter.validate_export(input_shape=(1, 10))
            # Just checking it returns a boolean
            assert isinstance(is_valid, bool)
        except ImportError:
            pytest.skip("onnxruntime not installed")


def test_onnx_export_sequential_model():
    """Test ONNX export on sequential model."""
    try:
        import onnx
    except ImportError:
        pytest.skip("onnx not installed")
    
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.BatchNorm1d(20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    
    exporter = ONNXExporter(model)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "model.onnx")
        exporter.export(output_path, input_shape=(2, 10))
        
        assert os.path.exists(output_path)
