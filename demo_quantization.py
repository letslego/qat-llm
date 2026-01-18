"""
Comprehensive demo showing fake vs real quantization pipeline.
Demonstrates: fake quantization → PTQ → real quantization → ONNX export → comparison.
"""

import torch
import torch.nn as nn
from pathlib import Path

from qat_llm.fake_quant import FakeQuantizeLinear, apply_fake_quant_to_model
from qat_llm.real_quant import RealQuantizeLinear, convert_to_real_quant
from qat_llm.ptq import PTQPipeline
from qat_llm.export import ONNXExporter
from qat_llm.quantization_comparison import QuantizationComparison


class SimpleDataLoader:
    """Toy data loader for demonstration."""
    
    def __init__(self, num_batches: int = 10, batch_size: int = 32, input_size: int = 784):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.input_size = input_size
    
    def __iter__(self):
        for _ in range(self.num_batches):
            x = torch.randn(self.batch_size, self.input_size)
            y = torch.randint(0, 10, (self.batch_size,))
            yield x, y


def create_sample_model(input_size: int = 784, num_classes: int = 10) -> nn.Module:
    """Create a simple neural network model."""
    return nn.Sequential(
        nn.Linear(input_size, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )


def demo_fake_quantization():
    """Demonstrate fake quantization training."""
    print("\n" + "=" * 80)
    print("FAKE QUANTIZATION DEMO".center(80))
    print("=" * 80)
    
    # Create model
    model = create_sample_model()
    print(f"\n✓ Created original model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Convert to fake quantization
    fake_quant_model = apply_fake_quant_to_model(model, bits=8)
    print("✓ Applied fake quantization layers")
    
    # Simulate training
    fake_quant_model.train()
    calibration_loader = SimpleDataLoader(num_batches=5)
    
    print("\nCalibrating fake quantization layers...")
    with torch.no_grad():
        for i, (x, _) in enumerate(calibration_loader):
            output = fake_quant_model(x)
            # In real training, you would backprop gradients here
            if (i + 1) % 2 == 0:
                print(f"  Batch {i + 1}/5 - Output shape: {output.shape}")
    
    print("✓ Fake quantization training complete")
    return fake_quant_model


def demo_ptq():
    """Demonstrate Post-Training Quantization (PTQ)."""
    print("\n" + "=" * 80)
    print("POST-TRAINING QUANTIZATION (PTQ) DEMO".center(80))
    print("=" * 80)
    
    # Create original model
    original_model = create_sample_model()
    print(f"\n✓ Created original model with {sum(p.numel() for p in original_model.parameters())} parameters")
    
    # Create PTQ pipeline
    pipeline = PTQPipeline(original_model, bits=8)
    print("✓ Created PTQ pipeline (8-bit quantization)")
    
    # Calibrate
    print("\nCalibrating model with representative data...")
    calibration_loader = SimpleDataLoader(num_batches=5, batch_size=32)
    pipeline.calibrate(calibration_loader, num_batches=3)
    
    # Convert to real quantization
    print("\nConverting to real quantization...")
    quantized_model = pipeline.convert_to_real_quant()
    print("✓ Converted all Linear layers to RealQuantizeLinear")
    
    # Quantize weights
    print("\nQuantizing weights to int8...")
    pipeline.model = quantized_model
    pipeline.quantize_weights()
    print("✓ Weights quantized and stored as int8")
    
    # Get model sizes
    original_size, quantized_size = pipeline.get_model_size()
    compression_ratio = original_size / quantized_size
    
    print(f"\n📊 Model Size Comparison:")
    print(f"  Original (float32):  {original_size:.4f} MB")
    print(f"  Quantized (int8):    {quantized_size:.4f} MB")
    print(f"  Compression Ratio:   {compression_ratio:.2f}x")
    
    return quantized_model


def demo_onnx_export():
    """Demonstrate ONNX export."""
    print("\n" + "=" * 80)
    print("ONNX EXPORT DEMO".center(80))
    print("=" * 80)
    
    # Create and quantize model
    model = create_sample_model()
    quantized_model = convert_to_real_quant(model, bits=8)
    print("\n✓ Created real quantized model")
    
    # Export to ONNX
    print("\nExporting to ONNX format...")
    exporter = ONNXExporter(quantized_model, model_name="qat_demo_model")
    
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "qat_demo_model.onnx"
    
    exporter.export(
        str(output_path),
        input_shape=(1, 784),
        opset_version=14,
        optimize_model=False,  # Skip optimization for demo
    )
    
    # Get model info
    info = exporter.get_onnx_model_info(str(output_path))
    print(f"\n📋 ONNX Model Information:")
    print(f"  Inputs:  {info['inputs']}")
    print(f"  Outputs: {info['outputs']}")
    print(f"  Parameters: {info['parameters']}")
    print(f"  Opset Version: {info['opset_version']}")
    
    return output_path


def demo_comparison():
    """Demonstrate quantization comparison."""
    print("\n" + "=" * 80)
    print("QUANTIZATION COMPARISON DEMO".center(80))
    print("=" * 80)
    
    # Create models
    print("\n📦 Setting up models...")
    original_model = create_sample_model()
    fake_quant_model = apply_fake_quant_to_model(create_sample_model(), bits=8)
    real_quant_model = convert_to_real_quant(create_sample_model(), bits=8)
    
    print("✓ Original model created")
    print("✓ Fake quantized model created")
    print("✓ Real quantized model created")
    
    # Create comparison
    comparison = QuantizationComparison(original_model)
    comparison.setup_models(fake_quant_model, real_quant_model)
    print("\n✓ Comparison engine initialized")
    
    # Run comparison
    print("\n🚀 Running benchmarks...")
    test_loader = SimpleDataLoader(num_batches=5, batch_size=32)
    
    results = comparison.compare(
        test_loader,
        input_shape=(32, 784),
        device="cpu",
        num_eval_batches=2
    )
    
    print("\n✓ Benchmarks complete!")
    
    # Print results
    comparison.print_comparison()
    
    # Save report
    report_path = Path("models") / "comparison_report.json"
    report_path.parent.mkdir(exist_ok=True)
    comparison.save_comparison_report(str(report_path))
    
    return comparison


def demo_full_pipeline():
    """Run the complete pipeline."""
    print("\n" + "🎯" * 40)
    print("QUANTIZATION-AWARE TRAINING (QAT) FULL PIPELINE DEMO".center(80))
    print("🎯" * 40)
    
    print("\nThis demo shows:")
    print("  1️⃣  Fake Quantization - Simulating quantization during training")
    print("  2️⃣  Post-Training Quantization (PTQ) - Quantizing a trained model")
    print("  3️⃣  Real Quantization - Converting to int8 operations")
    print("  4️⃣  ONNX Export - Exporting for deployment")
    print("  5️⃣  Comparison - Benchmarking different quantization methods")
    
    try:
        # Demo 1: Fake Quantization
        fake_quant_model = demo_fake_quantization()
        
        # Demo 2: PTQ
        real_quant_model = demo_ptq()
        
        # Demo 3: ONNX Export
        onnx_path = demo_onnx_export()
        
        # Demo 4: Comparison
        comparison = demo_comparison()
        
        # Summary
        print("\n" + "=" * 80)
        print("DEMO COMPLETE ✓".center(80))
        print("=" * 80)
        print("\n📁 Generated files:")
        print("  • models/qat_demo_model.onnx - Exported ONNX model")
        print("  • models/comparison_report.json - Detailed comparison report")
        
        print("\n🔑 Key Takeaways:")
        print("  • Fake quantization simulates int8 behavior during training")
        print("  • PTQ quickly quantizes pre-trained models without retraining")
        print("  • Real quantization uses actual int8 operations for deployment")
        print("  • ONNX export enables deployment across different platforms")
        print("  • Comparison shows quantization's impact on performance vs accuracy")
        
        print("\n📚 Next Steps:")
        print("  • Fine-tune quantization parameters for your use case")
        print("  • Deploy ONNX models on edge devices (mobile, IoT)")
        print("  • Use PTQ for quick model compression")
        print("  • Combine with knowledge distillation for better accuracy")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_full_pipeline()
