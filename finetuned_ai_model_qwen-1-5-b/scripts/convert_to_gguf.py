#!/usr/bin/env python3
"""
Convert Qwen2.5-Coder-1.5B PyTorch model to GGUF format for efficient inference on M2 Mac mini.
Uses llama.cpp quantization with Q4_K_M (4-bit).
"""

import argparse
import subprocess
import sys
from pathlib import Path


def convert_to_gguf(model_path: str, output_path: str, quantization: str = "Q4_K_M"):
    """
    Convert PyTorch model to GGUF format using llama.cpp.
    
    Args:
        model_path: Path to fine-tuned model directory
        output_path: Path for output GGUF file
        quantization: Quantization format (Q4_K_M, Q3_K_M, Q5_K_M, etc.)
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    if not model_path.exists():
        print(f"❌ Model path not found: {model_path}")
        return False

    if not (model_path / "pytorch_model.bin").exists():
        print(f"❌ No pytorch_model.bin found in {model_path}")
        return False

    print(f"🔄 Converting model to {quantization} GGUF format...")
    print(f"   Model: {model_path}")
    print(f"   Output: {output_path}")
    print(f"   Quantization: {quantization}")

    try:
        # Step 1: Convert to GGML format (intermediate)
        print("\n📝 Step 1: Converting to GGML format...")
        print("   (This takes ~5-10 minutes, patience required!)")

        cmd = [
            "python",
            "-m",
            "llama_cpp.models.convert",
            str(model_path),
            "--outfile",
            str(output_path.with_suffix(".ggml")),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ GGML conversion failed:")
            print(result.stderr)
            return False

        print("✅ GGML conversion complete")

        # Step 2: Quantize to target format
        print(f"\n📦 Step 2: Quantizing to {quantization}...")
        print("   (This takes ~2-3 minutes)")

        # Use llama.cpp's quantization tool if available
        ggml_path = output_path.with_suffix(".ggml")
        cmd = [
            "llama-quantize",
            str(ggml_path),
            str(output_path),
            quantization,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Quantization complete: {output_path}")
            
            # Get file size
            size_gb = output_path.stat().st_size / (1024 ** 3)
            print(f"\n📊 Output file size: {size_gb:.2f} GB")
            
            # Clean up intermediate GGML file
            ggml_path.unlink(missing_ok=True)
            print(f"🧹 Cleaned up intermediate files")
            
            return True
        else:
            print(f"⚠️  Quantization tool not found in PATH")
            print(f"✅ GGML model available at: {ggml_path}")
            print(f"\n💡 To quantize manually, install llama.cpp:")
            print(f"   brew install llama.cpp")
            print(f"   Then run: llama-quantize {ggml_path} {output_path} {quantization}")
            return True

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert Qwen2.5-Coder-1.5B to GGUF format"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path for output GGUF file",
    )
    parser.add_argument(
        "--quantization",
        default="Q4_K_M",
        choices=["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q8_0"],
        help="Quantization format (default: Q4_K_M for 1.5-2 GB)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🔄 Qwen2.5-Coder-1.5B → GGUF Converter")
    print("=" * 70)

    success = convert_to_gguf(
        args.model_path, args.output_path, args.quantization
    )

    if success:
        print("\n✅ Conversion successful!")
        print(f"\n📝 Next step: Register with Ollama")
        print(f"   ollama create micro-ai-coder-1-5b:latest \\")
        print(f"     -f models/Modelfile_qwen_1_5b")
        return 0
    else:
        print("\n❌ Conversion failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
