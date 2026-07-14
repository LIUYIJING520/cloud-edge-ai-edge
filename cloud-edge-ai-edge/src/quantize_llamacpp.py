"""llama.cpp INT4 quantization script.

Run this on a Linux/WSL server with the model downloaded.
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_MODEL = PROJECT_ROOT / "data" / "models" / "DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_GGUF = PROJECT_ROOT / "deploy" / "edge_model_q4.gguf"
LLAMACPP_DIR = PROJECT_ROOT / "tools" / "llama.cpp"


def main():
    if not SOURCE_MODEL.exists():
        print(f"Model not found: {SOURCE_MODEL}")
        print("Run scripts/download_data.py first")
        sys.exit(1)
    if not LLAMACPP_DIR.exists():
        print(f"Cloning llama.cpp to {LLAMACPP_DIR}...")
        LLAMACPP_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "git", "clone", "--depth=1",
            "https://gh-proxy.com/https://github.com/ggerganov/llama.cpp.git",
            str(LLAMACPP_DIR),
        ], check=True)
    convert_script = LLAMACPP_DIR / "convert_hf_to_gguf.py"
    if convert_script.exists():
        intermediate = PROJECT_ROOT / "deploy" / "edge_model_fp16.gguf"
        if not intermediate.exists():
            print(f"HF -> FP16 GGUF...")
            subprocess.run([
                sys.executable, str(convert_script),
                str(SOURCE_MODEL),
                "--outfile", str(intermediate),
                "--outtype", "f16",
            ], cwd=str(LLAMACPP_DIR), check=True)
        OUTPUT_GGUF.parent.mkdir(parents=True, exist_ok=True)
        quantize_bin = LLAMACPP_DIR / "build" / "bin" / "quantize"
        if quantize_bin.exists():
            print(f"INT4 量化...")
            subprocess.run([
                str(quantize_bin),
                str(intermediate),
                str(OUTPUT_GGUF),
                "Q4_K_M",
            ], check=True)
            print(f"\nDone: {OUTPUT_GGUF}")
            print(f"Size: {OUTPUT_GGUF.stat().st_size / 1024 / 1024:.0f} MB")
        else:
            print(f"Build llama.cpp first:")
            print(f"  cd {LLAMACPP_DIR}")
            print(f"  cmake -B build")
            print(f"  cmake --build build --config Release --target quantize")
    else:
        print(f"convert_hf_to_gguf.py not found in {LLAMACPP_DIR}")


if __name__ == "__main__":
    main()
