#!/usr/bin/env python3
"""Pre-download embedding model to avoid cold-start timeout on cloud servers.

Usage:
    python download_emb_model.py                        # downloads to ./models/bge-small-zh-v1.5
    python download_emb_model.py --output ./custom/path
"""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download BGE small-zh embedding model")
    parser.add_argument("--output", type=str, default="./models/bge-small-zh-v1.5",
                        help="Output directory (default: ./models/bge-small-zh-v1.5)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    cache_env = str(output_dir.parent / "huggingface" / "hub")

    print(f"[1] Downloading via huggingface-cli to {output_dir}")
    env = {**dict(__import__("os").environ),
           "HF_HUB_CACHE": cache_env}

    # Try HF mirror first (China-friendly), then raw HF
    mirrors = [
        ("https://hf-mirror.com", ""),
        ("https://huggingface.co", ""),
    ]

    model_id = "BAAI/bge-small-zh-v1.5"
    
    success = False
    for base_url, hf_cache in mirrors:
        print(f"\n[*] Trying {base_url}...")
        try:
            result = subprocess.run(
                ["python", "-m", "huggingface_hub", "snapshot_download", model_id,
                 "--local-dir", str(output_dir)],
                capture_output=True, text=True,
                env={**env, "HF_MIRROR": base_url.split("://")[0]},
                timeout=600
            )
            if result.returncode == 0:
                print(f"✓ Downloaded to {output_dir}")
                print(f"  Contents: {[f.name for f in output_dir.iterdir()]}")
                success = True
                break
            else:
                print(f"✗ Failed: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout downloading from {base_url}")
        except FileNotFoundError:
            print("✗ huggingface_hub module not installed. Run: pip install huggingface_hub")

    if not success:
        print("\n⚠ All mirrors failed. Manual download:")
        print(f"  pip install huggingface_hub")
        print(f"  huggingface-cli download {model_id} --local-dir {output_dir}")
        sys.exit(1)

    print(f"\n✓ Ready! Set EMB_MODEL={output_dir}/snapshots/master in your .env")

if __name__ == "__main__":
    main()
