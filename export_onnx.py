"""
export_onnx.py -- Export all ARL encoder variants to ONNX format.

Exports a single 224x224 (or --img_size) forward pass for each encoder.
Dynamic batch axis is enabled so the exported model accepts any batch size
at inference time. Outputs land in --out_dir (default: ./onnx_exports/).

ONNX files are used by:
  - compute_flops.py / benchmark_latency.py for latency profiling
  - Netron (https://netron.app) for architecture visualisation
  - nn-Meter (https://nn-meter.microsoft.com) for edge latency estimates

Usage:
    python export_onnx.py
    python export_onnx.py --out_dir ./onnx_exports --img_size 224

Requirements:
    pip install torch onnx
"""

import argparse
import os
from typing import List

import torch

from models.get_encoder import get_encoder


ENCODERS: List[str] = [
    "vanilla_vae",
    "beta_vae",
    "residual_vae",
    "beta_tc_vae",
    "disentangled_beta_vae",
    "vq_vae",
]


def export_encoder(name: str, img_size: int, out_dir: str) -> None:
    """Export a single encoder to ONNX, skipping if the file already exists."""
    out_path = os.path.join(out_dir, f"{name}.onnx")
    if os.path.exists(out_path):
        print(f"  [skip]  {name}.onnx already exists")
        return

    try:
        enc = get_encoder(name, img_size=img_size)
        enc.eval()
        dummy = torch.zeros(1, 3, img_size, img_size)
        torch.onnx.export(
            enc,
            dummy,
            out_path,
            input_names=["image"],
            output_names=["recon"],
            opset_version=17,
            dynamic_axes={"image": {0: "batch"}, "recon": {0: "batch"}},
        )
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  [ok]    {name}.onnx  ({size_mb:.1f} MB)")
    except Exception as exc:
        print(f"  [error] {name}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ARL encoders to ONNX")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./onnx_exports",
        help="Directory to write .onnx files (created if absent)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Spatial resolution of the dummy input (use 64 for VQ-VAE trained at 64px)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nExporting {len(ENCODERS)} encoders to ONNX  "
          f"(img_size={args.img_size})  ->  {args.out_dir}/\n")

    for name in ENCODERS:
        export_encoder(name, args.img_size, args.out_dir)

    onnx_files = [f for f in os.listdir(args.out_dir) if f.endswith(".onnx")]
    print(f"\n{len(onnx_files)} ONNX file(s) in {args.out_dir}/")
    print("\nNext steps:")
    print("  Visualise architecture : python -c \"import netron; netron.start('<file>.onnx')\"")
    print("  Latency benchmark      : python benchmark_latency.py --onnx_dir", args.out_dir)
    print("  Online viewer          : https://netron.app")


if __name__ == "__main__":
    main()
