"""
export_onnx.py — Export all encoder variants to ONNX format.

ONNX files can be uploaded to online profiling tools such as:
  - https://netron.app          (visualize network structure)
  - https://onnxinsights.com    (online ONNX profiler)
  - https://nn-meter.microsoft.com (latency prediction for edge devices)
  - https://onn-fpo.ml          (FP ops counter)

Usage:
    python export_onnx.py [--out_dir ./onnx_exports] [--img_size 224]
"""

import argparse
import os
import torch
from models.get_encoder import get_encoder


ENCODERS = [
    "vanilla_vae",
    "beta_vae",
    "residual_vae",
    "beta_tc_vae",
    "disentangled_beta_vae",
    "vq_vae",
]


def export(name: str, img_size: int, out_dir: str):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./onnx_exports")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nExporting encoders to ONNX  (img_size={args.img_size})  →  {args.out_dir}/\n")

    for name in ENCODERS:
        export(name, args.img_size, args.out_dir)

    onnx_files = [f for f in os.listdir(args.out_dir) if f.endswith(".onnx")]
    print(f"\n{len(onnx_files)} ONNX file(s) written to {args.out_dir}/")
    print("\nNext steps — upload to one of these free online tools:")
    print("  Netron visualizer     : https://netron.app")
    print("  nn-Meter (Microsoft)  : https://nn-meter.microsoft.com")
    print("  ONNX Model Zoo runner : https://onnxinsights.com")


if __name__ == "__main__":
    main()
