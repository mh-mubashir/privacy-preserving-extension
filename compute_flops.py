"""
compute_flops.py -- FLOPs, MACs, and parameter count for all ARL encoder variants.

Profiles each encoder on a single input image using thop and prints:
  1. Per-encoder MACs and parameter count
  2. Comparison table with relative compute cost
  3. Estimated inference latency on common edge devices (Jetson Nano, RPi 5, Coral TPU)

Edge device throughput estimates are order-of-magnitude figures from vendor
datasheets; real latency depends on memory bandwidth, quantisation, and batch
size. Use benchmark_latency.py for measured ONNX Runtime latency.

Results from this script informed the edge deployment analysis in Table VI
of the project report and the <200ms Jetson Nano deployment target.

Usage:
    python compute_flops.py
    python compute_flops.py --img_size 64    # for VQ-VAE trained at 64px

Requirements:
    pip install thop
"""

import argparse
from typing import Dict, List, Optional, Tuple

import torch

from models.get_encoder import get_encoder

try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    print("ERROR: thop not installed. Run: pip install thop")
    HAS_THOP = False

# Edge device throughput estimates (MACs/second, order-of-magnitude)
# Sources: NVIDIA Jetson Nano datasheet, RPi foundation benchmarks, Google Coral docs
EDGE_DEVICES: Dict[str, float] = {
    "Jetson Nano (4W)":    6e9,    # ~6 GMACs/s at 4W
    "Raspberry Pi 5":      2e9,    # ~2 GMACs/s
    "Coral Edge TPU":      4e12,   # ~4 TMACs/s (INT8 only; FP32 shown as lower bound)
}

ENCODERS: List[Tuple[str, str]] = [
    ("vanilla_vae",           "Vanilla VAE"),
    ("beta_vae",              "Beta-VAE (b=4)"),
    ("residual_vae",          "Residual VAE"),
    ("beta_tc_vae",           "Beta-TC VAE"),
    ("disentangled_beta_vae", "Disentangled Beta-VAE"),
    ("vq_vae",                "VQ-VAE"),
]


def profile_encoder(name: str, img_size: int = 224) -> Tuple[float, float]:
    """Return (MACs, param_count) for the named encoder on a single image."""
    enc = get_encoder(name, img_size=img_size)
    enc.eval()
    x = torch.randn(1, 3, img_size, img_size)
    macs, params = profile(enc, inputs=(x,), verbose=False)
    return macs, params


def latency_ms(macs: float, device_macs_per_sec: float) -> float:
    """Estimate inference latency in milliseconds from MACs and device throughput."""
    return (macs / device_macs_per_sec) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile encoder FLOPs and MACs.")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image resolution (default: 224). Use 64 for VQ-VAE.")
    args = parser.parse_args()

    if not HAS_THOP:
        return

    print("\n" + "=" * 90)
    print(f"  ENCODER SYSTEM CHARACTERISATION — FLOPs / MACs / Parameters")
    print(f"  Input: single {args.img_size}×{args.img_size} RGB image")
    print("=" * 90)

    results = []
    for name, label in ENCODERS:
        try:
            macs, params = profile_encoder(name, img_size=args.img_size)
            results.append((name, label, macs, params))
            macs_str, params_str = clever_format([macs, params], "%.2f")
            print(f"\n  {label}")
            print(f"    MACs       : {macs_str}")
            print(f"    Parameters : {params_str}")
        except Exception as e:
            print(f"\n  {label}")
            print(f"    ERROR: {e}")
            results.append((name, label, None, None))

    print("\n\n" + "=" * 90)
    print("  COMPARISON TABLE")
    print("=" * 90)
    print(f"  {'Encoder':<28} {'MACs':>10}  {'Params':>10}  {'Relative Cost':>14}")
    print(f"  {'-'*66}")

    valid = [(n, l, m, p) for n, l, m, p in results if m is not None]
    min_macs = min(r[2] for r in valid) if valid else 1

    for name, label, macs, params in results:
        if macs is None:
            print(f"  {name:<28} {'ERROR':>10}  {'ERROR':>10}")
            continue
        macs_str, params_str = clever_format([macs, params], "%.2f")
        relative = macs / min_macs
        bar = "█" * int(relative * 5)
        print(f"  {name:<28} {macs_str:>10}  {params_str:>10}  {relative:>5.1f}x  {bar}")

    print("\n\n" + "=" * 90)
    print("  EDGE LATENCY ESTIMATES (ms per image, single-image inference, FP32)")
    print("  Note: real latency depends on memory bandwidth, quantisation, and batching.")
    print("=" * 90)
    header = f"  {'Encoder':<28}"
    for dev in EDGE_DEVICES:
        header += f"  {dev[:20]:>20}"
    print(header)
    print(f"  {'-'*86}")

    for name, label, macs, params in results:
        if macs is None:
            continue
        row = f"  {name:<28}"
        for dev, throughput in EDGE_DEVICES.items():
            ms = latency_ms(macs, throughput)
            row += f"  {ms:>19.1f}ms"
        print(row)

    print("\n" + "=" * 90)
    print("  DEPLOYMENT CONSTRAINT TARGETS (from project proposal)")
    print("  Utility AUC target : 1.0  |  Privacy AUC target : 0.5")
    print("  Latency target     : <100ms per image on Jetson Nano (real-time edge)")
    print("  Model size target  : <50M parameters (fits in 256MB DRAM)")
    print("=" * 90)

    print("\n  Models within Jetson Nano <100ms target:")
    for name, label, macs, params in results:
        if macs is None:
            continue
        ms = latency_ms(macs, EDGE_DEVICES["Jetson Nano (4W)"])
        status = "✓ within target" if ms < 100 else "✗ exceeds target"
        print(f"    {name:<28}  {ms:>6.1f}ms  {status}")

    print()


if __name__ == "__main__":
    main()
