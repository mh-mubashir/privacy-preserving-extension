"""
compute_flops.py — FLOPs, MACs, and parameter count for all encoder variants.

Addresses the professor's feedback: "define deployment constraints (latency,
model size, power targets) and tie your FLOPs/MACs analysis to those limits."

Uses thop (pip install thop) to profile each encoder on a single 224x224 image.
Prints a comparison table and rough latency estimates for three edge devices.

Usage:
    python compute_flops.py

Requirements:
    pip install thop
"""

import torch
from models.get_encoder import get_encoder

try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    print("ERROR: thop not installed. Run: pip install thop")
    HAS_THOP = False

# Edge device throughput estimates (MACs/second, rough order-of-magnitude)
# Sources: vendor datasheets and published benchmarks
EDGE_DEVICES = {
    "Jetson Nano (4W)":    6e9,    # ~6 GMACs/s at 4W
    "Raspberry Pi 5":      2e9,    # ~2 GMACs/s
    "Coral Edge TPU":      4e12,   # ~4 TMACs/s (INT8 only, FP32 shown as lower bound)
}

ENCODERS = [
    ("vanilla_vae",           "Member 1 — Vanilla VAE"),
    ("beta_vae",              "Member 1 — Beta-VAE (β=4)"),
    ("residual_vae",          "Member 1 — Residual VAE"),
    ("beta_tc_vae",           "Member 2 — Beta-TC VAE"),
    ("disentangled_beta_vae", "Member 2 — Disentangled Beta-VAE"),
    ("vq_vae",                "Member 3 — VQ-VAE"),
]


def profile_encoder(name, img_size=224):
    enc = get_encoder(name, img_size=img_size)
    enc.eval()
    x = torch.randn(1, 3, img_size, img_size)
    macs, params = profile(enc, inputs=(x,), verbose=False)
    return macs, params


def latency_ms(macs, device_macs_per_sec):
    return (macs / device_macs_per_sec) * 1000


def main():
    if not HAS_THOP:
        return

    print("\n" + "=" * 90)
    print("  ENCODER SYSTEM CHARACTERISATION — FLOPs / MACs / Parameters")
    print("  Input: single 224×224 RGB image")
    print("=" * 90)

    results = []
    for name, label in ENCODERS:
        try:
            macs, params = profile_encoder(name)
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
