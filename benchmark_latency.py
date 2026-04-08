"""
benchmark_latency.py — Measure actual CPU inference latency for all encoder variants.

Uses ONNX Runtime to run each exported model 50 times and reports
mean / std / min latency per image. This is real measured latency,
not an estimate from MACs alone.

Usage:
    python benchmark_latency.py [--onnx_dir ./onnx_exports] [--runs 50]
"""

import argparse
import os
import time
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: pip install onnxruntime")
    exit(1)


ENCODERS = [
    ("vanilla_vae",           "Vanilla VAE"),
    ("beta_vae",              "Beta-VAE (β=4)"),
    ("residual_vae",          "Residual VAE"),
    ("beta_tc_vae",           "Beta-TC VAE"),
    ("disentangled_beta_vae", "Disentangled Beta-VAE"),
    ("vq_vae",                "VQ-VAE"),
]

# Edge device CPU performance ratios relative to a modern laptop CPU
# (approximate, based on published benchmarks)
DEVICE_RATIOS = {
    "Your CPU (measured)": 1.0,
    "Jetson Nano CPU":     8.0,
    "Raspberry Pi 5":      4.0,
    "Cortex-A76 mobile":   3.0,
}


def benchmark(onnx_path: str, runs: int = 50, img_size: int = 224):
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1   # single-threaded to simulate edge device
    sess = ort.InferenceSession(onnx_path, sess_opts,
                                providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    # warmup
    for _ in range(5):
        sess.run(None, {inp_name: dummy})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), np.std(times), np.min(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", type=str, default="./onnx_exports")
    parser.add_argument("--runs",     type=int, default=50)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"  INFERENCE LATENCY BENCHMARK — ONNX Runtime (CPU, single-threaded)")
    print(f"  Input: 1 × 3 × {args.img_size} × {args.img_size}   Runs: {args.runs}")
    print(f"{'='*80}\n")

    results = []
    for name, label in ENCODERS:
        path = os.path.join(args.onnx_dir, f"{name}.onnx")
        if not os.path.exists(path):
            print(f"  [skip]  {label} — {path} not found")
            results.append((name, label, None, None, None))
            continue
        print(f"  Benchmarking {label}...", end=" ", flush=True)
        mean_ms, std_ms, min_ms = benchmark(path, args.runs, args.img_size)
        print(f"{mean_ms:.1f}ms ± {std_ms:.1f}ms")
        results.append((name, label, mean_ms, std_ms, min_ms))

    valid = [(n, l, m, s, mn) for n, l, m, s, mn in results if m is not None]

    print(f"\n\n{'='*80}")
    print(f"  RESULTS TABLE — CPU latency per image (ms)")
    print(f"{'='*80}")
    print(f"  {'Encoder':<28} {'Mean':>8} {'Std':>7} {'Min':>7}  {'<100ms?':>8}")
    print(f"  {'-'*60}")
    for name, label, mean_ms, std_ms, min_ms in results:
        if mean_ms is None:
            print(f"  {name:<28}  {'not found':>8}")
            continue
        ok = "✓" if mean_ms < 100 else "✗"
        print(f"  {name:<28} {mean_ms:>7.1f}ms {std_ms:>6.1f}ms {min_ms:>6.1f}ms  {ok:>8}")

    print(f"\n\n{'='*80}")
    print(f"  ESTIMATED LATENCY ON EDGE DEVICES (scaled from measured CPU)")
    print(f"  Note: actual latency depends on memory, OS, and hardware specifics.")
    print(f"{'='*80}")
    header = f"  {'Encoder':<28}"
    for dev in DEVICE_RATIOS:
        header += f"  {dev[:18]:>18}"
    print(header)
    print(f"  {'-'*76}")
    for name, label, mean_ms, std_ms, min_ms in results:
        if mean_ms is None:
            continue
        row = f"  {name:<28}"
        for dev, ratio in DEVICE_RATIOS.items():
            est = mean_ms * ratio
            row += f"  {est:>16.0f}ms"
        print(row)

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
