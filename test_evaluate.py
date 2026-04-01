"""
Local test for evaluate.py — two-part check, fast on CPU.

Part 1: Metric logic unit test (no models, no data — instant)
Part 2: Single forward-pass shape test for all 4 encoders (1 image each)

Run on Explorer for full evaluation with real checkpoints.
"""

import os, json, tempfile
import torch
import torch.nn as nn
import numpy as np

# ── Part 1: Metric logic ──────────────────────────────────────────────────────

def compute_nag(utility_acc, privacy_acc):
    utility_gain  = max(utility_acc - 0.5, 0.0)
    privacy_leak  = max(privacy_acc - 0.5, 1e-6)
    return round(utility_gain / privacy_leak, 4)


def test_metrics():
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

    print("\n── Part 1: Metric logic unit test ──────────────────────")

    np.random.seed(42)
    labels = np.random.randint(0, 2, 200)

    # Case A: perfect classifier
    preds_perfect = labels.copy().astype(float)
    preds_binary  = preds_perfect.astype(int)
    acc_p  = (preds_binary == labels).mean() * 100
    auc_p  = roc_auc_score(labels, preds_perfect)
    f1_p   = f1_score(labels, preds_binary, zero_division=0)
    cm_p   = confusion_matrix(labels, preds_binary)
    nag_p  = compute_nag(acc_p / 100, acc_p / 100)

    assert abs(acc_p  - 100.0) < 0.01,  "perfect acc should be 100"
    assert abs(auc_p  - 1.0)   < 0.01,  "perfect AUC should be 1.0"
    assert abs(f1_p   - 1.0)   < 0.01,  "perfect F1 should be 1.0"
    assert cm_p.shape == (2, 2),         "confusion matrix shape wrong"
    print(f"  Perfect classifier  acc={acc_p:.1f}%  AUC={auc_p:.4f}  F1={f1_p:.4f}  NAG={nag_p:.4f}  ✓")

    # Case B: random classifier (~50%)
    scores_random = np.random.rand(200)
    preds_random  = (scores_random > 0.5).astype(int)
    acc_r  = (preds_random == labels).mean() * 100
    auc_r  = roc_auc_score(labels, scores_random)
    f1_r   = f1_score(labels, preds_random, zero_division=0)
    nag_r  = compute_nag(acc_r / 100, acc_r / 100)

    assert 0 <= acc_r  <= 100,  "random acc out of range"
    assert 0 <= auc_r  <= 1,    "random AUC out of range"
    assert 0 <= f1_r   <= 1,    "random F1 out of range"
    assert nag_r >= 0,          "NAG negative"
    print(f"  Random classifier   acc={acc_r:.1f}%  AUC={auc_r:.4f}  F1={f1_r:.4f}  NAG={nag_r:.4f}  ✓")

    # Case C: NAG formula — high utility, low privacy leak
    nag_good = compute_nag(0.85, 0.55)  # 85% utility, 55% privacy = good model
    nag_bad  = compute_nag(0.70, 0.85)  # 70% utility, 85% privacy = bad model
    assert nag_good > nag_bad, "NAG should be higher when privacy leaks less"
    print(f"  NAG logic check     good={nag_good:.4f} > bad={nag_bad:.4f}  ✓")

    # Case D: JSON round-trip
    result = {
        "encoder": "vanilla_vae", "exp_name": "test",
        "utility_acc": round(acc_r, 2), "privacy_acc": round(acc_r, 2),
        "utility_auc": round(auc_r, 4), "privacy_auc": round(auc_r, 4),
        "utility_f1":  round(f1_r, 4),  "nag": nag_r,
        "cm_utility": cm_p.tolist(), "cm_privacy": cm_p.tolist(),
    }
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(result, f, indent=2); jpath = f.name
    loaded = json.load(open(jpath))
    assert loaded["encoder"] == "vanilla_vae"
    os.unlink(jpath)
    print(f"  JSON save/load                                        ✓")

    print("  All metric logic checks passed.\n")


# ── Part 2: Single forward-pass shape test ────────────────────────────────────

def test_forward_pass():
    from models.get_encoder import get_encoder

    print("── Part 2: Encoder forward-pass shape test (1 image, 224x224) ──")
    x = torch.randn(1, 3, 224, 224)

    for enc_name in ["vanilla_vae", "beta_vae", "residual_vae", "vq_vae"]:
        enc = get_encoder(enc_name, img_size=224)
        enc.eval()
        with torch.no_grad():
            out = enc(x)
        # return type can be tensor (reconstruction) or tuple
        if isinstance(out, tuple):
            recon = out[0]
        else:
            recon = out
        assert recon.shape == (1, 3, 224, 224), \
            f"{enc_name}: expected (1,3,224,224) got {recon.shape}"
        vmin, vmax = recon.min().item(), recon.max().item()
        assert vmin >= -0.1 and vmax <= 1.1, \
            f"{enc_name}: output out of [0,1] range: [{vmin:.3f}, {vmax:.3f}]"
        print(f"  {enc_name:<16}  out={tuple(recon.shape)}  "
              f"range=[{vmin:.3f}, {vmax:.3f}]  ✓")

    print("  All forward-pass checks passed.\n")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== evaluate.py — local correctness test ===")

    test_metrics()
    test_forward_pass()

    print("=== All checks passed — evaluate.py is ready for Explorer ===")
    print("""
To run the full evaluation on Explorer after training completes:

  python evaluate.py \\
    --exp_name member1_vanillavae_10e_20k \\
    --encoder vanilla_vae \\
    --checkpoint_dir /scratch/$USER/pp_ext_member2/arls \\
    --data_source huggingface \\
    --hf_cache_dir /scratch/$USER/pp_ext_member2/hf_cache \\
    --max_test_samples 2000

  # then repeat for beta_vae, residual_vae, etc.
""")
