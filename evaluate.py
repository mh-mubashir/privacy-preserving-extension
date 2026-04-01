"""
evaluate.py — Shared evaluation script for all ARL encoder variants.

Works for every encoder: unet, vanilla_vae, beta_vae, residual_vae,
cvae, factor_vae, vq_vae. Usable by all three team members.

Metrics computed:
  - Utility Accuracy  (smile detection %, want HIGH)
  - Privacy Accuracy  (gender classification %, want ~50%)
  - AUC-ROC           (both tasks, better than accuracy alone)
  - F1 Score          (utility)
  - Confusion Matrix  (both tasks)
  - NAG               (Normalised Accuracy Gap — combined score)
  - Privacy at K% Utility (privacy accuracy when utility is constrained to K%)

Usage (single model):
  python evaluate.py \\
    --encoder vanilla_vae \\
    --exp_name member1_vanillavae_10e_20k \\
    --checkpoint_dir /scratch/$USER/pp_ext_member2/arls \\
    --data_source huggingface \\
    --hf_cache_dir /scratch/$USER/pp_ext_member2/hf_cache

Usage (compare all models at once):
  python evaluate.py \\
    --encoder vanilla_vae,beta_vae,residual_vae \\
    --exp_name member1_vanillavae_10e_20k,member1_betavae_10e_20k,member1_residualvae_10e_20k \\
    --checkpoint_dir /scratch/$USER/pp_ext_member2/arls \\
    --data_source huggingface \\
    --hf_cache_dir /scratch/$USER/pp_ext_member2/hf_cache

Outputs:
  - Printed metric table
  - eval_<exp_name>.json  (full results per model)
  - eval_summary.csv      (one row per model, appended)

Requirements:
  pip install scikit-learn
"""

import os
import json
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from models.get_encoder import get_encoder
from models.cifar_like.resnet import ResNet18

try:
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    print("WARNING: scikit-learn not found. Run: pip install scikit-learn")
    HAS_SKLEARN = False

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# CelebA attribute indices (same as adversarial_training.py)
U_TASK = 31   # Smiling
P_TASK = 20   # Male
CELEBA_ATTRS = [
    '5_o_Clock_Shadow','Arched_Eyebrows','Attractive','Bags_Under_Eyes','Bald',
    'Bangs','Big_Lips','Big_Nose','Black_Hair','Blond_Hair','Blurry','Brown_Hair',
    'Bushy_Eyebrows','Chubby','Double_Chin','Eyeglasses','Goatee','Gray_Hair',
    'Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open','Mustache',
    'Narrow_Eyes','No_Beard','Oval_Face','Pale_Skin','Pointy_Nose',
    'Receding_Hairline','Rosy_Cheeks','Sideburns','Smiling','Straight_Hair',
    'Wavy_Hair','Wearing_Earrings','Wearing_Hat','Wearing_Lipstick',
    'Wearing_Necklace','Wearing_Necktie','Young',
]

# Default K values for Privacy at K% Utility
DEFAULT_K_LEVELS = [70, 75, 80, 85]


# ── Dataset ──────────────────────────────────────────────────────────────────

class CelebAHFDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform, cache_dir=None):
        from datasets import load_dataset
        self.data = load_dataset("flwrlabs/celeba", split=split, cache_dir=cache_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data[i]
        img = row['image']
        if hasattr(img, 'convert'):
            img = img.convert('RGB')
        img = self.transform(img)
        attrs = torch.tensor(
            [1.0 if row.get(k, False) else 0.0 for k in CELEBA_ATTRS],
            dtype=torch.float32,
        )
        return img, attrs


def build_test_loader(args):
    tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    if args.data_source == 'huggingface':
        ds = CelebAHFDataset('test', tfm, cache_dir=args.hf_cache_dir)
    else:
        import torchvision.datasets as dsets
        ds = dsets.CelebA(root=args.data_dir, split='test',
                          target_type='attr', transform=tfm, download=False)
    if args.max_test_samples and args.max_test_samples < len(ds):
        ds = Subset(ds, list(range(args.max_test_samples)))
    return DataLoader(ds, batch_size=args.batch_size,
                      shuffle=False, num_workers=args.num_workers, pin_memory=True)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_nag(utility_acc, privacy_acc):
    """
    Normalised Accuracy Gap (NAG).

    NAG >> 1 : lots of utility kept, adversary near random  → good
    NAG = 1  : utility gain equals privacy leak             → neutral
    NAG < 1  : more privacy leaked than utility kept        → poor
    """
    u_gain = max(utility_acc - 0.5, 0.0)
    p_leak = max(privacy_acc - 0.5, 1e-6)
    return round(u_gain / p_leak, 4)


def compute_privacy_at_k_utility(labels_u, probs_u, labels_p, probs_p, k_levels):
    """
    For each K in k_levels, find the classifier threshold at which utility
    accuracy is >= K%, then report what the adversary's privacy accuracy is
    at that same threshold.

    Returns a dict: {k: {"threshold": t, "utility_acc": u, "privacy_acc": p}}
    If a model cannot reach K% utility at any threshold, that K is marked None.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    results = {}

    for k in k_levels:
        best = None
        # sweep thresholds to find ones that achieve >= k% utility accuracy
        # among those, pick the one with lowest privacy accuracy (most private)
        for t in thresholds:
            preds_u = (probs_u >= t).astype(float)
            preds_p = (probs_p >= t).astype(float)
            u_acc = float((preds_u == labels_u).mean() * 100)
            p_acc = float((preds_p == labels_p).mean() * 100)

            if u_acc >= k:
                if best is None or p_acc < best["privacy_acc"]:
                    best = {
                        "threshold":   round(float(t), 3),
                        "utility_acc": round(u_acc, 2),
                        "privacy_acc": round(p_acc, 2),
                    }

        results[k] = best  # None if model can never reach k% utility

    return results


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(encoder_name, exp_name, checkpoint_dir, device):
    """Load encoder, classifier, and adversary from checkpoints."""
    encoder = get_encoder(encoder_name, img_size=224).to(device)
    clf = ResNet18(); clf.linear = nn.Linear(512, 1); clf = clf.to(device)
    adv = ResNet18(); adv.linear = nn.Linear(512, 1); adv = adv.to(device)

    for tag, model in [("encoder", encoder), ("clf", clf), ("adv", adv)]:
        path = os.path.join(checkpoint_dir, f"{tag}_model_{exp_name}.pt")
        if not os.path.exists(path):
            print(f"  ERROR: checkpoint not found: {path}")
            return None, None, None
        model.load_state_dict(torch.load(path, map_location=device))

    encoder.eval(); clf.eval(); adv.eval()
    return encoder, clf, adv


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(encoder, clf, adv, test_loader, encoder_name, device):
    """Run forward pass and collect labels and predicted probabilities."""
    labels_u, labels_p = [], []
    probs_u,  probs_p  = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            lu = targets[:, U_TASK].float()
            lp = targets[:, P_TASK].float()

            recon = encoder(inputs, lu.to(device)) if encoder_name == 'cvae' \
                    else encoder(inputs)

            pu = torch.sigmoid(clf(recon).flatten()).cpu()
            pp = torch.sigmoid(adv(recon).flatten()).cpu()

            labels_u.extend(lu.numpy())
            labels_p.extend(lp.numpy())
            probs_u.extend(pu.numpy())
            probs_p.extend(pp.numpy())

    return (np.array(labels_u), np.array(labels_p),
            np.array(probs_u),  np.array(probs_p))


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_metrics(labels_u, labels_p, probs_u, probs_p, k_levels):
    """Compute all metrics from labels and predicted probabilities."""
    preds_u = (probs_u >= 0.5).astype(float)
    preds_p = (probs_p >= 0.5).astype(float)

    util_acc = float((preds_u == labels_u).mean() * 100)
    priv_acc = float((preds_p == labels_p).mean() * 100)
    nag      = compute_nag(util_acc / 100, priv_acc / 100)

    util_auc = float(roc_auc_score(labels_u, probs_u)) if HAS_SKLEARN else None
    priv_auc = float(roc_auc_score(labels_p, probs_p)) if HAS_SKLEARN else None
    util_f1  = float(f1_score(labels_u, preds_u, zero_division=0)) if HAS_SKLEARN else None
    cm_u     = confusion_matrix(labels_u, preds_u).tolist() if HAS_SKLEARN else None
    cm_p     = confusion_matrix(labels_p, preds_p).tolist() if HAS_SKLEARN else None

    pak = compute_privacy_at_k_utility(labels_u, probs_u, labels_p, probs_p, k_levels)

    return {
        "util_acc": util_acc, "priv_acc": priv_acc, "nag": nag,
        "util_auc": util_auc, "priv_auc": priv_auc, "util_f1": util_f1,
        "cm_u": cm_u, "cm_p": cm_p, "pak": pak,
    }


# ── Printing ──────────────────────────────────────────────────────────────────

def print_results(encoder_name, exp_name, n, m):
    print(f"\n{'='*62}")
    print(f"  Encoder : {encoder_name}")
    print(f"  Run     : {exp_name}")
    print(f"{'='*62}")
    print(f"\n  Samples : {n}")
    print(f"\n  {'Metric':<26} {'Utility (Smile)':>18}  {'Privacy (Gender)':>18}")
    print(f"  {'-'*64}")
    print(f"  {'Accuracy':<26} {m['util_acc']:>17.2f}%  {m['priv_acc']:>17.2f}%")
    if HAS_SKLEARN:
        print(f"  {'AUC-ROC':<26} {m['util_auc']:>18.4f}  {m['priv_auc']:>18.4f}")
        print(f"  {'F1 Score':<26} {m['util_f1']:>18.4f}  {'—':>18}")
    print(f"  {'NAG (combined)':<26} {m['nag']:>38.4f}")

    if   m['nag'] > 2.0: note = "Good — utility well preserved relative to privacy leakage"
    elif m['nag'] > 1.0: note = "Acceptable — utility gain exceeds privacy leak"
    else:                 note = "Needs improvement — privacy leakage exceeds utility preservation"
    print(f"\n  NAG note : {note}")

    # Privacy at K% Utility table
    print(f"\n  Privacy at K% Utility")
    print(f"  {'K% Utility':>12}  {'Threshold':>10}  {'Actual Util%':>13}  {'Privacy%':>10}  {'Note':>20}")
    print(f"  {'-'*72}")
    for k, res in m['pak'].items():
        if res is None:
            print(f"  {k:>11}%  {'—':>10}  {'—':>13}  {'—':>10}  {'Model cannot reach K%':>20}")
        else:
            # lower privacy acc = better (closer to random 50%)
            gap  = res['privacy_acc'] - 50.0
            note = "good" if gap < 5 else "acceptable" if gap < 15 else "leaking"
            print(f"  {k:>11}%  {res['threshold']:>10.3f}"
                  f"  {res['utility_acc']:>12.2f}%"
                  f"  {res['privacy_acc']:>9.2f}%"
                  f"  {note:>20}")

    if HAS_SKLEARN and m['cm_u']:
        print(f"\n  Confusion Matrix — Utility (Smile):")
        print(f"    Pred→  Not Smiling   Smiling")
        print(f"    Not Smiling  {m['cm_u'][0][0]:6d}   {m['cm_u'][0][1]:6d}")
        print(f"    Smiling      {m['cm_u'][1][0]:6d}   {m['cm_u'][1][1]:6d}")
        print(f"\n  Confusion Matrix — Privacy (Gender)  [low TP = good privacy]:")
        print(f"    Pred→  Female   Male")
        print(f"    Female {m['cm_p'][0][0]:6d}   {m['cm_p'][0][1]:6d}")
        print(f"    Male   {m['cm_p'][1][0]:6d}   {m['cm_p'][1][1]:6d}")


# ── Core evaluation ──────────────────────────────────────────────────────────

def evaluate_one(encoder_name, exp_name, checkpoint_dir, test_loader, device, k_levels):
    encoder, clf, adv = load_models(encoder_name, exp_name, checkpoint_dir, device)
    if encoder is None:
        return None

    labels_u, labels_p, probs_u, probs_p = run_inference(
        encoder, clf, adv, test_loader, encoder_name, device)

    m = compute_metrics(labels_u, labels_p, probs_u, probs_p, k_levels)
    n = len(labels_u)

    print_results(encoder_name, exp_name, n, m)

    # flatten pak for JSON serialisation
    pak_serialisable = {str(k): v for k, v in m['pak'].items()}

    return {
        "encoder":     encoder_name,
        "exp_name":    exp_name,
        "n_test":      n,
        "utility_acc": round(m['util_acc'], 2),
        "privacy_acc": round(m['priv_acc'], 2),
        "utility_auc": round(m['util_auc'], 4) if m['util_auc'] is not None else "N/A",
        "privacy_auc": round(m['priv_auc'], 4) if m['priv_auc'] is not None else "N/A",
        "utility_f1":  round(m['util_f1'],  4) if m['util_f1']  is not None else "N/A",
        "nag":         m['nag'],
        "cm_utility":  m['cm_u'],
        "cm_privacy":  m['cm_p'],
        "privacy_at_k_utility": pak_serialisable,
    }


def save_results(r, checkpoint_dir):
    json_path = os.path.join(checkpoint_dir, f"eval_{r['exp_name']}.json")
    with open(json_path, 'w') as f:
        json.dump(r, f, indent=2)
    print(f"\n  Saved → {json_path}")

    csv_path = os.path.join(checkpoint_dir, "eval_summary.csv")
    fields   = ["encoder", "exp_name", "n_test", "utility_acc", "privacy_acc",
                 "utility_auc", "privacy_auc", "utility_f1", "nag"]
    # add one column per K level
    for k in r["privacy_at_k_utility"]:
        fields.append(f"priv_at_{k}pct_util")

    # read existing rows to avoid duplicates
    existing = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            existing = list(csv.DictReader(f))
    existing = [row for row in existing if row.get("exp_name") != r["exp_name"]]

    # build new row
    new_row = {k: r[k] for k in ["encoder", "exp_name", "n_test",
                                   "utility_acc", "privacy_acc",
                                   "utility_auc", "privacy_auc",
                                   "utility_f1",  "nag"]}
    for k, res in r["privacy_at_k_utility"].items():
        new_row[f"priv_at_{k}pct_util"] = res["privacy_acc"] if res else "N/A"

    existing.append(new_row)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(existing)
    print(f"  Saved → {csv_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate ARL encoder checkpoints — works for all variants"
    )
    p.add_argument('--encoder',          type=str, required=True)
    p.add_argument('--exp_name',         type=str, required=True)
    p.add_argument('--checkpoint_dir',   type=str, required=True)
    p.add_argument('--data_source',      type=str, default='huggingface',
                   choices=['huggingface', 'local'])
    p.add_argument('--hf_cache_dir',     type=str, default=None)
    p.add_argument('--data_dir',         type=str, default=None)
    p.add_argument('--max_test_samples', type=int, default=None)
    p.add_argument('--batch_size',       type=int, default=32)
    p.add_argument('--num_workers',      type=int, default=1)
    p.add_argument('--device',           type=str, default='cuda')
    p.add_argument('--k_levels',         type=str, default='70,75,80,85',
                   help='Comma-separated utility % thresholds for Privacy at K metric')
    args = p.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    k_levels = [int(k.strip()) for k in args.k_levels.split(',')]
    encoders = [e.strip() for e in args.encoder.split(',')]
    names    = [e.strip() for e in args.exp_name.split(',')]

    if len(encoders) != len(names):
        raise ValueError(f"--encoder ({len(encoders)}) and --exp_name ({len(names)}) "
                         f"must have the same number of entries")

    print(f"\nDevice : {device}")
    print(f"K levels for Privacy@K : {k_levels}")
    print(f"Loading test data...")
    loader = build_test_loader(args)
    print(f"Ready.\n")

    results = []
    for enc, name in zip(encoders, names):
        r = evaluate_one(enc, name, args.checkpoint_dir, loader, device, k_levels)
        if r:
            save_results(r, args.checkpoint_dir)
            results.append(r)

    if len(results) > 1:
        print(f"\n\n{'='*80}")
        print(f"  COMPARISON TABLE — {len(results)} models evaluated")
        print(f"{'='*80}")
        hdr = f"  {'Encoder':<16} {'Run':<32} {'Util%':>6} {'Priv%':>6} {'U-AUC':>7} {'P-AUC':>7} {'NAG':>7}"
        for k in k_levels:
            hdr += f"  {'P@'+str(k)+'%':>8}"
        print(hdr)
        print(f"  {'-'*78}")
        for r in results:
            row = (f"  {r['encoder']:<16} {r['exp_name']:<32} "
                   f"{r['utility_acc']:>6.1f} {r['privacy_acc']:>6.1f} "
                   f"{str(r['utility_auc']):>7} {str(r['privacy_auc']):>7} "
                   f"{r['nag']:>7.3f}")
            for k in k_levels:
                res = r["privacy_at_k_utility"].get(str(k))
                val = f"{res['privacy_acc']:.1f}%" if res else "N/A"
                row += f"  {val:>8}"
            print(row)

        best_util = max(results, key=lambda x: x['utility_acc'])
        best_priv = min(results, key=lambda x: x['privacy_acc'])
        best_nag  = max(results, key=lambda x: x['nag'])
        print(f"\n  Best utility  : {best_util['encoder']} ({best_util['utility_acc']:.2f}%)")
        print(f"  Best privacy  : {best_priv['encoder']} ({best_priv['privacy_acc']:.2f}%)")
        print(f"  Best NAG      : {best_nag['encoder']}  ({best_nag['nag']:.4f})")

        # best Privacy@K per level
        print(f"\n  Best Privacy@K (lowest privacy acc = most private):")
        for k in k_levels:
            valid = [r for r in results if r["privacy_at_k_utility"].get(str(k)) is not None]
            if valid:
                best = min(valid, key=lambda x: x["privacy_at_k_utility"][str(k)]["privacy_acc"])
                val  = best["privacy_at_k_utility"][str(k)]["privacy_acc"]
                print(f"    K={k}%  →  {best['encoder']} ({val:.2f}% privacy acc)")
            else:
                print(f"    K={k}%  →  No model reached this utility level")
        print(f"{'='*80}\n")