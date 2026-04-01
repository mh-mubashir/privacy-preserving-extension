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

    Measures utility preserved vs privacy leaked, both normalised against
    the random-chance baseline (50% for binary classification).

        NAG = (utility_acc - 0.5) / max(privacy_acc - 0.5, 1e-6)

    NAG >> 1 : lots of utility kept, adversary near random  → good
    NAG = 1  : utility gain equals privacy leak             → neutral
    NAG < 1  : more privacy leaked than utility kept        → poor
    """
    u_gain = max(utility_acc - 0.5, 0.0)
    p_leak = max(privacy_acc - 0.5, 1e-6)
    return round(u_gain / p_leak, 4)


# ── Core evaluation ──────────────────────────────────────────────────────────

def evaluate_one(encoder_name, exp_name, checkpoint_dir, test_loader, device):
    print(f"\n{'='*62}")
    print(f"  Encoder : {encoder_name}")
    print(f"  Run     : {exp_name}")
    print(f"{'='*62}")

    # Build models
    encoder = get_encoder(encoder_name, img_size=224).to(device)
    clf = ResNet18(); clf.linear = nn.Linear(512, 1); clf = clf.to(device)
    adv = ResNet18(); adv.linear = nn.Linear(512, 1); adv = adv.to(device)

    # Load checkpoints
    for tag, model in [("encoder", encoder), ("clf", clf), ("adv", adv)]:
        path = os.path.join(checkpoint_dir, f"{tag}_model_{exp_name}.pt")
        if not os.path.exists(path):
            print(f"  ERROR: checkpoint not found: {path}")
            return None
        model.load_state_dict(torch.load(path, map_location=device))

    encoder.eval(); clf.eval(); adv.eval()

    labels_u_all, labels_p_all = [], []
    probs_u_all,  probs_p_all  = [], []
    preds_u_all,  preds_p_all  = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            lu = targets[:, U_TASK].float()
            lp = targets[:, P_TASK].float()

            if encoder_name == 'cvae':
                recon = encoder(inputs, lu.to(device))
            else:
                recon = encoder(inputs)

            pu = torch.sigmoid(clf(recon).flatten()).cpu()
            pp = torch.sigmoid(adv(recon).flatten()).cpu()

            labels_u_all.extend(lu.numpy())
            labels_p_all.extend(lp.numpy())
            probs_u_all.extend(pu.numpy())
            probs_p_all.extend(pp.numpy())
            preds_u_all.extend((pu > 0.5).float().numpy())
            preds_p_all.extend((pp > 0.5).float().numpy())

    lu = np.array(labels_u_all); lp = np.array(labels_p_all)
    pu = np.array(probs_u_all);  pp = np.array(probs_p_all)
    du = np.array(preds_u_all);  dp = np.array(preds_p_all)
    n  = len(lu)

    # Core metrics
    util_acc = float((du == lu).mean() * 100)
    priv_acc = float((dp == lp).mean() * 100)
    nag      = compute_nag(util_acc / 100, priv_acc / 100)

    util_auc = float(roc_auc_score(lu, pu)) if HAS_SKLEARN else None
    priv_auc = float(roc_auc_score(lp, pp)) if HAS_SKLEARN else None
    util_f1  = float(f1_score(lu, du, zero_division=0)) if HAS_SKLEARN else None
    cm_u = confusion_matrix(lu, du).tolist() if HAS_SKLEARN else None
    cm_p = confusion_matrix(lp, dp).tolist() if HAS_SKLEARN else None

    # Print
    print(f"\n  Samples : {n}")
    print(f"\n  {'Metric':<26} {'Utility (Smile)':>18}  {'Privacy (Gender)':>18}")
    print(f"  {'-'*64}")
    print(f"  {'Accuracy':<26} {util_acc:>17.2f}%  {priv_acc:>17.2f}%")
    if HAS_SKLEARN:
        print(f"  {'AUC-ROC':<26} {util_auc:>18.4f}  {priv_auc:>18.4f}")
        print(f"  {'F1 Score':<26} {util_f1:>18.4f}  {'—':>18}")
    print(f"  {'NAG (combined)':<26} {nag:>38.4f}")

    if nag > 2.0:
        note = "Good — utility well preserved relative to privacy leakage"
    elif nag > 1.0:
        note = "Acceptable — utility gain exceeds privacy leak"
    else:
        note = "Needs improvement — privacy leakage exceeds utility preservation"
    print(f"\n  NAG note : {note}")

    if HAS_SKLEARN and cm_u:
        print(f"\n  Confusion Matrix — Utility (Smile):")
        print(f"    Pred→  Not Smiling   Smiling")
        print(f"    Not Smiling  {cm_u[0][0]:6d}   {cm_u[0][1]:6d}")
        print(f"    Smiling      {cm_u[1][0]:6d}   {cm_u[1][1]:6d}")
        print(f"\n  Confusion Matrix — Privacy (Gender)  [low TP = good privacy]:")
        print(f"    Pred→  Female   Male")
        print(f"    Female {cm_p[0][0]:6d}   {cm_p[0][1]:6d}")
        print(f"    Male   {cm_p[1][0]:6d}   {cm_p[1][1]:6d}")

    return {
        "encoder":     encoder_name,
        "exp_name":    exp_name,
        "n_test":      n,
        "utility_acc": round(util_acc, 2),
        "privacy_acc": round(priv_acc, 2),
        "utility_auc": round(util_auc, 4) if util_auc is not None else "N/A",
        "privacy_auc": round(priv_auc, 4) if priv_auc is not None else "N/A",
        "utility_f1":  round(util_f1,  4) if util_f1  is not None else "N/A",
        "nag":         nag,
        "cm_utility":  cm_u,
        "cm_privacy":  cm_p,
    }


def save_results(r, checkpoint_dir):
    json_path = os.path.join(checkpoint_dir, f"eval_{r['exp_name']}.json")
    with open(json_path, 'w') as f:
        json.dump(r, f, indent=2)
    print(f"\n  Saved → {json_path}")

    csv_path   = os.path.join(checkpoint_dir, "eval_summary.csv")
    new_file   = not os.path.exists(csv_path)
    fields     = ["encoder","exp_name","n_test","utility_acc","privacy_acc",
                  "utility_auc","privacy_auc","utility_f1","nag"]
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow({k: r[k] for k in fields})
    print(f"  Appended → {csv_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate ARL encoder checkpoints — works for all variants"
    )
    p.add_argument('--encoder',          type=str, required=True,
                   help='Encoder name(s), comma-separated')
    p.add_argument('--exp_name',         type=str, required=True,
                   help='Experiment name(s) matching checkpoint filenames, comma-separated')
    p.add_argument('--checkpoint_dir',   type=str, required=True,
                   help='Directory containing encoder_model_<exp_name>.pt etc.')
    p.add_argument('--data_source',      type=str, default='huggingface',
                   choices=['huggingface', 'local'])
    p.add_argument('--hf_cache_dir',     type=str, default=None)
    p.add_argument('--data_dir',         type=str, default=None)
    p.add_argument('--max_test_samples', type=int, default=None,
                   help='Cap test samples (default: full test set ~20k)')
    p.add_argument('--batch_size',       type=int, default=32)
    p.add_argument('--num_workers',      type=int, default=1)
    p.add_argument('--device',           type=str, default='cuda')
    args = p.parse_args()

    device   = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    encoders = [e.strip() for e in args.encoder.split(',')]
    names    = [e.strip() for e in args.exp_name.split(',')]

    if len(encoders) != len(names):
        raise ValueError(f"--encoder ({len(encoders)}) and --exp_name ({len(names)}) "
                         f"must have the same number of entries")

    print(f"\nDevice : {device}")
    print(f"Loading test data...")
    loader = build_test_loader(args)
    print(f"Ready.\n")

    results = []
    for enc, name in zip(encoders, names):
        r = evaluate_one(enc, name, args.checkpoint_dir, loader, device)
        if r:
            save_results(r, args.checkpoint_dir)
            results.append(r)

    if len(results) > 1:
        print(f"\n\n{'='*80}")
        print(f"  COMPARISON TABLE — {len(results)} models evaluated")
        print(f"{'='*80}")
        hdr = f"  {'Encoder':<16} {'Run':<32} {'Util%':>6} {'Priv%':>6} {'U-AUC':>7} {'P-AUC':>7} {'NAG':>7}"
        print(hdr)
        print(f"  {'-'*78}")
        for r in results:
            print(f"  {r['encoder']:<16} {r['exp_name']:<32} "
                  f"{r['utility_acc']:>6.1f} {r['privacy_acc']:>6.1f} "
                  f"{str(r['utility_auc']):>7} {str(r['privacy_auc']):>7} "
                  f"{r['nag']:>7.3f}")

        best_util = max(results, key=lambda x: x['utility_acc'])
        best_priv = min(results, key=lambda x: x['privacy_acc'])
        best_nag  = max(results, key=lambda x: x['nag'])
        print(f"\n  Best utility  : {best_util['encoder']} ({best_util['utility_acc']:.2f}%)")
        print(f"  Best privacy  : {best_priv['encoder']} ({best_priv['privacy_acc']:.2f}%)")
        print(f"  Best NAG      : {best_nag['encoder']}  ({best_nag['nag']:.4f})")
        print(f"{'='*80}\n")
