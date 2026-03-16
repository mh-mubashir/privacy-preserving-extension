# Privacy-Preserving Adversarial Training

This project implements **adversarial training** for learning an image encoder that preserves a **utility attribute** (e.g., smile) while hiding a **private attribute** (e.g., gender) from a downstream adversary. The encoder is a UNet; a ResNet **utility classifier** is trained to predict the utility attribute, and a ResNet **adversary** is trained to predict the private attribute. The encoder is trained to preserve utility (low classifier loss) and to fool the adversary (high adversary loss).

In addition to the UNet-based encoder used inside the adversarial loop, we also train **stand‑alone disentangled VAEs on CelebA** using the [`pythae`](https://github.com/clementchadebec/Benchmark_VAE) library:

- `DisentangledBetaVAE`
- `BetaTCVAE`
- `FactorVAE`

These VAE models are trained separately (outside the ARL loop) and can be used as disentangled feature extractors or as baselines for comparison. See [docs/Pythae_VAE_Training_CelebA.md](docs/Pythae_VAE_Training_CelebA.md) for full details and results.

> **Documentation:** See [docs/PROJECT_PROPOSAL.md](docs/PROJECT_PROPOSAL.md) for the team's project proposal—planned VAE encoder variants (Vanilla VAE, β-VAE, CVAE, Factor VAE, VQ-VAE), modular implementation strategy, and evaluation plan.

---

## Table of Contents

- [Data Preparation](#data-preparation)
- [Environment](#environment)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Outputs](#outputs)
- [Command-Line Reference](#command-line-reference)

---

## Data Preparation

### 1. CelebA Dataset

Training uses the [CelebA (Large-scale CelebFaces Attributes)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset. Each image has 40 binary attribute labels. You will use one attribute as **utility** (to preserve) and another as **private** (to hide from the adversary).

**Option A: Automatic download via PyTorch**

If you pass `download=True` to `torchvision.datasets.CelebA`, the dataset will be downloaded (requires `gdown` for Google Drive). The script in this repo does **not** set `download=True` by default, so you must prepare the data manually or add it for first-time setup.

**Option B: Manual download**

1. Create a root directory for the dataset, e.g. `/projects/xz-group/datasets/`.
2. Download the CelebA files into a subfolder named `celeba`:
   - **Images**: `img_align_celeba.zip` (align-cropped faces). Extract so that images lie under `celeba/img_align_celeba/`.
   - **Attribute list**: `list_attr_celeba.txt` (one header line + one line per image with 40 binary attributes).
   - **Split**: `list_eval_partition.txt` (train/valid/test assignment per image).
   - **Optional** (required by `torchvision.datasets.CelebA` for full integrity): `identity_CelebA.txt`, `list_bbox_celeba.txt`, `list_landmarks_align_celeba.txt`.

3. Final layout:

```text
<data_dir>/
└── celeba/
    ├── img_align_celeba/
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    ├── list_attr_celeba.txt
    ├── list_eval_partition.txt
    ├── identity_CelebA.txt
    ├── list_bbox_celeba.txt
    └── list_landmarks_align_celeba.txt
```

4. Set `--data_dir` to `<data_dir>` (e.g. `/projects/xz-group/datasets/`) when running the script.

### 2. Attribute Indices

CelebA’s `list_attr_celeba.txt` has a header line with 40 attribute names; the columns are in a fixed order. The script uses **0-based indices** into this attribute vector:

- **`u_task`** (default: `31`): utility attribute — the downstream task we want to keep (e.g. **Smiling**). The frozen classifier is trained to predict this; the encoder is trained to keep this predictable.
- **`p_task`** (default: `20`): private attribute — the one we want to hide (e.g. **Male**). The adversary tries to predict this; the encoder is trained to make this hard to predict.

You can change these in the script or extend the parser to pass them as arguments. The exact mapping depends on the order in `list_attr_celeba.txt` (see the header line or [CelebA documentation](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)).

### 3. Data Splits and Subsets

The script uses CelebA’s built-in `train` / `valid` / `test` splits, then takes subsets for speed/memory:

- **Train**: first 60,000 samples  
- **Valid**: first 10,000 samples  
- **Test**: first 10,000 samples  

You can change the `Subset` ranges in `adversarial_training.py` if you want to use the full splits or different sizes.

### 4. Transforms

- **Training**: Resize to 224×224, random crop (224 with padding 4), random horizontal flip, then `ToTensor()` (values in [0, 1]). No grayscale; images are **3-channel RGB** for the UNet and ResNets.
- **Validation / test**: Resize to 224×224 and `ToTensor()` only.

---

## Environment

- **Python**: 3.9+ (tested with 3.12)
- **PyTorch** and **torchvision** (with CUDA if you use GPU)
- **NumPy**
- **wandb** (optional, for logging with `--use_wandb`)
- **gdown** (optional, for CelebA download via `torchvision`)

**Lenovo Legion (NVIDIA GPU):** If you use a Legion laptop, see [docs/GPU_SETUP_LEGION.md](docs/GPU_SETUP_LEGION.md) for the exact GPU setup used on a Legion 7 with an RTX 5080 (including the working Python 3.12 + CUDA 12.8 environment).

### Setup

**Using requirements.txt:**

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) and replace the `torch`/`torchvision` lines in `requirements.txt` accordingly.

On the Legion 7 + RTX 5080 used for this project, a dedicated env **`.venv312_cu128`** with Python 3.12 and CUDA 12.8 wheels was created via:

```bash
.\.venv312_cu128\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

See `docs/GPU_SETUP_LEGION.md` for details on that environment and how it was verified (`torch.cuda.is_available()` and a small CUDA matmul).

**Optional packages:**

```bash
pip install wandb      # Weights & Biases logging
pip install gdown      # CelebA download via torchvision
pip install torchprofile   # MACs/FLOPs profiling (for UNet __main__)
pip install pythae     # Extra VAE variants (DisentangledBetaVAE, BetaTCVAE, FactorVAE, etc.)
```

---

## Model Architecture

1. **Encoder (trainable)**  
   - **UNet** with 3 input channels and 3 output channels, `size='tiny'`.  
   - Input: RGB image `(B, 3, 224, 224)`.  
   - Output: same spatial size, 3 channels, values in [0, 1] (clamped).  
   - Acts as a learned “obfuscation” that preserves utility and suppresses the private attribute.

2. **Utility classifier (trainable)**  
   - **ResNet-18** (from `models/cifar_like/resnet.py`), final layer replaced with `nn.Linear(512, 1)`.  
   - Input: encoder output (3-channel image).  
   - Trained together with the encoder to minimize the **utility loss** (BCE on the utility attribute). Updated with the same objective as the encoder: minimize `loss_clf - lambda_clf * loss_adv`.

3. **Adversary (trainable)**  
   - Same ResNet-18 with `nn.Linear(512, 1)`.  
   - Tries to predict the **private** attribute from the encoder output.  
   - Trained to minimize BCE on the private attribute; the encoder (and classifier) are trained to maximize this loss (so the private attribute becomes hard to predict).

ResNet uses `F.adaptive_avg_pool2d(out, 1)` so it works for 224×224 inputs.

---

## Training

**Objective**

- **Adversary**: minimize `loss_adv` (BCE for private attribute).
- **Encoder**: minimize `loss_clf - lambda_clf * loss_adv`, i.e. preserve utility (low `loss_clf`) and fool the adversary (high `loss_adv`).

Each step:

1. Forward: image → encoder → blurred image → utility classifier (logits_u) and adversary (adv_logits).
2. Compute `loss_clf` (utility) and `loss_adv` (private).
3. Update adversary: `loss_adv.backward(retain_graph=True)`, then `optimizer_adv.step()`.
4. Update encoder and utility classifier: `(loss_clf - lambda_clf * loss_adv).backward()`, then `optimizer_enc.step()` and `optimizer_clf.step()`.

Learning rates use **CosineAnnealingLR** over the number of epochs for encoder, classifier, and adversary.

---

## Outputs

- **Checkpoints** (in the current working directory):
  - `encoder_model_<exp_name>.pt` — encoder state dict.
  - `clf_model_<exp_name>.pt` — utility classifier state dict.
  - `adv_model_<exp_name>.pt` — adversary state dict.

- **Console**: per-epoch validation accuracy for utility and adversary; final test accuracy for both.

- **Weights & Biases**: if `--use_wandb` is set, training/validation images, losses, and accuracies are logged (see script for exact keys).

---

## Command-Line Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch_size` | int | 64 | Batch size for train/val/test loaders. |
| `--num_epochs` | int | 50 | Number of training epochs. |
| `--learning_rate_enc` | float | 0.001 | Adam learning rate for the encoder. |
| `--learning_rate_clf` | float | 0.001 | Adam learning rate for the utility classifier. |
| `--learning_rate_adv` | float | 0.001 | Adam learning rate for the adversary. |
| `--encoder` | str | `unet` | Encoder architecture: `unet`, `cvae`, or `factor_vae`. |
| `--vae_weight` | float | 0.1 | Weight for VAE (recon + KL) loss in ARL when using CVAE/Factor VAE. |
| `--vae_beta` | float | 1.0 | Beta for KL weight in VAE loss. |
| `--vae_gamma` | float | 10.0 | Gamma for Factor VAE total correlation term. |
| `--device` | str | `"cuda"` | Device (e.g. `cuda`, `cpu`). |
| `--data_dir` | str |  | Root directory containing the `celeba` folder. |
| `--seed` | int | 42 | Random seed for reproducibility. |
| `--use_wandb` | flag | False | Enable Weights & Biases logging. |
| `--lambda_clf` | float | 1.0 | Weight for utility vs privacy: encoder minimizes `loss_clf - lambda_clf * loss_adv`. |
| `--exp_name` | str | `celeb` | Experiment name; used in checkpoint filenames and wandb run name. |

**Examples**

Baseline (UNet encoder):

```bash
python adversarial_training.py --data_dir /projects/xz-group/datasets/ --exp_name my_run --num_epochs 50
```

CVAE encoder (conditions on utility label):

```bash
python adversarial_training.py --data_dir /path/to/datasets --encoder cvae --exp_name cvae_run --vae_weight 0.1 --vae_beta 1.0
```

Factor VAE encoder (disentanglement via total correlation):

```bash
python adversarial_training.py --data_dir /path/to/datasets --encoder factor_vae --exp_name factor_vae_run --vae_weight 0.1 --vae_gamma 10.0
```

Pythae VAE variants (trained separately on CelebA, using the `pythae_training.py` script):

```bash
# DisentangledBetaVAE
python pythae_training.py --variant disentangled_betavae --data_source huggingface --hf_cache_dir ./hf_cache --img_size 64 --latent_dim 32 --beta 4.0 --batch_size 48 --num_epochs 50

# BetaTCVAE
python pythae_training.py --variant betatcvae --data_source huggingface --hf_cache_dir ./hf_cache --img_size 64 --latent_dim 32 --beta 2.0 --gamma 5.0 --learning_rate 5e-4 --batch_size 32 --num_epochs 50

# FactorVAE (adversarial trainer)
python pythae_training.py --variant factorvae --data_source huggingface --hf_cache_dir ./hf_cache --img_size 64 --latent_dim 32 --gamma 10.0 --learning_rate 5e-4 --batch_size 32 --num_epochs 50
```

The exact hyperparameters above correspond to the stable runs documented in [docs/Pythae_VAE_Training_CelebA.md](docs/Pythae_VAE_Training_CelebA.md); you can adjust batch size and learning rate based on your GPU memory and desired trade‑offs.

With W&B:

```bash
python adversarial_training.py --data_dir /path/to/datasets --use_wandb --exp_name celeb_smile_gender
```

---

## Summary

- **Data**: CelebA under `<data_dir>/celeba/` with standard files and 3-channel RGB images, 224×224.
- **Training**: The **encoder** (UNet), **utility classifier** (ResNet), and **adversary** (ResNet) are all trained. The encoder and classifier minimize utility loss and maximize adversary loss; the adversary minimizes its own prediction loss.
- **Goal**: Encoder outputs that keep the utility attribute predictable while making the private attribute hard to predict by the adversary.
