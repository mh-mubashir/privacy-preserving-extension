# Privacy-Preserving Edge Vision via Adversarial Representation Learning

A systematic study of VAE encoder architectures for privacy-preserving face analysis on CelebA. The system trains an encoder that reliably detects smiling while preventing downstream models from inferring gender from the same representation.

**Task mapping (CelebA attributes):**
- Utility: attribute 31 — Smiling (want high accuracy)
- Privacy: attribute 20 — Male (want accuracy near 50%, i.e. random chance)

---

## Repository Structure

```
.
├── adversarial_training.py   # Main ARL training script
├── evaluate.py               # Evaluation: accuracy, AUC, NAG, Privacy@K
├── pythae_training.py        # Standalone Pythae VAE training (prototyping)
├── export_onnx.py            # Export all encoders to ONNX
├── compute_flops.py          # MACs / parameter count via thop
├── benchmark_latency.py      # CPU + Jetson latency via ONNX Runtime
├── make_charts.py            # Generate result charts for the report
├── models/
│   ├── get_encoder.py        # Encoder factory -- single entry point
│   ├── vanilla_vae.py        # VanillaVAE (beta=1)
│   ├── beta_vae.py           # Beta-VAE (beta=4)
│   ├── residual_vae.py       # ResidualVAE (skip connections)
│   ├── cvae.py               # Conditional VAE (conditioned on smile label)
│   ├── factor_vae.py         # Factor VAE (TC via discriminator)
│   ├── beta_tc_vae.py        # Beta-TC VAE (analytic TC penalty)
│   ├── disentangled_beta_vae.py  # Disentangled Beta-VAE (privacy bottleneck)
│   ├── vqvae.py              # VQ-VAE core
│   ├── vqvae_wrapper.py      # VQ-VAE ARL wrapper + GradientReversalLayer
│   └── cifar_like/resnet.py  # ResNet-18 classifier / adversary
├── scripts/                  # SLURM sbatch wrappers for Explorer HPC
├── results/                  # Per-checkpoint eval JSON + eval_summary.csv
├── docs/                     # Architecture analysis, Netron graphs
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

For GPU (CUDA 12.8 wheels, e.g. Legion RTX 5080):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. Set up CelebA

**Option A -- HuggingFace (recommended, no manual download):**
```bash
pip install datasets
# Pass --data_source huggingface --hf_cache_dir ./hf_cache when training
```

**Option B -- Local torchvision layout:**
```
<data_dir>/celeba/
    img_align_celeba/   <- extracted images
    list_attr_celeba.txt
    list_eval_partition.txt
    identity_CelebA.txt
    list_bbox_celeba.txt
    list_landmarks_align_celeba.txt
```

### 3. Train

```bash
# VanillaVAE, HuggingFace CelebA
python adversarial_training.py \
  --encoder vanilla_vae \
  --data_source huggingface --hf_cache_dir ./hf_cache \
  --num_epochs 10 --batch_size 16 \
  --exp_name member2_vanillavae_10e

# BetaTCVAE
python adversarial_training.py \
  --encoder beta_tc_vae \
  --data_source huggingface --hf_cache_dir ./hf_cache \
  --num_epochs 10 --batch_size 16 \
  --exp_name member2_betatcvae_10e

# FactorVAE
python adversarial_training.py \
  --encoder factor_vae \
  --data_source huggingface --hf_cache_dir ./hf_cache \
  --num_epochs 10 --batch_size 16 --vae_gamma 10.0 \
  --exp_name member2_factorvae_10e
```

On Explorer HPC, use the SLURM wrappers in `scripts/`:
```bash
sbatch scripts/run_vanillavae_phase2.sh
sbatch scripts/run_residualvae_bottleneck.sh
```

### 4. Evaluate

```bash
python evaluate.py \
  --encoder vanilla_vae,beta_vae \
  --exp_name member2_vanillavae_10e,member2_betavae_10e \
  --checkpoint_dir . \
  --data_source huggingface --hf_cache_dir ./hf_cache \
  --k_levels 70,75,80,85
```

Saves per-run JSON to `<checkpoint_dir>/eval_<exp_name>.json` and appends a row to `eval_summary.csv`.

---

## Encoder Architectures

All encoders share a common ARL interface:

| Encoder | Key | Description |
|---|---|---|
| VanillaVAE | `vanilla_vae` | Standard VAE, beta=1 ELBO |
| Beta-VAE | `beta_vae` | beta=4 KL penalty for disentanglement |
| ResidualVAE | `residual_vae` | Residual blocks for richer multi-scale features |
| CVAE | `cvae` | Conditioned on smile label |
| Factor VAE | `factor_vae` | TC penalty via latent discriminator |
| Beta-TC VAE | `beta_tc_vae` | Analytic TC decomposition, no extra network |
| Disentangled Beta-VAE | `disentangled_beta_vae` | Skip connections + privacy latent sink |
| VQ-VAE | `vq_vae` | Discrete codebook; stabilised with EMA init |

**Common interface:**
```python
recon = encoder(x)                                   # (B, 3, H, W) in [0, 1]
recon, mu, logvar, z = encoder(x, return_aux=True)   # VAE variants only
```

---

## ARL Training Flags

| Flag | Default | Description |
|---|---|---|
| `--encoder` | `unet` | Encoder architecture key (see table above) |
| `--num_epochs` | 50 | Total training epochs |
| `--batch_size` | 64 | Batch size |
| `--lambda_clf` | 1.0 | Privacy penalty weight lambda |
| `--vae_weight` | 0.1 | VAE reconstruction+KL loss weight |
| `--vae_beta` | 1.0 | Beta for KL term (use 4.0 for Beta-VAE) |
| `--vae_gamma` | 10.0 | Gamma for Factor VAE TC discriminator |
| `--latent_dim` | 256 | Bottleneck size D |
| `--warmup_epochs` | 0 | Utility-only warmup epochs before adversary activates |
| `--freeze_clf` | off | Alternating optimisation: encoder step then clf on detached recon |
| `--freeze_utility_clf_arl` | off | Freeze utility clf weights during ARL phases |
| `--latent_adv` | off | Run adversary on latent z instead of reconstructed image |
| `--adv_steps` | 1 | Adversary update steps per encoder step |
| `--cycle_utility_epochs` | 0 | Macro-cycle: utility-only epochs per cycle |
| `--cycle_arl_epochs` | 0 | Macro-cycle: ARL epochs per cycle |
| `--data_source` | `torchvision` | `torchvision` or `huggingface` |
| `--hf_cache_dir` | None | HuggingFace cache directory |
| `--device` | `cuda` | `cuda`, `cpu`, or `auto` |
| `--exp_name` | `celeb` | Checkpoint filename stem |

---

## Pythae Prototyping

`pythae_training.py` trains VAE variants via the [Pythae](https://github.com/clementchadebec/benchmark_VAE) library outside the ARL loop. Used during early development to validate encoder behaviour before integrating into the ARL pipeline.

```bash
python pythae_training.py \
  --variant disentangled_betavae \
  --data_source huggingface --hf_cache_dir ./hf_cache \
  --img_size 64 --latent_dim 32 --beta 4.0 \
  --batch_size 48 --num_epochs 50
```

Supported variants: `disentangled_betavae`, `betatcvae`, `factorvae`.

---

## Compute Profiling

```bash
# MACs + parameter count for all encoders
python compute_flops.py

# Export all encoders to ONNX (outputs to ./onnx_exports/)
python export_onnx.py

# Measure real CPU latency via ONNX Runtime + Jetson estimates
python benchmark_latency.py --onnx_dir ./onnx_exports --runs 50
```

Edge deployment summary:

| Model | GMACs | Jetson Nano | Edge-ready |
|---|---|---|---|
| VanillaVAE | 1.99 | ~161 ms | yes |
| Beta-VAE | 1.99 | ~152 ms | yes |
| ResidualVAE | 2.60 | ~226 ms | marginal |
| VQ-VAE | 3.48 | ~400 ms | marginal |
| BetaTCVAE | 9.98 | ~1664 ms | no |
| Disentangled Beta-VAE | 32.14 | ~5357 ms | no |

---

## Metrics

- **Utility accuracy** -- smile classification (%, want high)
- **Privacy accuracy** -- gender classification (%, want ~50%)
- **NAG** -- Normalised Accuracy Gap: `(util% - 50) / (priv% - 50)`, target > 1.0
- **AUC-ROC** -- threshold-independent discriminability for both tasks
- **Privacy at K% Utility** -- gender accuracy when smile accuracy is constrained to >=K%

---

## Results Summary

Main protocol: 224x224, wvae=0.1, 10 ARL epochs, 20k training samples.

| Model | Util% | Priv% | NAG |
|---|---|---|---|
| VanillaVAE | 86.1 | 88.4 | 0.940 |
| BetaVAE | 81.5 | 85.5 | 0.886 |
| ResidualVAE | 85.9 | 86.3 | 0.988 |
| BetaTCVAE | 84.6 | 85.3 | 0.980 |
| FactorVAE | 87.0 | 88.7 | 0.956 |

Full results and ablations in `results/eval_summary.csv` and `Report.pdf`.

---

## Team

| Member | Branch | Contributions |
|---|---|---|
| Sindhu SureshKumar | `member1-sindhu` | VanillaVAE, BetaVAE, ResidualVAE; IB ablation; training controls; evaluation pipeline |
| Hamza Mubashir | `dev/member_2_hamza` | BetaTCVAE, FactorVAE, CVAE, Disentangled Beta-VAE; Pythae prototyping; compute profiling |
| Malia Howe | `member3-malia` | VQ-VAE stack; codebook stabilisation; GRL training path |
