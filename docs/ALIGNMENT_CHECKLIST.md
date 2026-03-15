# Project Proposal Alignment Checklist

This document verifies that the current implementation aligns with the goals outlined in the Project Proposal (docs/PROJECT_PROPOSAL.md).

---

## 1. Dataset & Tasks ✓

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| CelebA dataset | ✓ | `torchvision.datasets.CelebA` with `--data_dir`, `--download` |
| Utility task: Smile detection | ✓ | `u_task = 31` (CelebA attr index) |
| Privacy task: Gender classification | ✓ | `p_task = 20` (CelebA attr index) |
| 224×224 RGB images | ✓ | Resize + ToTensor transforms |

---

## 2. Encoder Architectures ✓

| VAE Variant | Status | Module |
|-------------|--------|--------|
| Vanilla VAE | Member 1 | — |
| β-VAE | Member 1 | — |
| **CVAE** | ✓ Implemented | `models/cvae.py` |
| **Factor VAE** | ✓ Implemented | `models/factor_vae.py` |
| VQ-VAE | Member 3 | — |

---

## 3. Modular Encoder Interface ✓

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Unified abstraction layer | ✓ | `models/get_encoder.py` |
| Drop-in encoder replacement | ✓ | `--encoder unet\|cvae\|factor_vae` |
| No modification to utility/adversary | ✓ | ResNet classifiers unchanged |
| Same output contract (B, 3, 224, 224) in [0,1] | ✓ | All encoders produce classifier-ready output |

---

## 4. Adversarial Training Loop ✓

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| ARL objective (minimize utility loss, maximize adv loss) | ✓ | `loss_clf - lambda_clf * loss_adv` |
| Adversary trained to predict private attr | ✓ | `adv_model`, BCE on gender |
| Utility classifier trained to predict smile | ✓ | `clf_model`, BCE on smile |
| VAE encoders: recon + KL + (Factor: TC) losses | ✓ | `vae_weight`, `vae_beta`, `vae_gamma` |

---

## 5. Evaluation Metrics (Proposal §2.3)

| Metric | Status | Notes |
|--------|--------|------|
| Utility Accuracy | ✓ | Smile detection acc on encoded data; logged per epoch |
| Privacy Accuracy | ✓ | Gender acc (target ~50%); logged per epoch |
| AUC | Member 3 | Evaluation pipeline responsibility |
| NAG (MASS) | Member 3 | Evaluation pipeline responsibility |
| FLOPs / MACs | Optional | `torchprofile` in UNet `__main__` |

---

## 6. Expected Outcomes (Proposal §4)

| Outcome | Status |
|---------|--------|
| Modular, reusable encoder framework | ✓ |
| Comparative analysis capability | ✓ (UNet, CVAE, Factor VAE swappable) |
| Benchmark vs PrivateEye (85.4% smile / 62.7% gender) | To be measured after training |

---

## 7. Task Division – Member 2 ✓

| Responsibility | Status |
|----------------|--------|
| CVAE implementation | ✓ |
| Factor VAE implementation | ✓ |
| Adversarial training loop validation | ✓ |
