# Project Proposal: Evaluating Variational Autoencoder Architectures for Privacy-Preserving Edge Vision

## 1. Introduction & Motivation

This project extends the baseline Adversarial Representation Learning (ARL) framework for privacy-preserving edge vision by **systematically replacing and evaluating multiple Variational Autoencoder (VAE) architectures** as the encoder backbone.

Using the CelebA dataset, we will benchmark each VAE variant on two tasks:
- **Utility**: Smile detection
- **Privacy**: Gender classification

The goal is to identify which architecture best preserves task performance while suppressing sensitive attribute leakage. The best-performing architecture will then be further optimized.

A key deliverable is a **modular encoder interface** in the existing codebase, enabling clean, repeatable swapping of VAE variants without disrupting the training pipeline.

---

## 2. Proposed Approach

### 2.1 Architecture Survey & Selection

We will implement and evaluate the following VAE variants as drop-in encoder replacements within the ARL framework:

| VAE Variant | Description |
|-------------|-------------|
| **Vanilla VAE** | Standard variational autoencoder; serves as the primary VAE baseline |
| **β-VAE** | Introduces a β > 1 weighting on the KL divergence term, enforcing stronger disentanglement of latent factors |
| **Conditional VAE (CVAE)** | Conditions the latent space on auxiliary labels, enabling more controlled representation learning |
| **Factor VAE** | Encourages disentanglement through a total correlation penalty term |
| **VQ-VAE** (Vector Quantized VAE) | Replaces continuous latent distributions with discrete codebook representations |

Each architecture will be evaluated as a direct replacement for the encoder component in the existing ARL pipeline.

### 2.2 Modular Implementation Strategy

A core engineering contribution will be the design of a **modular encoder interface** through a unified abstraction layer. This allows any VAE variant to be plugged into the existing ARL training loop without modifying surrounding components (utility classifier, privacy adversary, training logic).

Tasks:
- Review and understand the existing codebase
- Identify encoder input/output contracts within the current implementation
- Build a standardized encoder wrapper class compatible with all VAE variants
- Validate smooth integration of each architecture before comparative experiments begin

### 2.3 Evaluation & Optimization

All architectures will be evaluated under identical training conditions on the CelebA dataset using:

| Metric | Description |
|--------|-------------|
| **Utility Accuracy** | Smile detection accuracy on encoded representations |
| **Privacy Accuracy** | Gender classification accuracy (target: chance-level ~50%) |
| **AUC** | For both utility (target: 1.0) and privacy (target: 0.5) classifiers |
| **NAG** | Normalized Accuracy Gain, as defined in the MASS framework |
| **System Characterization** | FLOPs and MAC operations per architecture |

If time allows, the best-performing VAE variant will be selected for a focused optimization phase:
- Tuning λu/λp trade-off hyperparameters
- Latent space dimensionality
- Adversarial training stability techniques

---

## 3. Task Division

| Team Member | Responsibility |
|-------------|----------------|
| Member 1 | Codebase review; modular encoder interface; Vanilla VAE & β-VAE integration |
| Member 2 | CVAE & Factor VAE implementation; adversarial training loop validation |
| Member 3 | VQ-VAE implementation; evaluation pipeline (metrics, plots); optimization of best model |

*Note: Task division is subject to change after further discussion among team members.*

---

## 4. Expected Outcomes

By the end of this project, we expect to deliver:

1. **Modular, reusable encoder evaluation framework**
2. **Rigorous comparative analysis** of VAE architectures under the ARL privacy-utility objective
3. **Optimized implementation** of the best-performing architecture

Results will be benchmarked against the reported baseline: **85.4% smile / 62.7% gender accuracy** (PrivateEye).

---

## 5. References

- **Baseline Codebase**: [Spring-2026-privacy-preserving-extension](https://github.com/northeastern-eece5698/Spring-2026-privacy-preserving-extension) (private repo)
- **VAE**: Kingma & Welling, 2013 | **β-VAE**: Higgins et al., 2017 | **VQ-VAE**: van den Oord et al., 2017  
- **CVAE**: Harvey, W., Naderiparizi, S., Wood, F. | **Factor VAE**: Duan, Y., Zhang, Q., & Li, J. (2022)
- **PrivateEye** (WACV 2025), **MASS** (2024)
- **CelebA Dataset**: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
