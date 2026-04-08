# VAE Architecture Differences and Performance Analysis

## Overview

This project uses six VAE-based encoders as the privacy-preserving layer in an
Adversarial Representation Learning (ARL) pipeline. Each encoder takes a raw
CelebA face image (224×224 RGB) and outputs a reconstructed image that should
preserve smile detection (utility) while suppressing gender (privacy).

The key question is: **why do different VAE variants produce different
utility–privacy trade-offs?** This document answers that for each variant.

---

## 1. Vanilla VAE

**Architecture:** Conv encoder (3→32→64→128→256→512 channels) → bottleneck
(μ, σ) → mirrored deconv decoder.

**Loss:** `MSE(recon, input) + β × KL(q(z|x) || N(0,I))` where β = 1.

**Privacy mechanism:** The KL term forces the latent vector toward a standard
normal. Because the encoder must compress the image through a narrow
bottleneck, fine-grained gender cues (jawline width, brow thickness) are more
likely to be discarded than coarse smile cues (lip curvature is a salient
low-frequency pattern). Privacy is a side effect of lossy compression, not an
explicit design choice.

**Why it performs well on utility but not great on privacy:**
A β = 1 KL penalty is weak. The encoder can afford to memorise most image
details, including gender. The ARL adversary therefore still finds enough
signal. Result: high utility (~86%), but privacy adversary also stays high
(~88%).

---

## 2. Beta-VAE (β = 4)

**Architecture:** Same as Vanilla VAE.

**Loss:** Same MSE + KL, but KL is weighted ×4 more heavily.

**Privacy mechanism:** The stronger KL penalty forces the latent distribution
to be more Gaussian. This disentangles latent dimensions — each dimension
captures one independent factor of variation. Gender and smile, being
independent facial attributes, tend to map to separate latent dimensions.
When the ARL adversary tries to read gender from the reconstructed image, the
encoder can selectively suppress those dimensions.

**Why it has slightly lower utility than Vanilla VAE:**
A strong KL penalty sacrifices reconstruction quality. The encoder discards
more information to satisfy the prior, occasionally losing smile-related
texture. Result: ~82% utility but privacy also drops to ~85% — slightly
better privacy-utility gap than vanilla.

---

## 3. Residual VAE

**Architecture:** ResNet-style residual blocks in the encoder (shortcut
connections), Vanilla VAE decoder. Skip connections within the encoder let
gradient flow directly to early layers.

**Loss:** Same as Vanilla VAE (β = 1 KL).

**Privacy mechanism:** Identical to Vanilla VAE in loss design. The
architectural difference is that residual connections allow the encoder to
learn a richer hierarchy of features without vanishing gradients. This means
the bottleneck can represent more abstract, task-relevant features.

**Why it achieves the best NAG score:**
Residual encoders empirically learn more task-specific representations. After
ARL training, the encoder tends to retain smile-relevant features and push
gender-related features toward the prior. This yields the best utility/privacy
ratio (NAG ≈ 0.99 in our experiments vs. ~0.94 for vanilla).

---

## 4. Beta-TC VAE

**Architecture:** Wider channels (3→64→128→256→512→1024) with residual blocks.
Explicit analytical Total Correlation (TC) penalty added to the loss.

**Loss:** `MSE + β_KL × (MI term) + β_TC × TC(z) + β_dim × dim-KL`

The TC term directly penalises statistical dependencies *between* latent
dimensions. This is different from standard β-VAE where the KL term indirectly
encourages independence.

**Privacy mechanism:** By explicitly minimising TC, the encoder forces latent
dimensions to be independent. If smile lives in dimensions 0–3 and gender in
dimensions 4–7, a perfect TC-VAE bottleneck makes those groups truly
independent — the ARL adversary cannot use the smile dimensions to infer
gender.

**Expected advantage over standard β-VAE:**
More principled disentanglement. In practice, with only 10 epochs the TC
penalty may not fully converge, so privacy improvements are moderate but the
direction is correct.

---

## 5. Disentangled Beta-VAE

**Architecture:** Skip-connection encoder. The latent vector is split into two
halves: z_util (for smile) and z_priv (for gender). Only z_util is decoded.
z_priv is effectively a "privacy sink" — the encoder can route gender
information there, where it is never decoded back to image space.

**Loss:** Standard β-VAE loss, but reconstruction is only from z_util.

**Privacy mechanism:** This is the most explicit privacy bottleneck of all
variants. By design, z_priv never contributes to the reconstructed image.
If the encoder routes gender information into z_priv, the output image will
contain no gender signal — the adversary gets nothing to work with.

**Limitation:** The split is implicit — the encoder is not explicitly forced to
put gender in z_priv. With longer training, the adversary signal in ARL will
gradually push gender information toward z_priv, but this may require more
epochs than the other variants to fully converge.

---

## 6. VQ-VAE (Member 3's implementation)

**Architecture:** Encoder → vector quantiser (discrete codebook) → decoder.
The bottleneck is discrete rather than continuous.

**Loss:** `MSE + codebook loss + commitment loss`. No KL term.

**Privacy mechanism:** Discretisation is a strong form of information
compression. Only patterns that appear frequently enough in the training set
get a codebook entry. Rare gender-correlated patterns (e.g., specific beard
textures) may not have codebook entries and are therefore discarded.

**Why it collapsed in our experiments:**
VQ-VAE collapse occurs when most input images map to the same codebook entry
(codebook collapse). The reconstructed image then looks like an average face
regardless of input, so *both* utility and privacy classifiers see random noise.
Our experiments confirmed this: utility 46%, adversary 76% (the adversary
predicts the prior, not learned features). The fix is either (a) higher
codebook commitment loss, (b) exponential moving average codebook updates, or
(c) training at a lower resolution to reduce the quantisation burden.

---

## Compute Cost Characterisation (224×224 input, measured with thop)

| Encoder | MACs | Parameters | Relative Cost | Jetson Nano (est.) |
|---|---|---|---|---|
| Vanilla VAE | 1.99G | 23.65M | 1.0x | ~331ms |
| Beta-VAE (β=4) | 1.99G | 23.65M | 1.0x | ~331ms |
| Residual VAE | 2.60G | 26.97M | 1.3x | ~433ms |
| VQ-VAE | 3.48G | 0.66M | 1.8x | ~580ms |
| Beta-TC VAE | 9.98G | 38.14M | 5.0x | ~1664ms |
| Disentangled Beta-VAE | 32.14G | 17.78M | 16.2x | ~5357ms |

All variants exceed the 100ms real-time target at 224×224. VanillaVAE and
BetaVAE are the most edge-friendly. To meet deployment constraints, reducing
input resolution to 64×64 would give ~16x speedup; INT8 quantization on
supported hardware provides another ~4x. VQ-VAE has the smallest parameter
count (662K) making it the best candidate for further pruning and quantization,
despite its current collapse issue in ARL training.

Network graphs for all six variants were visualised using netron.app after
ONNX export, confirming the architectural differences described above.

---

## Performance Summary (λ = 1.0, standard single-phase training, 10 epochs)

| Encoder               | Utility% | Privacy% | NAG    | Root Cause of Privacy Level |
|-----------------------|----------|----------|--------|------------------------------|
| Vanilla VAE           | 86.1     | 88.4     | 0.94   | Weak KL; most info preserved |
| Beta-VAE (β=4)        | 82.1     | 85.3     | 0.91   | Stronger KL, some disentangle|
| Residual VAE          | 85.9     | 86.4     | 0.99   | Rich features + ARL balance  |
| Beta-TC VAE (smoke)   | 72.7     | 82.0     | —      | TC not converged in 3 epochs |
| Disent. Beta-VAE (sm.)| 57.9     | 80.9     | —      | Split latent; 3 epochs       |
| VQ-VAE                | 46.1     | 76.2     | 0.00   | Codebook collapse            |

---

## Performance Summary (two-phase training, λ = 1.0, 3 warmup + 10 ARL epochs)

| Encoder | Utility% | Privacy% | NAG |
|---|---|---|---|
| Vanilla VAE | 87.40 | 88.85 | 0.963 |
| Beta-VAE | 83.55 | 85.75 | 0.939 |
| Residual VAE | 86.40 | 87.60 | 0.968 |

Two-phase training improved utility by 1–2% across all variants compared to
single-phase. Privacy suppression remained similar, motivating the λ=2.0
experiments described below.

## Performance Summary (two-phase training, λ = 2.0 — in progress)

Results pending. Higher lambda increases adversary pressure on the encoder
during the ARL phase. Expected outcome: lower privacy accuracy at the cost
of some utility loss, demonstrating the privacy-utility trade-off explicitly.

---

## Why Two-Phase Training Should Help

In single-phase training, the encoder receives two conflicting gradient signals
from epoch 1: "keep smile" (utility) and "remove gender" (adversary). Early
in training, the encoder has not yet learned to extract smile features at all.
The adversary pressure therefore causes the encoder to destroy the image
indiscriminately, hurting both utility and privacy.

In two-phase training:
1. **Warmup (epochs 1–3):** Only utility loss. The encoder learns to reconstruct
   faces well enough for smile detection. The adversary is not updated and has
   no influence.
2. **ARL phase (epochs 4–13):** The adversary is activated against an encoder
   that already has a strong utility representation. Now when the adversary
   applies pressure, the encoder selectively suppresses gender features rather
   than collapsing reconstruction quality.

The expected outcome is a lower privacy accuracy (gender harder to predict)
without a corresponding drop in utility accuracy — the key metric the professor
flagged as insufficient.
