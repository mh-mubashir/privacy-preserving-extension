#!/usr/bin/env python3
"""
make_diagram.py — Generate the ARL pipeline architecture figure for the report.

Outputs (saved to overleaf_figures/):
    arl_pipeline.png   200 dpi, for Overleaf upload
    arl_pipeline.pdf   vector, for Overleaf upload

Architecture depicted (accurate to adversarial_training.py + vqvae_wrapper.py):

  Phase 1 — Warmup:
      Input x  →  Encoder E_θ  →  Latent z  →  Decoder G_θ  →  x̂
      x̂  →  Utility Classifier C_φ  →  ŷ_u (smile)
      (Adversary inactive; encoder + decoder + utility clf update jointly)

  Phase 2 — ARL:
      x̂  →  Adversary A_ψ  →  ŷ_p (gender)
        - Adversary UPDATE path: x̂.detach() → A_ψ (only A_ψ weights update)
        - Encoder UPDATE path:   x̂ → [no_grad on A_ψ] → −λ·L_adv back to E_θ
          (manual gradient reversal; equivalent to GRL)
      Optional: --use_grl wraps input in GradientReversalLayer (Malia,
                models/vqvae_wrapper.py::GradientReversalLayer)

  VQ-VAE path (Malia Howe, branch member3-malia):
      Encoder → z_e → pre-quant conv → VQ Codebook (EMA) → z_q → Decoder
      VectorQuantizer uses EMA updates + lazy init to prevent codebook collapse.

  Training-schedule flags (Sindhu SureshKumar, branch member1-sindhu):
      --freeze_clf               : alternating optimisation (encoder step, then
                                   clf step on detached recon)
      --freeze_utility_clf_arl   : utility clf receives no optimiser steps in
                                   ARL phase (weights frozen after warmup)
      --cycle_utility_epochs /   : macro-epoch cycling — U epochs utility-only,
      --cycle_arl_epochs           then A epochs full ARL, repeating

  Overall encoder objective (Eq. 4 in report):
      L_enc = L_clf − λ · L*_adv + w_vae · L_vae
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT_DIR = 'overleaf_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────────
C = dict(
    enc      = '#2E75B6',   # blue        – encoder / decoder
    enc_fill = '#DDEEFF',
    vq       = '#C55A11',   # burnt-orange – VQ / codebook (Malia)
    vq_fill  = '#FDE8D5',
    util     = '#375623',   # dark-green  – utility classifier
    util_fill= '#E2EFDA',
    adv      = '#C00000',   # dark-red    – adversary / reversed gradient
    adv_fill = '#FCE4D6',
    neutral  = '#404040',   # dark-grey   – input / recon boxes
    bg1      = '#F0FAF0',   # pale-green  – Phase 1 region
    bg2      = '#FFF0F0',   # pale-red    – Phase 2 region
    eq_bg    = '#FFFDE7',   # pale-yellow – loss equation
    ctrl_bg  = '#F0FAF0',   # pale-green  – training controls note
)

# ── canvas ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.2)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── helpers ────────────────────────────────────────────────────────────────────

def rounded_box(cx, cy, w, h, title, sub=None,
                fc='white', ec='black', lw=1.6,
                tsz=9.5, ssz=7.5, bold=True, zorder=4):
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.08',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(rect)
    tw = 'bold' if bold else 'normal'
    if sub:
        ax.text(cx, cy + 0.17, title, ha='center', va='center',
                fontsize=tsz, fontweight=tw, color='black', zorder=zorder + 1)
        ax.text(cx, cy - 0.17, sub, ha='center', va='center',
                fontsize=ssz, color='#444444', style='italic', zorder=zorder + 1)
    else:
        ax.text(cx, cy, title, ha='center', va='center',
                fontsize=tsz, fontweight=tw, color='black', zorder=zorder + 1)


def arr(x1, y1, x2, y2, color='black', lw=1.6, ls='-',
        cs='arc3,rad=0.0', hw=0.22, hl=0.15, zorder=3):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=f'->,head_width={hw},head_length={hl}',
            color=color, lw=lw, linestyle=ls,
            connectionstyle=cs,
        ),
        zorder=zorder,
    )


def label(x, y, text, ha='center', va='center', sz=8, color='black',
          bold=False, italic=False, zorder=5):
    fw = 'bold' if bold else 'normal'
    fs = 'italic' if italic else 'normal'
    ax.text(x, y, text, ha=ha, va=va, fontsize=sz,
            fontweight=fw, style=fs, color=color, zorder=zorder)


# ══════════════════════════════════════════════════════════════════════════════
# Phase background regions
# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: warmup — covers encoder through utility clf
ax.add_patch(FancyBboxPatch(
    (0.18, 2.6), 11.5, 3.35,
    boxstyle='round,pad=0.12',
    facecolor=C['bg1'], edgecolor='#70AD47',
    linewidth=1.2, linestyle='--', zorder=0,
))
label(5.9, 5.82,
      'Phase 1 — Warmup   (encoder  +  decoder  +  utility classifier only)',
      sz=9, color='#375623', italic=True)

# Phase 2: ARL — adversary branch
ax.add_patch(FancyBboxPatch(
    (6.35, 0.22), 6.5, 2.4,
    boxstyle='round,pad=0.12',
    facecolor=C['bg2'], edgecolor=C['adv'],
    linewidth=1.2, linestyle='--', zorder=0,
))
label(9.6, 0.42, 'Phase 2 — ARL   (adversary active)',
      sz=9, color='#843C0C', italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline boxes  (main row y = 3.85)
# ══════════════════════════════════════════════════════════════════════════════
Y = 3.85

# Input
rounded_box(0.88, Y, 1.25, 0.80,
            r'Input $x$',
            r'$\mathbb{R}^{3\times224\times224}$',
            fc='#F4F4F4', ec=C['neutral'], tsz=9.5)

# Encoder
rounded_box(2.95, Y, 1.75, 0.80,
            r'Encoder  $E_\theta$',
            'VanillaVAE / BetaVAE\nResidualVAE / BetaTCVAE\nFactorVAE / VQ-VAE',
            fc=C['enc_fill'], ec=C['enc'], tsz=9.5, ssz=7)

# Latent space
rounded_box(5.15, Y, 1.75, 0.80,
            r'Latent  $z$',
            r'$z\!=\!\mu+\sigma\varepsilon$  (VAE, cont.)' + '\n' +
            r'$z_q = e_k$  (VQ-VAE, discrete)',
            fc=C['vq_fill'], ec=C['vq'], tsz=9.5, ssz=7)

# Decoder
rounded_box(7.35, Y, 1.5, 0.80,
            r'Decoder  $G_\theta$', 'reconstruction',
            fc=C['enc_fill'], ec=C['enc'], tsz=9.5)

# Reconstruction output
rounded_box(9.4, Y, 1.05, 0.75,
            r'$\hat{x}$', 'recon.',
            fc='#F4F4F4', ec=C['neutral'], tsz=11)

# Utility classifier (upper right)
rounded_box(11.55, 4.75, 1.65, 0.72,
            r'Utility  $C_\phi$', 'smile classifier',
            fc=C['util_fill'], ec=C['util'], tsz=9.5)

# Adversary (lower right)
rounded_box(11.55, 1.42, 1.65, 0.72,
            r'Adversary  $A_\psi$', 'gender classifier',
            fc=C['adv_fill'], ec=C['adv'], tsz=9.5)

# ══════════════════════════════════════════════════════════════════════════════
# Forward-pass arrows
# ══════════════════════════════════════════════════════════════════════════════
arr(1.52, Y, 2.07, Y)                         # input → encoder
arr(3.83, Y, 4.27, Y)                         # encoder → latent
arr(6.03, Y, 6.60, Y)                         # latent → decoder
arr(8.10, Y, 8.87, Y)                         # decoder → recon

# recon → utility clf (up-right, solid)
arr(9.93, 4.10, 10.72, 4.58, color=C['util'])

# recon → adversary (down-right, dashed = detached input)
arr(9.93, 3.60, 10.72, 1.72, color=C['adv'], ls='dashed')

# ══════════════════════════════════════════════════════════════════════════════
# Reversed adversary gradient back to encoder  (curved dashed red arrow)
# ══════════════════════════════════════════════════════════════════════════════
arr(10.72, 1.08, 2.95, 3.45,
    color=C['adv'], lw=1.8, ls='dashed',
    cs='arc3,rad=-0.28', hw=0.22, hl=0.17)

# ══════════════════════════════════════════════════════════════════════════════
# Output labels
# ══════════════════════════════════════════════════════════════════════════════
label(12.44, 4.75,
      r'$\hat{y}_u$' + '\n(smile)', sz=9, color=C['util'], bold=True)
label(12.44, 1.42,
      r'$\hat{y}_p$' + '\n(gender)', sz=9, color=C['adv'], bold=True)

# ══════════════════════════════════════════════════════════════════════════════
# Annotation: "detach" on adversary input arrow
# ══════════════════════════════════════════════════════════════════════════════
label(10.22, 2.62, 'detach', sz=7.5, color=C['adv'], italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# Annotation: reversed gradient label on curved arrow
# ══════════════════════════════════════════════════════════════════════════════
label(6.3, 2.22,
      r'$-\lambda\,\nabla\mathcal{L}_{adv}$   (reversed gradient  →  encoder)',
      sz=8.5, color=C['adv'], italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# Loss equation box (bottom left)
# ══════════════════════════════════════════════════════════════════════════════
ax.text(4.0, 0.88,
        r'$\mathcal{L}_{enc}\ =\ \mathcal{L}_{clf}\ -\ '
        r'\lambda\,\mathcal{L}^{*}_{adv}\ +\ w_{vae}\,\mathcal{L}_{vae}$',
        ha='center', va='center', fontsize=10.5,
        bbox=dict(facecolor=C['eq_bg'], edgecolor='#B8860B',
                  linewidth=1.2, pad=5, boxstyle='round,pad=0.4'),
        zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# Annotation: VQ-VAE codebook note (Malia)
# ══════════════════════════════════════════════════════════════════════════════
ax.text(5.15, 2.78,
        'VQ-VAE (Malia, member3-malia):\n'
        r'$z_q = \arg\min_k \|z_e - e_k\|$'  + '\n'
        'EMA codebook  +  lazy init\n'
        '(prevents codebook collapse)',
        ha='center', va='center', fontsize=7,
        bbox=dict(facecolor=C['vq_fill'], edgecolor=C['vq'],
                  linewidth=1, pad=3, boxstyle='round,pad=0.3'),
        zorder=5)
# small arrow up to latent box
arr(5.15, 3.08, 5.15, 3.44, color=C['vq'], lw=1.0, hw=0.12, hl=0.10)

# ══════════════════════════════════════════════════════════════════════════════
# Annotation: GRL option (Malia)
# ══════════════════════════════════════════════════════════════════════════════
label(10.7, 0.82,
      '* --use_grl: GradientReversalLayer\n  (Malia, models/vqvae_wrapper.py)',
      sz=7, color='#843C0C', italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# Annotation: training-schedule controls (Sindhu)
# ══════════════════════════════════════════════════════════════════════════════
ax.text(0.88, 2.84,
        'Training controls (Sindhu):\n'
        '--freeze_clf\n'
        '--freeze_utility_clf_arl\n'
        '--cycle_utility/arl_epochs',
        ha='center', va='center', fontsize=7,
        bbox=dict(facecolor=C['ctrl_bg'], edgecolor='#70AD47',
                  linewidth=1, pad=3, boxstyle='round,pad=0.3'),
        zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# Phase-1 / Phase-2 legend dots
# ══════════════════════════════════════════════════════════════════════════════
for xp, yp, col, lbl in [
    (0.42, 5.82, '#70AD47', 'Phase 1'),
    (0.42, 0.42, C['adv'],  'Phase 2'),
]:
    ax.plot(xp, yp, 'o', color=col, markersize=7, zorder=6)
    label(xp + 0.55, yp, lbl, sz=8, color=col, bold=True, ha='left')

# ── save ───────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.2)
for ext, kw in [('png', dict(dpi=200)), ('pdf', {})]:
    out = os.path.join(OUT_DIR, f'arl_pipeline.{ext}')
    plt.savefig(out, bbox_inches='tight', facecolor='white', **kw)
    print(f'Saved  {out}')

plt.close()
print('Done.')
