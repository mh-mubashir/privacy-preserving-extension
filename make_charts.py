"""
make_charts.py -- Generate all charts and visual assets for the project report.

Produces five bar/line charts and two image assets, all written to docs/charts/:

  chart_utility_privacy.png  -- Grouped bar: utility vs privacy accuracy per model
  chart_nag.png              -- Horizontal bar: NAG score ranking (higher = better balance)
  chart_lambda_sweep.png     -- Line: privacy-utility trade-off as lambda varies (1/2/3)
  chart_auc.png              -- Grouped bar: AUC-ROC for utility and privacy classifiers
  chart_bottleneck.png       -- Bar: latent_dim ablation on ResidualVAE (dim 32/64/256)
  recon_vanillavae.png       -- Collage: VanillaVAE reconstruction at epoch 1 vs epoch 10
  recon_betavae.png          -- Collage: BetaVAE reconstruction at epoch 1 vs epoch 10
  netron_vanilla_vs_residual.png -- Side-by-side Netron ONNX architecture diagrams

Data source: hardcoded evaluation results from Table II/III of the project report
(two-phase ARL training, lambda=2, CelebA 20k subset, utility=smile, privacy=gender).
Image inputs for collages come from visuals/ and docs/netron_graphs/.

Usage:
    python make_charts.py

Requirements:
    pip install matplotlib numpy Pillow
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import os

os.makedirs("docs/charts", exist_ok=True)

# ── Shared style ─────────────────────────────────────────────────────────────
BG    = '#1A1A2E'
PANEL = '#16213E'
BLUE  = '#2196F3'
RED   = '#F44336'
GOLD  = '#FFA726'
GREEN = '#4CAF50'
GREY  = '#90A4AE'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   PANEL,
    'axes.edgecolor':   GREY,
    'axes.labelcolor':  'white',
    'xtick.color':      'white',
    'ytick.color':      'white',
    'text.color':       'white',
    'grid.color':       '#2A2A4A',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
})

# ════════════════════════════════════════════════════════════════════════════
# CHART 1 — Utility vs Privacy Accuracy (grouped bar)
# ════════════════════════════════════════════════════════════════════════════
models  = ['VanillaVAE', 'BetaVAE', 'ResidualVAE', 'BetaTCVAE', 'FactorVAE', 'VQ-VAE']
utility = [86.1, 81.5, 85.9, 84.6, 87.0, 46.1]
privacy = [88.4, 85.5, 86.3, 85.3, 88.7, 76.2]

x = np.arange(len(models))
w = 0.36

fig, ax = plt.subplots(figsize=(13, 5.5))
b1 = ax.bar(x - w/2, utility, w, color=BLUE,  label='Utility Acc (Smile)',  zorder=3, edgecolor='none')
b2 = ax.bar(x + w/2, privacy, w, color=RED,   label='Privacy Acc (Gender)', zorder=3, edgecolor='none')
ax.axhline(50,  color='white', lw=1.2, ls=':', alpha=0.6, label='Random chance (50%)')
ax.axhline(85,  color=GREEN,   lw=1.2, ls='--', alpha=0.5, label='Utility target (85%)')

for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=9, color=BLUE)
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=9, color=RED)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylim(0, 103)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Utility vs Privacy Accuracy — All Models', fontsize=14, fontweight='bold', pad=12)
ax.legend(fontsize=10, loc='lower right')
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('docs/charts/chart_utility_privacy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1 done")

# ════════════════════════════════════════════════════════════════════════════
# CHART 2 — NAG Score comparison
# ════════════════════════════════════════════════════════════════════════════
nag_models = ['ResidualVAE', 'BetaTCVAE', 'FactorVAE', 'VanillaVAE', 'BetaVAE', 'VQ-VAE']
nag_vals   = [0.9876, 0.9802, 0.9561, 0.9401, 0.8859, 0.0]
nag_cols   = [GREEN, '#03A9F4', BLUE, '#29B6F6', GOLD, RED]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.barh(nag_models, nag_vals, color=nag_cols, edgecolor='none', zorder=3)
for bar, v in zip(bars, nag_vals):
    ax.text(v + 0.01, bar.get_y()+bar.get_height()/2,
            f'{v:.3f}', va='center', fontsize=11, color='white', fontweight='bold')
ax.set_xlim(0, 1.15)
ax.set_xlabel('Normalized Accuracy Gap (NAG)', fontsize=12)
ax.set_title('NAG Score — Higher = Better Privacy-Utility Balance\n(NAG = utility gain / privacy leak)',
             fontsize=13, fontweight='bold', pad=10)
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('docs/charts/chart_nag.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2 done")

# ════════════════════════════════════════════════════════════════════════════
# CHART 3 — Lambda sweep (privacy-utility trade-off)
# ════════════════════════════════════════════════════════════════════════════
lambdas    = [1.0, 2.0, 3.0]
van_util   = [86.1,  87.5,  86.1]
van_priv   = [88.4,  88.6,  88.4]
beta_util  = [81.45, 83.55, 82.1]
beta_priv  = [85.5,  85.75, 85.3]
res_util   = [85.9,  86.85, None]
res_priv   = [86.35, 87.6,  None]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for ax, util, priv, title in [
    (ax1, van_util, van_priv, 'VanillaVAE'),
    (ax2, beta_util, beta_priv, 'BetaVAE'),
]:
    ax.plot([1,2,3], util, 'o-', color=BLUE, lw=2.5, ms=8, label='Utility (Smile)')
    ax.plot([1,2,3], priv, 's--', color=RED,  lw=2.5, ms=8, label='Privacy (Gender)')
    ax.axhline(50, color='white', lw=1, ls=':', alpha=0.5)
    ax.axhline(85, color=GREEN,   lw=1, ls='--', alpha=0.4, label='85% target')
    for lam, u, p in zip([1,2,3], util, priv):
        if u: ax.annotate(f'{u:.1f}%', (lam, u), textcoords='offset points',
                           xytext=(5, 5), fontsize=9, color=BLUE)
        if p: ax.annotate(f'{p:.1f}%', (lam, p), textcoords='offset points',
                           xytext=(5,-12), fontsize=9, color=RED)
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(['λ=1', 'λ=2', 'λ=3'])
    ax.set_ylim(40, 98)
    ax.set_xlabel('Lambda (privacy penalty weight)', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'{title} — λ Sweep', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

fig.suptitle('Privacy-Utility Trade-off: Effect of λ', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('docs/charts/chart_lambda_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3 done")

# ════════════════════════════════════════════════════════════════════════════
# CHART 4 — AUC comparison radar-style bar
# ════════════════════════════════════════════════════════════════════════════
auc_models = ['VanillaVAE', 'BetaVAE', 'ResidualVAE', 'BetaTCVAE', 'FactorVAE', 'VQ-VAE']
u_auc = [0.9364, 0.9006, 0.9318, 0.9233, 0.9439, 0.512]
p_auc = [0.9370, 0.9106, 0.9242, 0.9170, 0.9484, 0.509]

x = np.arange(len(auc_models))
w = 0.36
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w/2, u_auc, w, color=BLUE,  label='Utility AUC-ROC', zorder=3)
ax.bar(x + w/2, p_auc, w, color=RED,   label='Privacy AUC-ROC', zorder=3)
ax.axhline(0.5, color='white', lw=1.2, ls=':', alpha=0.6, label='Random (0.5)')

for i, (u, p) in enumerate(zip(u_auc, p_auc)):
    ax.text(i-w/2, u+0.005, f'{u:.3f}', ha='center', va='bottom', fontsize=8.5, color=BLUE)
    ax.text(i+w/2, p+0.005, f'{p:.3f}', ha='center', va='bottom', fontsize=8.5, color=RED)

ax.set_xticks(x)
ax.set_xticklabels(auc_models, fontsize=10)
ax.set_ylim(0.3, 1.05)
ax.set_ylabel('AUC-ROC Score', fontsize=12)
ax.set_title('AUC-ROC Comparison — Utility vs Privacy Discriminability', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('docs/charts/chart_auc.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 4 done")

# ════════════════════════════════════════════════════════════════════════════
# ASSET — VanillaVAE reconstruction collage (epoch 1 vs epoch 10)
# ════════════════════════════════════════════════════════════════════════════
def make_recon_collage(model_prefix, label, out_path):
    e1  = Image.open(f'visuals/{model_prefix}_epoch01.png')
    e10 = Image.open(f'visuals/{model_prefix}_epoch10.png')

    W, H_img = e1.size
    gap  = 20
    text_h = 60
    total_h = H_img*2 + gap*3 + text_h*2
    collage = Image.new('RGB', (W + 80, total_h), (26, 26, 46))

    # paste images
    collage.paste(e1,  (40, text_h))
    collage.paste(e10, (40, text_h + H_img + gap + text_h))

    # add matplotlib text labels
    fig, ax = plt.subplots(figsize=(W/100, total_h/100))
    fig.patch.set_facecolor('#1A1A2E')
    ax.axis('off')
    ax.imshow(np.array(collage))
    ax.text(W/2+40, text_h*0.6, f'{label} — Epoch 1 Reconstruction',
            ha='center', va='center', color='#FFA726', fontsize=16, fontweight='bold')
    ax.text(W/2+40, text_h + H_img + gap + text_h*0.6,
            f'{label} — Epoch 10 Reconstruction',
            ha='center', va='center', color='#4CAF50', fontsize=16, fontweight='bold')
    ax.set_xlim(0, W+80)
    ax.set_ylim(total_h, 0)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

make_recon_collage('member1_vanillavae_10e_20k', 'VanillaVAE',
                   'docs/charts/recon_vanillavae.png')
print("Recon collage VanillaVAE done")

make_recon_collage('member1_betavae_10e_20k', 'BetaVAE',
                   'docs/charts/recon_betavae.png')
print("Recon collage BetaVAE done")

# ════════════════════════════════════════════════════════════════════════════
# ASSET — Side-by-side Netron graphs (VanillaVAE vs ResidualVAE)
# ════════════════════════════════════════════════════════════════════════════
def side_by_side_netron(path_a, label_a, path_b, label_b, out_path):
    img_a = Image.open(path_a)
    img_b = Image.open(path_b)
    h = min(img_a.height, img_b.height, 900)
    wa = int(img_a.width * h / img_a.height)
    wb = int(img_b.width * h / img_b.height)
    img_a = img_a.resize((wa, h), Image.LANCZOS)
    img_b = img_b.resize((wb, h), Image.LANCZOS)
    gap = 30
    total_w = wa + wb + gap
    fig, axes = plt.subplots(1, 2, figsize=(total_w/100, (h+80)/100),
                              gridspec_kw={'width_ratios': [wa, wb]})
    fig.patch.set_facecolor('#1A1A2E')
    for ax, img, label in [(axes[0], img_a, label_a), (axes[1], img_b, label_b)]:
        ax.imshow(np.array(img))
        ax.set_title(label, color='#FFA726', fontsize=14, fontweight='bold', pad=8)
        ax.axis('off')
    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()

side_by_side_netron(
    'docs/netron_graphs/vanilla_vae.onnx-4ecff880-038f-430f-b607-169d134780e3.png', 'VanillaVAE',
    'docs/netron_graphs/residual_vae.onnx-aacd6221-5d1f-4348-93ec-3dd3223e50ed.png', 'ResidualVAE',
    'docs/charts/netron_vanilla_vs_residual.png'
)
print("Netron side-by-side done")

# ════════════════════════════════════════════════════════════════════════════
# CHART 5 — Information Bottleneck: latent_dim ablation
# ════════════════════════════════════════════════════════════════════════════
dims      = [32, 64, 256]
ib_util   = [75.1, None, 86.9]
ib_priv   = [83.15, None, 87.6]
dim_labels = ['dim=32\n(bottleneck)', 'dim=64\n(running…)', 'dim=256\n(baseline)']

fig, ax = plt.subplots(figsize=(9, 5))
x = [0, 1, 2]
w = 0.32

# plot only known values
known_u = [ib_util[0], ib_util[2]]
known_p = [ib_priv[0], ib_priv[2]]
known_x = [0, 2]

ax.bar([xi - w/2 for xi in known_x], known_u, w, color=BLUE,  label='Utility Acc', zorder=3)
ax.bar([xi + w/2 for xi in known_x], known_p, w, color=RED,   label='Privacy Acc', zorder=3)

# pending bar
ax.bar(1 - w/2, 82, w, color=BLUE,  alpha=0.25, zorder=3, label='_nolegend_')
ax.bar(1 + w/2, 85, w, color=RED,   alpha=0.25, zorder=3, label='_nolegend_')
ax.text(1, 83, 'pending', ha='center', va='bottom', fontsize=10, color='white', style='italic')

for xi, u, p in zip(known_x, known_u, known_p):
    ax.text(xi-w/2, u+0.5, f'{u}%', ha='center', va='bottom', fontsize=10, color=BLUE, fontweight='bold')
    ax.text(xi+w/2, p+0.5, f'{p}%', ha='center', va='bottom', fontsize=10, color=RED,  fontweight='bold')

ax.axhline(50, color='white', lw=1.2, ls=':', alpha=0.6, label='Random (50%)')
ax.axhline(85, color=GREEN,   lw=1.2, ls='--', alpha=0.5, label='Utility target (85%)')
ax.set_xticks(x)
ax.set_xticklabels(dim_labels, fontsize=11)
ax.set_ylim(40, 100)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Information Bottleneck — Effect of Latent Dimension\n'
             'Privacy drops as bottleneck tightens (ResidualVAE, λ=2, two-phase)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('docs/charts/chart_bottleneck.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 5 (bottleneck ablation) done")

print("\nAll charts and assets saved to docs/charts/")
