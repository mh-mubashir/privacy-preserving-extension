#!/bin/bash
# Batch-eval all known ARL checkpoints with Privacy@K (writes eval_<exp>.json + eval_summary.csv).
#
# Run on a GPU node (login node has no CUDA):
#   sbatch scripts/sbatch_eval_all_privacy_atk.sh
# Or interactively:
#   bash scripts/eval_all_privacy_atk.sh
#
# Env:
#   REPO              repo root (default: $HOME/privacy-preserving-extension)
#   CHECKPOINT_DIR    where encoder_model_<exp>.pt lives (default scratch arls)
#   HF_CACHE          HuggingFace cache for CelebA
#   COPY_TO_REPO      if set to 1, copy eval_*.json + eval_summary.csv into $REPO/results/

set -euo pipefail

if [[ -f /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh ]]; then
  source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
  conda activate pp_ext_member2_v2
  module load cuda/12.1.1 2>/dev/null || true
fi

REPO="${REPO:-$HOME/privacy-preserving-extension}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/$USER/pp_ext_member2/arls}"
HF_CACHE="${HF_CACHE:-/scratch/$USER/pp_ext_member2/hf_cache}"
K_LEVELS="${K_LEVELS:-70,75,80,85}"

cd "$REPO"
export PYTHONPATH="$REPO"

run_one() {
  local enc="$1"
  local exp="$2"
  shift 2
  local encfile="${CHECKPOINT_DIR}/encoder_model_${exp}.pt"
  if [[ ! -f "$encfile" ]]; then
    echo "[SKIP] no checkpoint: $encfile"
    return 0
  fi
  echo ""
  echo "========== evaluate: $enc  $exp $@ =========="
  python evaluate.py \
    --encoder "$enc" \
    --exp_name "$exp" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data_source huggingface \
    --hf_cache_dir "$HF_CACHE" \
    --device auto \
    --k_levels "$K_LEVELS" \
    --num_workers "${NUM_WORKERS:-1}" \
    "$@"
}

# --- Member 1: 10 epoch + lambda3 (latent_dim 256) ---
run_one vanilla_vae member1_vanillavae_10e_20k
run_one beta_vae    member1_betavae_10e_20k
run_one residual_vae member1_residualvae_10e_20k
run_one vanilla_vae member1_vanillavae_lambda3
run_one beta_vae    member1_betavae_lambda3

# --- Phase-2 / sweeps (256) ---
run_one vanilla_vae p2_vanillavae_13e_20k
run_one vanilla_vae p2_vanillavae_lam2_13e_20k
run_one beta_vae    p2_betavae_13e_20k
run_one beta_vae    p2_betavae_lam2_13e_20k
run_one residual_vae p2_residualvae_13e_20k
run_one residual_vae p2_residualvae_lam2_13e_20k
run_one residual_vae p2_residualvae_lam5_13e_20k
run_one residual_vae p2_residualvae_latentadv_13e_20k

# --- Information bottleneck (match training latent_dim) ---
run_one residual_vae p2_residualvae_ib32_13e_20k --latent_dim 32
run_one residual_vae p2_residualvae_ib64_13e_20k --latent_dim 64

# --- Macro cycle + optional freeze runs ---
run_one residual_vae member1_residualvae_cycle1_1_13e_20k
run_one vanilla_vae  member1_vanillavae_freeze_utility_arl
run_one residual_vae member1_residualvae_freeze_utility_arl

# --- Member 2 (Hamza) ---
run_one cvae       member2_cvae_3e_6k
run_one factor_vae member2_factorvae_3e_6k

# --- Member 3 VQ-VAE (224 unless you trained at another size) ---
run_one vq_vae member3_vqvae_10e_20k

# --- Hamza: add more lines here when exp_name is known, e.g. ---
# run_one beta_tc_vae member2_betatc_10e_20k
# run_one disentangled_beta_vae member2_disentangled_10e_20k

echo ""
echo "Done. Outputs under: $CHECKPOINT_DIR"
ls -la "$CHECKPOINT_DIR"/eval_*.json 2>/dev/null | tail -20 || true

if [[ "${COPY_TO_REPO:-0}" == "1" ]]; then
  mkdir -p "$REPO/results"
  cp -v "$CHECKPOINT_DIR"/eval_*.json "$REPO/results/" 2>/dev/null || true
  cp -v "$CHECKPOINT_DIR"/eval_summary.csv "$REPO/results/" 2>/dev/null || true
  echo "Copied eval JSON + CSV → $REPO/results/"
fi
