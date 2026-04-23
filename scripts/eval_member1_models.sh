#!/bin/bash
# Evaluate Member 1 checkpoints with Malia-style Privacy@K metrics.
# Run from anywhere after: cd ~/privacy-preserving-extension && git pull
#
# Usage on Explorer:
#   Login node has NO GPU → slow CPU eval. Prefer:
#     sbatch scripts/sbatch_eval_member1.sh
#   Or after an interactive GPU (srun ...):
#     bash scripts/eval_member1_models.sh
#
# Override checkpoint dir if yours differs:
#   CHECKPOINT_DIR=/scratch/$USER/pp_ext_member2/arls bash scripts/eval_member1_models.sh

set -e
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate pp_ext_member2_v2
module load cuda/12.1.1

REPO="${REPO:-$HOME/privacy-preserving-extension}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/$USER/pp_ext_member2/arls}"
HF_CACHE="${HF_CACHE:-/scratch/$USER/pp_ext_member2/hf_cache}"

cd "$REPO"
export PYTHONPATH="$REPO"

python evaluate.py \
  --encoder vanilla_vae,beta_vae,residual_vae \
  --exp_name member1_vanillavae_10e_20k,member1_betavae_10e_20k,member1_residualvae_10e_20k \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --data_source huggingface \
  --hf_cache_dir "$HF_CACHE" \
  --device auto \
  --k_levels 70,75,80,85 \
  --latent_dim 256

echo "Done. JSON + eval_summary.csv under: $CHECKPOINT_DIR"
