#!/bin/bash
#SBATCH --job-name=eval_all_pak
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/pp_ext_member2/logs/eval_all_privacy_atk_%j.out
#SBATCH --error=/scratch/%u/pp_ext_member2/logs/eval_all_privacy_atk_%j.err
# Optional: COPY_TO_REPO=1 sbatch ... copies eval_*.json + eval_summary.csv → $REPO/results/

set -e
mkdir -p "/scratch/$USER/pp_ext_member2/logs"

REPO="${REPO:-$HOME/privacy-preserving-extension}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/$USER/pp_ext_member2/arls}"
HF_CACHE="${HF_CACHE:-/scratch/$USER/pp_ext_member2/hf_cache}"
COPY_TO_REPO="${COPY_TO_REPO:-0}"
export REPO CHECKPOINT_DIR HF_CACHE COPY_TO_REPO K_LEVELS NUM_WORKERS

cd "$REPO"

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
echo "Start: $(date)"

bash scripts/eval_all_privacy_atk.sh

echo "End: $(date)"
