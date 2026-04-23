#!/bin/bash
#SBATCH --job-name=eval_m1
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/pp_ext_member2/logs/eval_member1_%j.out
#SBATCH --error=/scratch/%u/pp_ext_member2/logs/eval_member1_%j.err

set -e
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate pp_ext_member2_v2
module load cuda/12.1.1

REPO="${REPO:-$HOME/privacy-preserving-extension}"
mkdir -p "/scratch/$USER/pp_ext_member2/logs"
cd "$REPO"
export PYTHONPATH="$REPO"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/$USER/pp_ext_member2/arls}"
HF_CACHE="${HF_CACHE:-/scratch/$USER/pp_ext_member2/hf_cache}"

echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo n/a)"
bash scripts/eval_member1_models.sh

echo "Finished: $(date)"
