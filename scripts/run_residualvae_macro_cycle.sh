#!/bin/bash
#SBATCH --job-name=m1_res_cycle
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/pp_ext_member2/logs/residualvae_cycle_%j.out
#SBATCH --error=/scratch/%u/pp_ext_member2/logs/residualvae_cycle_%j.err

# Macro-cycle: after warmup, alternate 1 epoch utility-only with 1 epoch full ARL.
# Compare to baseline without --cycle_* flags.

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate pp_ext_member2_v2
module load cuda/12.1.1

REPO_DIR="${REPO:-$HOME/privacy-preserving-extension}"
OUT_DIR="/scratch/$USER/pp_ext_member2/arls"
mkdir -p "$OUT_DIR" "/scratch/$USER/pp_ext_member2/logs"
cd "$OUT_DIR"
export PYTHONPATH="$REPO_DIR"

echo "== ResidualVAE macro cycle 1+1 after 3-warmup =="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

PYTHONPATH="$REPO_DIR" python "$REPO_DIR/adversarial_training.py" \
  --encoder         residual_vae \
  --data_source     huggingface \
  --hf_cache_dir    "/scratch/$USER/pp_ext_member2/hf_cache" \
  --device          cuda \
  --exp_name        member1_residualvae_cycle1_1_13e_20k \
  --num_epochs      13 \
  --warmup_epochs   3 \
  --cycle_utility_epochs 1 \
  --cycle_arl_epochs       1 \
  --batch_size      16 \
  --max_train_samples 20000 \
  --max_val_samples   2000 \
  --max_test_samples  2000 \
  --vae_weight      0.05 \
  --vae_beta        1.0 \
  --lambda_clf      2.0 \
  --latent_dim      256 \
  --learning_rate_enc 0.0003 \
  --learning_rate_clf 0.001 \
  --learning_rate_adv 0.001 \
  --num_workers     0 \
  2>&1 | tee "$OUT_DIR/member1_residualvae_cycle1_1_13e_20k.log"

echo "End: $(date)"
