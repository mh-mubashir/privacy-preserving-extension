#!/bin/bash
#SBATCH --job-name=residualvae_freeze_clf
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -e

echo "=== ResidualVAE  --freeze_clf  alternating-optimisation run ==="
echo "Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

source ~/.bashrc
conda activate privacy 2>/dev/null || true

python adversarial_training.py \
  --encoder         residual_vae \
  --num_epochs      13 \
  --warmup_epochs   3 \
  --lambda_clf      2.0 \
  --vae_weight      0.05 \
  --vae_beta        1.0 \
  --latent_dim      256 \
  --freeze_clf \
  --data_source     huggingface \
  --max_train_samples 20000 \
  --max_val_samples   2000 \
  --max_test_samples  2000 \
  --batch_size      64 \
  --num_workers     4 \
  --exp_name        member1_residualvae_freeze_clf

echo "End: $(date)"
