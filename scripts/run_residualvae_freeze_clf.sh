#!/bin/bash
#SBATCH --job-name=m1_res_freeze
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/pp_ext_member2/logs/residualvae_freeze_%j.out
#SBATCH --error=/scratch/%u/pp_ext_member2/logs/residualvae_freeze_%j.err

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate pp_ext_member2_v2
module load cuda/12.1.1

REPO_DIR="$HOME/privacy-preserving-extension"
OUT_DIR="/scratch/$USER/pp_ext_member2/arls"
mkdir -p "$OUT_DIR" "/scratch/$USER/pp_ext_member2/logs"
cd "$OUT_DIR"

echo "== ResidualVAE --freeze_clf alternating-optimisation =="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

PYTHONPATH="$REPO_DIR" python "$REPO_DIR/adversarial_training.py" \
  --encoder         residual_vae \
  --data_source     huggingface \
  --hf_cache_dir    /scratch/$USER/pp_ext_member2/hf_cache \
  --device          cuda \
  --exp_name        member1_residualvae_freeze_clf \
  --num_epochs      13 \
  --warmup_epochs   3 \
  --batch_size      16 \
  --max_train_samples 20000 \
  --max_val_samples   2000 \
  --max_test_samples  2000 \
  --vae_weight      0.05 \
  --vae_beta        1.0 \
  --lambda_clf      2.0 \
  --latent_dim      256 \
  --freeze_clf \
  --learning_rate_enc 0.0003 \
  --learning_rate_clf 0.001 \
  --learning_rate_adv 0.001 \
  --num_workers     0 \
  2>&1 | tee "$OUT_DIR/member1_residualvae_freeze_clf.log"

echo "End: $(date)"
