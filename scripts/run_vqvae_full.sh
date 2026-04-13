#!/bin/bash
#SBATCH --job-name=m3_vqvae
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/pp_ext_member2/logs/vqvae_%j.out
#SBATCH --error=/scratch/%u/pp_ext_member2/logs/vqvae_%j.err

# VQ-VAE full ARL training
# Uses discrete codebook quantisation; reconstruction loss is MSE only (no KL term)

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate pp_ext_member2_v2
module load cuda/12.1.1

REPO_DIR="/courses/EECE5698.202630/students/sureshkumar.si/pp_ext_member2/privacy-preserving-extension"
OUT_DIR="/scratch/$USER/pp_ext_member2/arls"
mkdir -p "$OUT_DIR" "/scratch/$USER/pp_ext_member2/logs"
cd "$OUT_DIR"

echo "== VQ-VAE Full ARL Training =="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

PYTHONPATH="$REPO_DIR" python "$REPO_DIR/adversarial_training.py" \
  --encoder vq_vae \
  --data_source huggingface \
  --hf_cache_dir /scratch/$USER/pp_ext_member2/hf_cache \
  --device cuda \
  --exp_name member3_vqvae_10e_20k \
  --num_epochs 10 \
  --batch_size 16 \
  --max_train_samples 20000 \
  --max_val_samples 2000 \
  --max_test_samples 2000 \
  --vae_weight 0.1 \
  --vae_beta 1.0 \
  --lambda_clf 1.0 \
  --learning_rate_enc 0.0003 \
  --learning_rate_clf 0.001 \
  --learning_rate_adv 0.001 \
  --num_workers 1 \
  2>&1 | tee "$OUT_DIR/member3_vqvae_10e_20k.log"

echo "End: $(date)"
