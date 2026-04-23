# scripts/

SLURM sbatch wrappers for training and evaluation on the
Northeastern Explorer HPC cluster (partition: `courses-gpu`, 1x GPU).

All scripts set `REPO_DIR` from `$HOME` and activate the `.venv` virtualenv.
Run them from the repo root:

```bash
sbatch scripts/<script_name>.sh
```

---

## Training scripts

### VanillaVAE

| Script | Config |
|---|---|
| `run_vanillavae_full.sh` | Baseline: lambda=1, 10 epochs |
| `run_vanillavae_lambda3.sh` | Lambda sweep: lambda=3, 10 epochs |
| `run_vanillavae_phase2.sh` | Two-phase: warmup=3 + 10 ARL epochs, lambda=2 |
| `run_vanillavae_phase2_lam2.sh` | Two-phase, lambda=2 |
| `run_vanillavae_freeze_clf.sh` | Two-phase + `--freeze_utility_clf_arl` |

### BetaVAE

| Script | Config |
|---|---|
| `run_betavae_full.sh` | Baseline: lambda=1, beta=4, 10 epochs |
| `run_betavae_lambda3.sh` | Lambda sweep: lambda=3 |
| `run_betavae_phase2.sh` | Two-phase: warmup=3 + 10 ARL epochs |
| `run_betavae_phase2_lam2.sh` | Two-phase, lambda=2 |

### ResidualVAE

| Script | Config |
|---|---|
| `run_residualvae_full.sh` | Baseline: lambda=1, 10 epochs |
| `run_residualvae_phase2.sh` | Two-phase: warmup=3 + 10 ARL epochs |
| `run_residualvae_phase2_lam2.sh` | Two-phase, lambda=2 |
| `run_residualvae_phase2_lam5.sh` | Two-phase, lambda=5 (over-penalisation check) |
| `run_residualvae_bottleneck.sh` | IB ablation: latent_dim=32, lambda=2 |
| `run_residualvae_ib64.sh` | IB ablation: latent_dim=64, lambda=2 |
| `run_residualvae_latentadv.sh` | Latent-space adversary (`--latent_adv`) |
| `run_residualvae_freeze_clf.sh` | Two-phase + `--freeze_utility_clf_arl` |
| `run_residualvae_macro_cycle.sh` | Macro-cycle: `--cycle_utility_epochs 1 --cycle_arl_epochs 1` |

### VQ-VAE

| Script | Config |
|---|---|
| `run_vqvae_full.sh` | Member 3 recipe: 64x64, wvae=0.5, optional GRL |

---

## Evaluation scripts

| Script | Description |
|---|---|
| `eval_member1_models.sh` | Evaluate all member1 checkpoints (local runner) |
| `eval_all_privacy_atk.sh` | Full Privacy@K eval across all archived checkpoints |
| `sbatch_eval_member1.sh` | sbatch wrapper for `eval_member1_models.sh` on GPU node |
| `sbatch_eval_all_privacy_atk.sh` | sbatch wrapper for `eval_all_privacy_atk.sh` on GPU node |

GPU eval scripts are needed because the Explorer login node has no CUDA;
inference on 19,962 test images requires a compute node.

---

## Adding a new experiment

Copy the closest existing script and update `--exp_name`, `--encoder`, and any
hyperparameter flags. The checkpoint files (`encoder_model_<exp_name>.pt` etc.)
will land in `$CHECKPOINT_DIR` as defined inside the script.
