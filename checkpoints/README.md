## Checkpoints

This repository includes **all code** and **evaluation summaries** needed to
reproduce results, but we do **not** commit large training checkpoints (model
weights) directly to git by default (see `.gitignore` for `*.pt` / `*.pth`).

### What to use as the “best checkpoint”

- **Best privacy–utility trade-off (post-feedback, alternate protocol)**:
  `member3_vqvae_10e_20k` (see `results/eval_member3_vqvae_10e_20k.json`).
- **Best trade-off within the matched main protocol (224×224 sweep)**:
  ResidualVAE by NAG (see `results/eval_summary.csv` and the report).

### How to obtain weights

You have two options:

1. **Reproduce (recommended)**: run `adversarial_training.py` with the
   experiment name used in the evaluation JSON, then re-run `evaluate.py`.
2. **Store weights separately**: if you want to ship weights, use Git LFS or a
   release artifact and document the download link + hash here.

