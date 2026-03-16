### Pythae VAE Training on CelebA (DisentangledBetaVAE, BetaTCVAE, FactorVAE)

This document summarizes how we trained two Pythae VAE variants on CelebA, plus how to run a third (FactorVAE) using the same training script.

---

## Data, Environment, and Access Pattern

- **Dataset**
  - **CelebA** face dataset, using the **Hugging Face** hub dataset `flwrlabs/celeba`.
  - Splits: `train` and `valid` used as Pythae’s train/eval data.
  - Images are resized to **64×64**, converted to **3‑channel RGB**, and normalized to values in \([0, 1]\).

- **Data loading**
  - A custom `CelebAHFDataset` wraps the HF dataset:
    - Each `__getitem__` returns **only the image tensor**, converted to RGB, resized to 64×64, and turned into a PyTorch tensor with shape `(3, 64, 64)`.
  - For torchvision CelebA (optional path), we build loaders that return `(images, attrs)` and wrap them in `Subset` for `max_train_samples` / `max_val_samples`.
  - In `pythae_training.py` we normalize both styles to tensors of images before feeding Pythae:
    - If a batch is `(images, attrs)`, we take `batch[0]`.
    - If a batch is already a tensor, we use it as‑is.
    - All batches are concatenated into `train_data` and `eval_data` tensors of shape `(N, 3, 64, 64)`.

- **Environment / GPU**
  - Python **3.12** venv: **`.venv312_cu128`**.
  - PyTorch (`torch`, `torchvision`, `torchaudio`) built for **CUDA 12.8**, compatible with an **RTX 5080 Laptop GPU (sm_120)**.
  - The environment and Windows/NVIDIA settings are configured so `torch.cuda.is_available() == True` and CUDA is used reliably.
  - VRAM budget is roughly **≤ 5 GB**, so batch sizes were tuned accordingly.

---

## Training Script and How We Run Pythae

- **Script**: `pythae_training.py`

- **Core flow**
  1. Parse CLI arguments (variant, data source, image size, latent dim, β/γ, batch size, epochs, LR, etc.).
  2. Build **train** and **validation** dataloaders from either:
     - Hugging Face (`--data_source huggingface`), or
     - Torchvision (`--data_source torchvision`, using `--data_dir`).
  3. Construct the **Pythae model**:
     - `DisentangledBetaVAE` with `DisentangledBetaVAEConfig`, or
     - `BetaTCVAE` with `BetaTCVAEConfig`.
     - `FactorVAE` with `FactorVAEConfig`.
  4. Create a `BaseTrainerConfig` and a `TrainingPipeline`.
  5. Convert the dataloaders into large tensors `train_data` / `eval_data` and call the pipeline:
     - `pipeline(train_data=train_data, eval_data=val_data)`.
  6. Pythae runs the full training loop, logging epoch‑wise losses and saving:
     - A **training directory** under `pythae_runs/<variant>/...`
     - A **`final_model/`** folder with:
       - `training_config.json`
       - `model_config.json`
       - `environment.json`
       - Saved weights.

- **Example commands** (from project root, using `.venv312_cu128`)

  - **DisentangledBetaVAE** (final successful run):

    ```powershell
    .\.venv312_cu128\Scripts\python pythae_training.py `
      --variant disentangled_betavae `
      --data_source huggingface `
      --hf_cache_dir ./.hf_cache `
      --img_size 64 `
      --latent_dim 32 `
      --beta 4.0 `
      --batch_size 48 `
      --num_epochs 50
    ```

  - **BetaTCVAE** (final successful run):

    ```powershell
    .\.venv312_cu128\Scripts\python pythae_training.py `
      --variant betatcvae `
      --data_source huggingface `
      --hf_cache_dir ./.hf_cache `
      --img_size 64 `
      --latent_dim 32 `
      --beta 2.0 `
      --gamma 5.0 `
      --learning_rate 0.0005 `
      --batch_size 32 `
      --num_epochs 50
    ```

  - **FactorVAE** (run with the same script):

    ```powershell
    .\.venv312_cu128\Scripts\python pythae_training.py `
      --variant factorvae `
      --data_source huggingface `
      --hf_cache_dir ./.hf_cache `
      --img_size 64 `
      --latent_dim 32 `
      --gamma 10.0 `
      --learning_rate 0.0005 `
      --batch_size 32 `
      --num_epochs 50
    ```

---

## Model Variants and Hyperparameters

### DisentangledBetaVAE

- **Config** (`model_config.json`)
  - `input_dim`: `[3, 64, 64]`
  - `latent_dim`: `32`
  - `beta`: `4.0`
  - `C`: `50.0`
  - `warmup_epoch`: `25`
  - Reconstruction loss: **MSE**

- **Trainer config** (`training_config.json`)
  - `per_device_train_batch_size`: `48`
  - `per_device_eval_batch_size`: `48`
  - `num_epochs`: `50`
  - `learning_rate`: `0.001`
  - Optimizer: **Adam**

- **Why these settings?**
  - **β = 4.0**: stronger than a vanilla VAE (β = 1), pushing the posterior closer to the factorized prior and encouraging **disentangled latent factors**.
  - **C and warmup**: C‑targeted KL warmup helps avoid early over‑regularization:
    - Gradually increases the effective KL “budget” so the model first learns decent reconstructions, then tightens the prior.
  - **Batch size 48**: chosen to **fill GPU VRAM** up to ~5 GB without exceeding it, for efficient training on the RTX 5080.

### BetaTCVAE

- **Config** (`model_config.json`)
  - `input_dim`: `[3, 64, 64]`
  - `latent_dim`: `32`
  - `alpha`: `1.0`
  - `beta`: `2.0`
  - `gamma`: `5.0`
  - `use_mss`: `true`
  - Reconstruction loss: **MSE**

- **Trainer config**
  - `per_device_train_batch_size`: `32`
  - `per_device_eval_batch_size`: `32`
  - `num_epochs`: `50`
  - `learning_rate`: `0.0005`
  - Optimizer: **Adam**

- **Why these settings?**
  - We initially tried more aggressive settings (higher β/γ, higher LR) and hit **NaN losses**, typical when the **total correlation (TC) term** becomes unstable.
  - Final stable choice:
    - **β = 2.0** and **γ = 5.0**: still substantial regularization, but not so strong as to explode.
    - **LR = 5e‑4**: smaller learning rate for smoother, stable optimization.
    - **Batch size 32**: fits safely in VRAM and gives enough gradient signal for the more complex TC‑based objective.

### FactorVAE

- **Config (key knobs)**
  - `gamma`: total correlation penalty strength (larger gamma = stronger independence pressure)
  - Reconstruction loss is typically MSE with Pythae defaults

- **Why this variant?**
  - FactorVAE is another disentangling VAE approach that trains a discriminator to estimate and penalize total correlation, encouraging **independent latent factors**.
  - In practice, it often needs more careful tuning (especially `gamma` and `learning_rate`) to remain stable.

---

## What’s Unique About These VAE Variants

- **DisentangledBetaVAE**
  - A β‑VAE variant specifically tuned for **disentangled representation learning**.
  - Increases KL weight (`beta > 1`) and uses a **C‑target** schedule:
    - Encourages each latent dimension to capture a distinct, interpretable factor (pose, lighting, expression, etc.).
    - Tends to produce **cleaner latent traversals** than a plain VAE, with moderate impact on reconstruction quality.

- **BetaTCVAE**
  - Decomposes the KL term and explicitly penalizes **total correlation** (TC) of the latent variables.
  - **γ** controls the strength of the TC penalty: higher γ pushes latents towards statistical independence.
  - This often achieves **stronger factorization** than β‑VAE, but:
    - Is more numerically fragile.
    - Usually yields **higher total loss** and slightly worse reconstructions, because it spends capacity on independence constraints.

- **FactorVAE**
  - Targets disentanglement by explicitly penalizing total correlation, using a discriminator-based estimate during training.
  - Similar goal to BetaTCVAE (factorized latents), but with a different training mechanism.

Both models aim for **disentangled latent representations**, but they enforce this via different mechanisms (KL upweighting vs explicit TC penalty).

---

## Training Results

### DisentangledBetaVAE Results

- **Variant**: `disentangled_betavae`

- **Key training stats** (loss = reconstruction + regularization)
  - Epoch 1:
    - Train loss ≈ **261.64**
    - Eval loss ≈ **213.61**
  - Epoch 10:
    - Train loss ≈ **133.37**
    - Eval loss ≈ **133.96**
  - Epoch 20:
    - Train loss ≈ **108.79**
    - Eval loss ≈ **108.61**
  - Final epoch (50/50):
    - Train loss ≈ **101.55**
    - Eval loss ≈ **102.25**

- **Interpretation**
  - Loss decreases **smoothly and monotonically**, then flattens around epochs 25–35.
  - Train and eval losses remain **very close** throughout (final gap ≈ 0.7), which suggests:
    - **Good generalization** on the CelebA validation data at 64×64.
    - No substantial overfitting.
  - The relatively **low final loss** implies:
    - Good reconstruction fidelity for the chosen resolution and latent size.
    - Successful regularization consistent with disentanglement (given β = 4 and C warmup) without collapsing performance.

### BetaTCVAE Results

- **Variant**: `betatcvae`

- **Key training stats**
  - Epoch 1:
    - Train loss ≈ **219.43**
    - Eval loss ≈ **187.40**
  - Epoch 5:
    - Train loss ≈ **170.99**
    - Eval loss ≈ **168.86**
  - Epoch 10:
    - Train loss ≈ **166.71**
    - Eval loss ≈ **167.18**
  - Mid training (~epoch 25–35):
    - Train loss ≈ **164.3–163.9**
    - Eval loss ≈ **164.3–162.8**
  - Final epoch (50/50):
    - Train loss ≈ **163.40**
    - Eval loss ≈ **162.71**

- **Interpretation**
  - After early epochs, the loss decreases more slowly and stabilizes around **163–164**.
  - Train and eval losses stay very close (final gap ≈ 0.7), again showing:
    - **Stable training** after hyperparameter tuning.
    - No marked overfitting.
  - The **higher overall loss** compared to DisentangledBetaVAE is expected:
    - The objective includes a strong **TC penalty** (γ = 5.0).
    - We are intentionally trading off reconstruction quality to enforce **more factorized, independent latent variables**.
  - The crucial point is that training remains **numerically stable** and convergent after dialing down β/γ and the learning rate.

---

## Comparative Interpretation and Effort

- **Numerical fit vs. disentanglement strength**
  - **DisentangledBetaVAE**:
    - Lower final loss (~102) and smooth convergence.
    - Likely **better reconstructions** and still good disentanglement via β and C warmup.
  - **BetaTCVAE**:
    - Higher final loss (~163), reflecting the cost of TC regularization.
    - Expected to produce **more strictly factorized latents**, at some cost in reconstruction quality.

- **Generalization**
  - Both models show **small train–eval gaps** and no overfitting signs:
    - This suggests our data splits, batch sizes, and regularization are reasonable for CelebA at 64×64.

- **Engineering effort and robustness**
  - We:
    - Built a **robust Pythae training pipeline** that works with both Hugging Face and torchvision data backends.
    - Resolved:
      - **Input shape mismatches** between HF and torchvision loaders.
      - **NaN training issues** in BetaTCVAE by principled hyperparameter tuning (β, γ, LR).
    - Tuned batch sizes and device handling to **respect VRAM limits** on an RTX 50‑series GPU.
  - End result: a **reusable, documented setup** for running advanced VAE variants on CelebA, with clear logs and saved final models for downstream use (e.g., as encoders in the privacy‑preserving adversarial framework).

