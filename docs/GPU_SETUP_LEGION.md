# GPU Setup on Lenovo Legion 7 (NVIDIA RTX)

This guide documents **how the discrete NVIDIA GPU was enabled** for this project on a Lenovo Legion 7 laptop and how to keep it working. The setup is **working**: PyTorch in the project’s Python 3.12 environments reports `torch.cuda.is_available() == True` and uses the **NVIDIA GeForce RTX 5080 Laptop GPU**.

---

## Setup status and how it was done

| Item | What was done |
|------|----------------|
| **Environment (legacy)** | A venv **`.venv312`** was created with **Python 3.12** (separate from the existing Python 3.14 venv) so that PyTorch GPU wheels are available. This env uses a CUDA 12.6 build compatible with the driver but shows a Blackwell (sm_120) warning. |
| **Environment (recommended)** | A dedicated venv **`.venv312_cu128`** was created with **Python 3.12** and PyTorch built for **CUDA 12.8**, which is the recommended combo for RTX 50-series (Blackwell) GPUs. |
| **PyTorch (legacy)** | In `.venv312`, **PyTorch 2.10.0+cu126** and **torchvision 0.25.0+cu126** were installed from `https://download.pytorch.org/whl/cu126`. This matches the driver’s CUDA 12.9 but does not fully target sm_120 (RTX 5080) and can emit a compatibility warning. |
| **PyTorch (recommended)** | In `.venv312_cu128`, **PyTorch 2.10.0+cu128**, **torchvision 0.25.0+cu128**, and **torchaudio 2.10.0+cu128** were installed via `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`. This uses CUDA 12.8 (reported by PyTorch as 12.8) and is designed for Blackwell / sm_120 GPUs while keeping `torch.cuda.is_available() == True`. |
| **Project deps** | Core project dependencies (`numpy`, `wandb`, etc.) were installed into both envs. `.venv312_cu128` is the primary environment for training going forward. |
| **Verification** | Running `.\\.venv312_cu128\Scripts\python.exe -c "import torch; print(torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` returns CUDA `12.8`, `True`, and the RTX 5080 device name. A small CUDA matmul (`1024x1024` on `device='cuda'`) also succeeds. |
| **Legion / Windows** | To ensure Python uses the dGPU, GPU mode can be set in **Lenovo Vantage** (dGPU or Hybrid), and the project’s **python.exe** (from `.venv312_cu128\Scripts`) can be set to **High performance** in Windows **Settings → System → Display → Graphics** and in **NVIDIA Control Panel → Manage 3D settings**. |

The sections below give step-by-step instructions for **Lenovo Vantage**, **Windows Graphics**, and **NVIDIA Control Panel** so you can confirm or adjust the setup, and they document **RTX 50-series** compatibility notes.

---

## 1. Choose GPU mode (Lenovo Vantage)

**Lenovo Vantage** is the main way to set or change GPU mode on the Legion.

1. Open **Lenovo Vantage** (pre-installed; search in Start if needed).
2. Go to **Hardware Settings** (or **Device** / **Power**) and find **GPU mode** or **Graphics mode**.
3. Set one of:
   - **dGPU only / Discrete graphics** – laptop always uses the NVIDIA GPU (best for ML/training; uses more power).
   - **Hybrid / Auto** – system switches between iGPU and dGPU; ensure Python is set to use the dGPU (see steps 2 and 3 below).

If you don’t see GPU mode in Vantage, check for a **Lenovo Legion** or **Legion Edge** app; some models use that for power/GPU settings.

---

## 2. Prefer NVIDIA GPU for Python (Windows)

So that **Python** (and thus PyTorch) uses the NVIDIA GPU when in Hybrid mode:

1. Open **Settings** → **System** → **Display**.
2. Scroll to **Graphics** (or search “Graphics settings”).
3. Click **Add an app** → **Desktop app**.
4. Browse to your **Python executable**:
   - For this project’s venv:  
     `c:\Users\<You>\Documents\coding\privacy-preserving-extension\.venv312\Scripts\python.exe`
   - Or the global Python if you use that: e.g.  
     `C:\Users\<You>\AppData\Local\Programs\Python\Python312\python.exe`
5. Add it, then click **Options** for that app.
6. Select **High performance** (NVIDIA GPU) and save.

---

## 3. NVIDIA Control Panel (optional but recommended)

1. Right‑click the desktop → **NVIDIA Control Panel**.
2. Go to **Manage 3D settings** (under “3D Settings”).
3. Open the **Program Settings** tab.
4. Add the same **python.exe** as above (from `.venv312\Scripts` or your global Python).
5. Set **Preferred graphics processor** to **High-performance NVIDIA processor**.
6. Apply.

You can also set **Global settings** → **Preferred graphics processor** to **High-performance NVIDIA processor** so all apps default to the dGPU (handy if you use dGPU-only mode).

---

## 4. BIOS (if GPU mode isn’t in Vantage)

On some Legion models, GPU mode is in BIOS instead of (or in addition to) Vantage:

1. Restart and press **F2** (or the key shown at boot) to enter BIOS.
2. Go to **Configuration** or **Advanced** and find **Graphics** / **GPU mode**.
3. Set to **Discrete** or **dGPU only** if you want the NVIDIA GPU always on.
4. Save and exit (often **F10**).

---

## 5. This project’s working setups: CUDA 12.6 and CUDA 12.8

There are two Python 3.12 environments that can run this project on GPU:

- **Legacy (cu126) – `.venv312`**
  - **Venv path:** `.venv312` in the project root.
  - **PyTorch:** `torch 2.10.0+cu126`, `torchvision 0.25.0+cu126`.
  - **Activate (PowerShell):**
    ```powershell
    .\.venv312\Scripts\Activate.ps1
    ```
  - **Verify GPU:**
    ```powershell
    .\.venv312\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
    ```
  - This env was the first working GPU setup (CUDA 12.6 wheels on a CUDA 12.9 driver). It may show a **Blackwell / sm_120 warning** but still runs training.

- **Recommended (cu128) – `.venv312_cu128`**
  - **Venv path:** `.venv312_cu128` in the project root.
  - **PyTorch:** `torch 2.10.0+cu128`, `torchvision 0.25.0+cu128`, `torchaudio 2.10.0+cu128` installed via:
    ```powershell
    .\.venv312_cu128\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ```
  - **Activate (PowerShell):**
    ```powershell
    .\.venv312_cu128\Scripts\Activate.ps1
    ```
  - **Verify GPU (expected: `cuda_available: True` and device name):**
    ```powershell
    .\.venv312_cu128\Scripts\python.exe -c "import torch; print('CUDA version:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
    ```
  - This is the **primary environment** going forward: it uses CUDA 12.8 wheels, is designed for RTX 50‑series / sm_120 GPUs, and has been tested to perform a nontrivial CUDA matmul successfully.

If you ever see `CUDA: False` in either env:

- Confirm **nvidia-smi** in a terminal shows your GPU and driver.
- Redo **steps 1–3** (Vantage GPU mode, Windows Graphics, NVIDIA Control Panel) and restart.
- Ensure you’re running the **same** `python.exe` you added in Graphics settings (e.g. from `.venv312`).

---

## 6. Drivers

Keep the NVIDIA driver up to date:

- **NVIDIA GeForce Experience** (optional): can notify you of driver updates.
- Or download the latest driver for your GPU from [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx).

Your driver version must support at least the CUDA version used by PyTorch:

- `.venv312` (cu126) expects a CUDA 12.6‑compatible driver (drivers that report CUDA 12.x in `nvidia-smi` are fine).
- `.venv312_cu128` (cu128) expects a CUDA 12.8‑compatible driver (your Legion currently reports CUDA 12.9 in `nvidia-smi`, which is sufficient).

---

## 7. RTX 50-series (Blackwell, e.g. RTX 5080) and PyTorch

If you have an **RTX 5080** or other Blackwell GPU, older PyTorch wheels can show a warning that the GPU has **CUDA capability 12.0 (sm_120)** while the wheel only targets up to sm_90. In our setup:

- **Legacy setup (cu126):** In `.venv312`, `torch.cuda.is_available()` is `True` and the RTX 5080 is detected; training and typical PyTorch ops run, but a sm_120 warning can appear.
- **Recommended setup (cu128):** In `.venv312_cu128`, a CUDA 12.8 wheel is used, which is intended for RTX 50‑series / sm_120 GPUs. With a CUDA 12.9 driver, this is the most future‑proof choice without needing to build PyTorch from source.

---

**Summary:** The GPU setup for this project is complete and working. Use the **`.venv312_cu128`** interpreter for day‑to‑day training; PyTorch will use the NVIDIA RTX 5080 via CUDA 12.8. The `.venv312` env remains as a legacy fallback. The steps in sections 1–4 are optional tweaks to ensure the Legion consistently uses the dGPU for Python; section 5 documents both envs and the verification commands that confirm the setup.
qq