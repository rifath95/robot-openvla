# Cloud GPU inference setup

Prepared on 2026-10-07. The Mac three-action loop passed; **Linux/CUDA execution has not been tested on a rented GPU yet**. This guide runs both the simulator and OpenVLA on the cloud machine. A Mac-to-cloud inference server and fine-tuning pipeline are separate future steps.

## Machine to use

Start with a Linux x86_64 GPU instance providing one NVIDIA **A100 80GB**, Python 3.11, and a working NVIDIA driver compatible with CUDA 12.1. Choose an image with GPU access and EGL/OpenGL support for headless rendering. For containers, NVIDIA compute and graphics capabilities must be exposed (typically `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`).

Allow approximately 100 GB of persistent disk for environments, robot assets, the 15 GB model, and results; this is an initial inference workspace allowance, not a training-dataset budget. An initial host RAM target of at least 32 GB gives room for setup and loading. Storage and uptime billing depend on the provider.

This guide uses eager attention for a first comparable baseline. CUDA uses bfloat16 on compatible GPUs and otherwise float16. FlashAttention optimization is not enabled yet, and no prediction-speed guarantee is made.

## 1. Get the current code

First push the new code to GitHub from your Mac so the cloud clone includes these changes. Then, on the cloud machine:

```bash
git clone --recurse-submodules https://github.com/rifath95/robot-openvla.git
cd robot-openvla
nvidia-smi
python3.11 --version
```

If the code is already cloned, pull the latest changes and run `git submodule update --init --recursive`. `nvidia-smi` must show the GPU; a CUDA version shown there describes driver compatibility, not the installed PyTorch wheel.

## 2. Install the two Python environments

Use a cloud image with Python 3.11 and venv support already available. The simulation requirements here use NumPy 1.26.4 for Python 3.11 rather than the Mac's Python 3.14 simulation packages.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-sim-linux.txt

python3.11 -m venv .venv-openvla
.venv-openvla/bin/python -m pip install --upgrade pip
.venv-openvla/bin/python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
.venv-openvla/bin/python -m pip install -r requirements-openvla.txt
.venv-openvla/bin/python -m pip check
```

The CUDA wheel is installed first from PyTorch's official index; the shared requirements keep the compatible installed version. Do not copy the Mac's virtual environments to Linux. [Official PyTorch 2.2.2 installation commands](https://pytorch.org/get-started/previous-versions/#v222)

## 3. Check GPU inference and headless rendering

```bash
.venv-openvla/bin/python openvla_mac.py --preflight --device cuda
MUJOCO_GL=egl .venv/bin/python capture_camera.py
```

The preflight prints the GPU name, memory, and precision and tests a GPU matrix multiplication without loading the model. The capture should save `camera_image.png` showing the arm, table, and cube. Download and inspect that image before inference.

If CUDA is unavailable, check GPU exposure, the NVIDIA driver, and the CUDA PyTorch installation. If rendering fails, check that NVIDIA EGL libraries and graphics capabilities are exposed by the VM/container. Do not change the robot scene to work around a graphics-driver problem.

## 4. Download the checkpoint on this machine

```bash
.venv-openvla/bin/python openvla_mac.py --download
```

This downloads weights into the local `.cache-openvla/` directory and writes a manifest containing this machine's snapshot path. The download stage does not need an Apple GPU. Weights are not stored in GitHub. Persist the cache to avoid downloading it again when an instance restarts.

## 5. Run the same three-action benchmark

```bash
.venv/bin/python openvla_loop.py --device cuda
```

The launcher checks CUDA, sets `MUJOCO_GL=egl` unless already configured, and runs three successive predictions with one model load. Simulation pauses during inference. It preserves the existing 1 cm translation and 0.05 radian rotation bounds and provisional Bridge-to-Panda mapping. No desktop viewer is required.

Inspect `outputs/openvla_loop_<timestamp>/status.json`, `summary.json`, and the images in `step_01/` through `step_03/`. Successful completion means all three predictions and bounded movements finished; it does not mean the cube was picked up.

The Mac baseline recorded approximately **462, 601, and 554 seconds per prediction**, approximately **27.5 minutes total**. Compare individual CUDA prediction times and total elapsed time, including model loading and simulation. The first prediction may have startup overhead. Device and precision are recorded in each `prediction.json`; CUDA bfloat16 results need not exactly match Mac float16 results.

Copy results back to your Mac before stopping the instance. Stop the GPU instance when the experiment finishes and check the provider's persistent-storage charges.

## After the benchmark

If latency is useful, the next work is validating the action conventions and preparing demonstrations for domain adaptation. The official OpenVLA LoRA example uses one A100 80GB. Training dependencies, datasets, and a training launcher are **not included in this inference setup**. [OpenVLA fine-tuning instructions](https://github.com/openvla/openvla#fine-tuning-openvla-via-lora)
