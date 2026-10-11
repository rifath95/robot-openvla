# Blackwell training and benchmark image

This is a separate candidate image for GPUs such as the B300. It does not replace
the existing training/inference images or change an already running training job.
The image must pass its build checks and then be tested on the actual GPU before
we treat it as a validated OpenVLA setup.

## Dependency versions

| Component | Candidate version |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.14.1, CUDA 13.0 wheels |
| torchvision | 0.29.1, CUDA 13.0 wheels |
| Transformers / tokenizers | 4.40.1 / 0.19.1 |
| PEFT / timm | 0.11.1 / 0.9.10 |
| TensorFlow / TensorFlow Datasets | 2.15.0 / 4.9.3 |
| NumPy | 1.26.4 |

The GPU runtime is upgraded. The model and dataset APIs retain the versions
specified by [OpenVLA's dependency configuration](https://github.com/openvla/openvla/blob/main/pyproject.toml).
Upgrading all of those APIs together would require a separate migration and
validation of OpenVLA's custom model code. The two requirements files for this
candidate are `requirements/requirements-openvla-blackwell.txt` and
`requirements/requirements-training-blackwell.txt`.

The Dockerfile reuses the existing RunPod base for its SSH startup. Its CUDA 12.4
base label does not describe the runtime used by `/opt/openvla/bin/python`:
that environment installs CUDA 13.0 PyTorch wheels. The host must provide a
compatible NVIDIA driver. The GPU check verifies the actual runtime and executes
CUDA operations; leave ECC enabled if the GPU provides it.

## Build before renting a GPU

1. Push the code, then open GitHub → Actions → **Build Blackwell training image**
   → **Run workflow** on `main`.
2. Wait for a successful build. During the build, `pip check` and a CPU dependency
   smoke test run. The latter checks image processing, torchvision operations,
   TensorFlow and a tiny Llama LoRA forward/backward/optimizer update.
3. Open the new GitHub Container Registry package and make it public if necessary.
   Copy its image reference, without the `docker pull` prefix:

   ```text
   ghcr.io/rifath95/robot-openvla-training-blackwell:COMMIT_SHA
   ```

   Get `COMMIT_SHA` from the successful workflow commit or the package's published
   tag. Use that exact tag rather than `latest`.
4. Duplicate the existing training template and set its container image to that
   reference. Retain TCP port 22, the SSH public key, `HF_TOKEN` read secret and
   `RUNPOD_API_KEY` stop secret. Attach **robot-openvla-global-volume** at
   `/workspace` when deploying the benchmark pod. Its existing model export and
   original 50-trajectory dataset can be reused.

## Run from the Mac

In a local terminal inside this repository, replace `POD_IP` and `POD_PORT` with
the values in the new pod's Connect → SSH command. The launcher selects the
configured Mac SSH key as before.

```bash
caffeinate -i .venv/bin/python benchmark_cloud_session.py \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519' \
  --global-volume /workspace \
  --dataset /workspace/panda-training/training_v1 \
  --max-batch 128
```

The higher search ceiling allows a large B300 to test beyond microbatch 64. If
128 still fits, the report gives a tested lower bound on capacity. The existing
60-minute outer deadline still applies, including input staging and all tests.

Before testing capacity, this image runs a tiny CUDA LoRA update, BF16 matrix
operation, image transform and torchvision CUDA operation. A failure aborts the
benchmark and follows the supervisor's stop/diagnostic path. A successful check
is saved as `batch_stack_check.json` with package versions, CUDA runtime, GPU
name, compute capability and supported compiled architectures.

Microbatch 1 then tests the real OpenVLA training path, followed by capacity search
and speed refinement. The tiny check alone does not establish full OpenVLA
compatibility. Benchmark checkpoints remain disabled; model weights and datasets
on the volume are unchanged. Reports and logs are saved and downloaded as usual.

Keep the Mac awake for automatic downloads. If it disconnects, the detached
supervisor continues and saves reports on the volume for later fetching. It
requests a pod stop on completion, failure or timeout; confirm the stopped state
in RunPod because an API/network failure can prevent stopping.

Record this software stack when comparing throughput with older benchmarks:
differences between the CUDA/PyTorch versions can affect speed, so a comparison
across stacks does not isolate the GPU hardware alone.
