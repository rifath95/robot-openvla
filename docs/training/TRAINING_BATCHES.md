# Measure training microbatch size on the A40

For the separate B300/Blackwell image and its compatibility checks, see
[Blackwell training setup](BLACKWELL_TRAINING_SETUP.md).

## Automatic launch and local results

Use this command in a **local Mac terminal inside the workspace**. Paste the SSH
command from RunPod's Connect tab; the launcher selects the Mac's existing key.

```bash
caffeinate -i .venv/bin/python benchmark_cloud_session.py \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519'
```

The pod must already be running with the training image and the network volume.
The default dataset is `/workspace/panda-training/training_bowl_v1`, the new
300-trajectory cube-to-bowl bundle. Download and verify it on the attached volume
first. Older bundles are used only when explicitly selected with `--dataset`;
there is no fallback when the new bundle is missing. The launcher updates the cloud code, runs the adaptive capacity and
speed test, and automatically downloads all saved benchmark report folders from
the volume. Future benchmark candidates pass `--no-checkpoints`: they train the
LoRA adapters for measurement but do not save adapter checkpoint directories.
Metrics, model information, validation results, logs and reports are still saved.
The downloader also excludes adapter weights from older benchmark runs.

The default supervised launcher now starts with a batch-one inference benchmark
of the untouched base OpenVLA model, then runs the training benchmark. It takes one
fixed image/instruction from the selected training bundle, runs **5 warmup requests**
and **20 measured requests** on the cloud, and saves the raw request timings.
While the Mac stays connected, it then sends the same image/instruction through
an SSH tunnel for another **5 warmup + 20 measured requests**. These Mac timings
measure actual request round trips; cloud loopback timing is never substituted.
This test performs no robot movement, IK or local simulator rendering.
When Mac timings are available, reported server averages use those same requests
so the network/processing breakdown stays paired. Both raw sample sets remain saved.
The launcher prints **"Inference measurements saved locally and on the volume.
You can close the lid now; training continues in the cloud."** after the Mac
measurements are persisted. At that point you can disconnect and fetch training
results later. If you stay connected, the launcher watches training and downloads
its reports automatically when finished. Inference runs in its own child process
and exits before training starts, releasing its model weights and CUDA memory.

Both reports are refreshed: `docs/benchmarks/GPU_TRAINING_COMPARISON.md` and
`docs/benchmarks/GPU_INFERENCE_COMPARISON.md`. The inference report begins with
request-count-weighted means grouped by GPU type, followed by individual runs
and their configuration/timing breakdowns. Benchmark
100-step cost estimates cover requests only; full simulator-cycle costs still
come from real simulator trials. Rates must be confirmed separately.

If the Mac disconnects, the cloud still saves its 20 measured predictions and
waits up to **120 seconds** for the optional Mac measurements, then continues
to training. After training it publishes results and allows the bounded download
grace period before stopping.
The same 60-minute outer deadline covers training, inference, and both windows.
Use `--fetch-only` later to retrieve cloud-only measurements. This does not
recreate the missing Mac latency measurements. The legacy `--keep-pod-running`
path runs training benchmarks only, without this supervised inference follow-up.

Normal fine-tuning keeps checkpoint saving enabled by default, including the
5,000-update profile. Each launcher explicitly sets the checkpoint mode for its
child process, so a benchmark's setting cannot carry into a normal training run.
This repository change does not modify a training process already running on a
pod. No Docker rebuild is needed for future benchmarks using updated code.

Local results go into `outputs/gpu-benchmarks/<cloud-run-timestamp>/` along with
`outputs/gpu-benchmarks/GPU_TRAINING_COMPARISON.md`. The top-level workspace
`docs/benchmarks/GPU_TRAINING_COMPARISON.md` is automatically rebuilt too, retaining all downloaded
GPU runs. Per-GPU copies such as `A40_COMPARISON.md` are refreshed in that output
folder too; A100 filenames include the actual variant name reported by CUDA.
It is not necessary to manually copy results or edit the table. Each GPU run
section also includes total benchmark time, outside the summary table. Future
runs use a full benchmark timer; older runs use summed batch-test wall times.

Future runs also record the GPU's current and pending ECC modes through a
read-only `nvidia-smi` query. The comparison table shows current ECC beside the
GPU name; individual report sections include pending mode. The raw query is
saved in `benchmark_metadata.json`. Missing historical measurements show
`Unknown`, and unsupported/unavailable values are not reported as disabled.
No ECC settings are changed. This reporting update needs no Docker rebuild.

Each run also saves `software_stack` in `benchmark_metadata.json` and prints a
software table in both individual and comparison reports: Python, PyTorch,
torchvision, Transformers, tokenizers, PEFT, accelerate, timm, NumPy, TensorFlow,
TFDS, dlimp, Hugging Face Hub, PyTorch CUDA runtime, NVIDIA driver, precision,
attention implementation and benchmark code commit. Actual installed versions
are recorded instead of assuming the template's intended dependency versions.
Historical reports use available saved fields and mark missing values `Unknown`.

The default launcher starts a detached **cloud benchmark supervisor**. Keep the
Mac awake and connected if you want automatic report downloads; disconnecting
the Mac does not cancel the supervisor or its stop attempts. No tmux installation
or Docker rebuild is required for this benchmark supervisor.

The supervisor requests a pod **stop**, not termination, after success, a
benchmark error, or timeout. The Mac tries to download available results first
and also requests a stop even if that download fails. After normal completion or
failure, the cloud supervisor allows **up to two minutes** for this download,
then stops independently if the Mac is absent. The overall deadline takes
precedence over this download window; timed-out results may need fetching later.

### Time limits

- Per batch-size candidate: **10-minute training limit**, **15-minute hard limit**
  including loading, training, validation and saving.
- Entire supervised benchmark: **60-minute outer limit**. It starts when the
  cloud worker starts, after repository checkout and credential preflight; pod
  deployment and earlier idle time are not covered. All candidates and their
  loading/setup are inside this limit. At expiry, the supervisor kills the
  benchmark and its training-worker process groups, then requests a pod stop.
- Stop API requests are retried three times with bounded request timeouts;
  retrying can extend shutdown beyond the benchmark work deadline.

Override the outer limit with `--hard-limit-seconds`, for example `3600` for one
hour. `--keep-pod-running` intentionally disables this cloud supervisor and
automatic stop; per-candidate limits still apply and you must stop manually.

### Required pod environment

The training template must include **`RUNPOD_API_KEY`**, with its value selected
from your existing RunPod stop secret using the secret picker. RunPod normally
supplies **`RUNPOD_POD_ID`** automatically; it must identify the current pod.
Before starting GPU work, the launcher checks that the API identifies this pod
and its attached network volume. If SSH omitted either variable, preflight
recovers only those two variables from `/proc/1/environ`, without printing or
saving credentials. A genuinely absent variable or unresolved secret placeholder
is reported by name; correct the template/pod environment before trying again.
A GET check cannot guarantee stop permission. Invalid permissions, API/network
outages, or a failed supervisor can still prevent stopping; confirm stopped
status in RunPod and stop manually if needed.

### Saved diagnostics and reconnecting

Benchmark measurements remain under
`/workspace/panda-training/batch-benchmarks/<run-timestamp>/`. Supervisor timing,
outcome and logs are stored under
`/workspace/panda-training/batch-benchmarks/sessions/<session-timestamp>/`:
`status.json`, `console.log`, `supervisor.log`, and `pod_stop_status.json`.
The Mac downloads these alongside the benchmark reports. Direct Mac stop
requests also record `batch-benchmarks/pod_stop_status.json`.

If the pod has already stopped, resume it or attach the same volume to a new pod,
then download saved results without rerunning the benchmark:

```bash
.venv/bin/python benchmark_cloud_session.py \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519' \
  --fetch-only --stop-after-fetch
```

`--fetch-only` leaves the pod running unless `--stop-after-fetch` is supplied.
With that flag, a stop is attempted even if fetching fails. If the benchmark is
still running, leave its supervisor to finish before using this stop-after-fetch
command; otherwise it stops the pod and interrupts the active job.

## Automatic batch selection for training

`train_panda.py` defaults to `--batch-policy auto`. It detects the actual GPU and
looks in `/workspace/panda-training/batch-benchmarks` for matching capacity-mode
measurements. Compatibility includes GPU name/VRAM/compute capability, PyTorch,
Transformers, CUDA runtime, processed image shape/action-target format, model
export/config identity, LoRA rank/dropout, quantization, image augmentation, BF16 and eager
attention. The fastest completed candidate passing the headroom screen supplies
the microbatch; accumulation is set to **one** for the initial training baseline.

If there is no matching benchmark, the configured microbatch and accumulation
are retained and the reason is printed. Dataset names, checksum identities and
trajectory counts do not have to match: v1 results can configure v2 when the
GPU, model and training inputs/settings are compatible. Changing image shape,
model or training settings still requires compatible measurements.

The reviewed A40 run `20261011_054257_777818` predates context metadata and is
also supported. Its original candidate `model_info.json` files must remain on
the volume; GPU/VRAM, PyTorch, model path, LoRA settings and candidate batch
settings are checked. Its Transformers 4.40.1/CUDA 12.1/BF16/eager configuration
comes from the known pinned training image. Other legacy runs without context
are excluded. For this reviewed run, the existing base-model path is trusted;
new benchmarks additionally check model export/config identity.
To refuse training rather than fall back, pass `--require-benchmark` to
`train_panda.py`. To use manual settings, pass `--batch-policy config`.

Every training run saves `batch_selection.json`, including the effective batch
and source benchmark when one was selected. Actual batch/accumulation are also
saved in model details and checkpoint configs. Benchmark subprocesses always
use config policy so automatic training selection cannot override the batch
size under test. Selection does not change the learning rate or update count;
decide total data exposure and validation cadence with the chosen effective batch.

## Direct benchmark details

The A40 capacity run is complete; see [GPU_TRAINING_COMPARISON.md](../benchmarks/GPU_TRAINING_COMPARISON.md). The previous training run used microbatch
1 with four accumulation steps: four individual image–instruction–action examples
per parameter update, not four trajectories. Portal VRAM usage was about 27 GiB;
PyTorch peak allocated memory was about 18 GiB. Allocator caching and other GPU
allocations explain why allocated memory alone does not describe available VRAM.

The benchmark compares microbatch/accumulation **1/4, 2/2, 4/1**, keeping effective
batch four. Each candidate starts from the original base model with fresh LoRA
parameters, runs 25 updates, and saves its own logs/checkpoint. Benchmark adapters
are disposable diagnostics, not replacements for the trained adapters. The existing
1000-update training configuration remains unchanged.

After the Hugging Face upload, launch the existing training-image pod and attach
the network volume. Download training_bowl_v1 and verify its checksums first. In the
cloud terminal, inside /root/robot-openvla, with current main checked out, run:

```bash
/opt/openvla/bin/python -m training.benchmark_training_batches \
  --dataset /workspace/panda-training/training_bowl_v1
```

Run this alone on the GPU. Keep the connection open for this foreground benchmark;
it is not the detached training wrapper and **does not automatically stop the pod**.
Stop the pod manually afterward. No Docker rebuild is needed.

Results are saved under `/workspace/panda-training/batch-benchmarks/<timestamp>/`:

- `BATCH_BENCHMARK.md`: comparison table and fastest candidate passing the screen.
- `benchmark.json`: timings, memory measurements, GPU and PyTorch version.
- Candidate logs, configs, metrics, model details and short-run adapter checkpoints.

Throughput excludes the first five updates, final validation, checkpoint saving
and model loading. It includes actual data loading, forward/backward, optimizer,
and normal training logging. Total per-candidate wall time is also recorded.
Only one validation batch is used at the end: these are speed tests, not loss or
accuracy comparisons. Sample count in that validation batch varies with microbatch.

Memory reports include peak allocated, peak reserved, and free VRAM sampled at
completed updates. A candidate must leave at least 4 GiB based on both peak
reserved and sampled free memory before it is suggested. This is a conservative
screen, not a guarantee against OOM on longer sequences or longer training.
OOM candidates are reported; other errors stop the benchmark. Each candidate has
a 15-minute process-group time limit, but pod billing continues after a timeout.

Choose the fastest stable candidate, not the one that fills the most VRAM.
Confirm stability in a longer trial before a full run. If microbatch four is safe
and faster, it can keep the same effective batch using accumulation one. Larger
effective batches (e.g. microbatch four, accumulation two = eight examples) are a
separate learning experiment: the same update count would then process more data.
Do not double the learning rate automatically.

## Adaptive batch fitting

To find the tested memory boundary instead of checking only 1/2/4, run:

```bash
/opt/openvla/bin/python -m training.benchmark_training_batches \
  --find-capacity \
  --dataset /workspace/panda-training/training_bowl_v1
```

This uses the verified bowl bundle on the volume. Its recommendation can be reused for v2 with compatible input/training settings.

The adaptive mode tests 1, 2, 4, 8, ... until CUDA OOM. It then binary-searches
between the last fitting and first failing size (e.g. 4 fits, 8 fails, then test
6 and 7). Each size gets a fresh subprocess/model/optimizer and 25 real updates,
so this tests forward/backward and optimizer memory, not merely loading inputs.
Failed subprocesses release their GPU allocations; an unexpected non-OOM error
stops the search. Run it alone on the GPU.

After finding capacity, speed refinement starts at the largest tested power-of-two
batch that passes the 4 GiB headroom screen. It measures every missing integer size
from that batch up to fitting capacity, reusing all existing measurements. For
example, if 9 fits and 10 fails, but 8 lacks headroom and 4 passes, it tests 5, 6,
and 7. If 8 passes instead, it skips untested sizes below 8. With a 13/14 boundary
and a passing batch 8, it measures missed sizes 9, 10, and 11 (12/13 were already
measured). If no power-of-two batch passes, refinement conservatively starts at 1.
Skipping smaller sizes is a search-cost heuristic; powers of two do not guarantee
better throughput. The recommended batch is the fastest tested candidate passing
the screen. All measured candidates are retained, including OOMs and fast
candidates with insufficient headroom. Accumulation remains one throughout.
The combined report has a top summary table and detailed per-GPU/run sections.

For finite runtime/cost, `--max-batch` bounds the search at 64 by default. If 64
fits, the report states capacity is **at least 64**, not that 64 is the maximum.
The report separates largest fitting batch from fastest candidate passing the
4 GiB headroom screen. It assumes approximately monotonic memory use; short-run
fit results are specific to sampled inputs, model, LoRA and attention settings.

Accumulation is one in this mode: batch size and examples per update both change.
Compare examples/second for hardware efficiency; do not use these loss curves to
compare learning quality. The controlled 1/4, 2/2, 4/1 mode remains available by
omitting `--find-capacity`. No actual training configuration is changed by either
benchmark. GPU measurements are still pending; no capacity number is claimed yet.

Use the same command on an A100 to measure that GPU independently. Reports record
its actual GPU name and VRAM; a batch limit measured on an A40 does not transfer
automatically. Keep the cloud connection open and manually stop the pod when done.

Every benchmark refreshes `/workspace/panda-training/batch-benchmarks/GPU_TRAINING_COMPARISON.md`
from all saved benchmark runs on that volume. The workspace copy is
[GPU_TRAINING_COMPARISON.md](../benchmarks/GPU_TRAINING_COMPARISON.md). Its refresh command and
throughput definition are included in the file. Download benchmark directories
to `outputs/gpu-benchmarks` to regenerate the workspace comparison across GPUs.

## Compare training costs

`docs/benchmarks/GPU_TRAINING_COMPARISON.md` also estimates the time and GPU cost to process
10,000 examples at each run's recommended microbatch. Confirmed hourly rates
are saved per run in `training/gpu_benchmark_rates.json`, so regenerating the
report preserves them. Add a new run's actual rate after confirming it in
RunPod; rates are not guessed or inherited solely from a GPU name. Cost estimates
exclude setup/loading, validation/saving, idle time and storage. They compare
example-processing cost, not equal training quality.

## Benchmark on global storage

Use the explicit `--global-volume` mode described in
[GLOBAL_BENCHMARK_SETUP.md](../benchmarks/GLOBAL_BENCHMARK_SETUP.md). It stages inputs and keeps
locks on container disk, publishes reports to global storage, and retains the
detached 60-minute watchdog and automatic stop behavior. Regional network-volume
benchmark commands remain unchanged.
