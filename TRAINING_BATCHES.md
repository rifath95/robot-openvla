# Measure training microbatch size on the A40

Prepared locally; GPU measurements are pending. The previous run used microbatch
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
the network volume. Download training_v2 and verify its checksums first. In the
cloud terminal, inside /root/robot-openvla, with current main checked out, run:

```bash
/opt/openvla/bin/python benchmark_training_batches.py \
  --dataset /workspace/panda-training/training_v2
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
/opt/openvla/bin/python benchmark_training_batches.py \
  --find-capacity \
  --dataset /workspace/panda-training/training_v1
```

This can run on the existing v1 volume copy while the Mac uploads v2 to Hugging
Face. Confirm the selected batch on v2 before its full training run.

The adaptive mode tests 1, 2, 4, 8, ... until CUDA OOM. It then binary-searches
between the last fitting and first failing size (e.g. 4 fits, 8 fails, then test
6 and 7). Each size gets a fresh subprocess/model/optimizer and 25 real updates,
so this tests forward/backward and optimizer memory, not merely loading inputs.
Failed subprocesses release their GPU allocations; an unexpected non-OOM error
stops the search. Run it alone on the GPU.

After finding capacity, it times all missed fitting integer sizes in the final
doubling interval. For example, if 8 fits and 16 fails and the search narrows the
boundary to 13/14, it then measures missed sizes 9, 10, and 11 (12/13 were already
measured). It does not retest failed sizes 14–16. The recommended batch is the
fastest tested candidate that passes the headroom screen. All measured candidates
are retained, including OOMs and fast candidates with insufficient headroom.
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

Every benchmark refreshes `/workspace/panda-training/batch-benchmarks/GPU_BATCH_COMPARISON.md`
from all saved benchmark runs on that volume. The workspace copy is
[GPU_BATCH_COMPARISON.md](GPU_BATCH_COMPARISON.md). Its refresh command and
throughput definition are included in the file. Download benchmark directories
to `outputs/gpu-benchmarks` to regenerate the workspace comparison across GPUs.
