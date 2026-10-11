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
