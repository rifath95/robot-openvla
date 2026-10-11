# GPU batch capacity and training throughput

This table collects measured OpenVLA LoRA batch benchmarks. Each row is one run; different datasets or settings remain separate.

**Throughput** means training examples processed per second after warmup. One example is one camera image, its task instruction, and seven target action values—not one trajectory.
**Examples/second = effective batch ÷ seconds per completed parameter update.** Effective batch = microbatch × gradient accumulation (single GPU).
Measured update time includes data loading, image/text processing, forward/backward, optimizer, and ordinary training logging. Model loading, final validation, and checkpoint saving are excluded.

The speed columns describe the fastest tested candidate passing the 4 GiB memory-headroom screen. That candidate may differ from the largest fitting batch.
A fitting batch completed 25 real updates. The short-run capacity applies to tested inputs/settings; confirm it on the final dataset and a longer trial.

| GPU | VRAM GiB | Dataset | Mode | Fitting capacity | First OOM | Suggested microbatch / accumulation | Seconds/update | Examples/s | Peak allocated / reserved GiB | Run / details |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NVIDIA A40 | — | — | Planned | Not measured | — | — | — | — | — | Pending |
| NVIDIA A100 (variant to be recorded) | — | — | Planned | Not measured | — | — | — | — | — | Pending |

## Reading the results

- “Boundary found” means adjacent sizes were tested: N fitted and N+1 ran out of memory. Intermediate untested sizes are inferred using approximately monotonic memory demand.
- “Boundary not established” means the search is incomplete or its ceiling fitted. It is a lower bound, not a measured maximum.
- Suggested batches leave at least 4 GiB by peak reserved and sampled free VRAM. This is a screening rule, not a stability guarantee.
- Capacity mode uses accumulation one; effective batch changes. Controlled mode keeps effective batch four. These speed tests do not compare learning quality.
- Compare GPUs with the same dataset, model, LoRA rank, precision and attention settings. A100 40 GB and 80 GB variants should be recorded as separate runs.
- Each linked report has per-batch timings and memory. Its sibling benchmark_metadata.json, benchmark.json, and model_info.json inside candidate folders preserve the exact settings and GPU details.

## Updating this file

Every cloud benchmark refreshes a comparison file in the network-volume benchmark directory. To refresh this workspace copy after downloading benchmark folders into outputs/gpu-benchmarks:

```bash
.venv-training/bin/python summarize_gpu_benchmarks.py \
  --root outputs/gpu-benchmarks \
  --output GPU_BATCH_COMPARISON.md
```

Historical training memory usage alone does not establish maximum batch capacity.
