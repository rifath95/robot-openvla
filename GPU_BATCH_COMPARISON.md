# GPU batch capacity and training throughput

This table collects measured OpenVLA LoRA batch benchmarks. Each row is one run; different datasets or settings remain separate.

**Throughput** means training examples processed per second after warmup. One example is one camera image, its task instruction, and seven target action values—not one trajectory.
**Examples/second = effective batch ÷ seconds per completed parameter update.** Effective batch = microbatch × gradient accumulation (single GPU).
Measured update time includes data loading, image/text processing, forward/backward, optimizer, and ordinary training logging. Model loading, final validation, and checkpoint saving are excluded.

The speed columns describe the fastest tested candidate passing the 4 GiB memory-headroom screen. That candidate may differ from the largest fitting batch.
A fitting batch completed 25 real updates. The short-run capacity applies to tested inputs/settings; confirm it on the final dataset and a longer trial.

| GPU | VRAM GiB | Dataset | Mode | Fitting capacity | First OOM | Suggested microbatch / accumulation | Seconds/update | Examples/s | Peak allocated / reserved GiB | Run / details |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NVIDIA A40 | 47.4 | training_v1 | capacity | 9 (boundary found) | 10 | 8 / 1 | 2.140 | 3.74 | 40.25 / 42.71 | [20261011_054257_777818](outputs/gpu-benchmarks/A40_COMPARISON.md#run-20261011_054257_777818) |

## NVIDIA A40 — 47.4 GiB

### Run 20261011_054257_777818

Dataset: `/workspace/panda-training/training_v1`. Model: `/workspace/openvla-7b`.
Mode: capacity; PyTorch: 2.2.2+cu121; LoRA rank: 32; quantization: False; image augmentation: False.
Training uses BF16 autocast and eager attention. See the saved candidate configs for all settings.
Capacity finding: **9 (boundary found)**.
Fastest candidate with headroom: **microbatch 8, accumulation 1**, 3.74 examples/s.

| Microbatch | Accumulation | Phase | Status | Seconds/update | Examples/s | Peak allocated / reserved GiB | Headroom screen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | capacity | completed | 0.416 | 2.41 | 18.01 / 26.54 | True |
| 2 | 1 | capacity | completed | 0.632 | 3.16 | 21.25 / 25.71 | True |
| 4 | 1 | capacity | completed | 1.085 | 3.69 | 27.57 / 28.31 | True |
| 8 | 1 | capacity | completed | 2.140 | 3.74 | 40.25 / 42.71 | True |
| 9 | 1 | capacity | completed | 2.282 | 3.94 | 43.33 / 46.33 | False |
| 10 | 1 | capacity | out_of_memory | — | — | — | — |
| 12 | 1 | capacity | out_of_memory | — | — | — | — |
| 16 | 1 | capacity | out_of_memory | — | — | — | — |

## Reading the results

- “Boundary found” means adjacent sizes were tested: N fitted and N+1 ran out of memory. Intermediate untested sizes are inferred using approximately monotonic memory demand.
- “Boundary not established” means the search is incomplete or its ceiling fitted. It is a lower bound, not a measured maximum.
- Suggested batches leave at least 4 GiB by peak reserved and sampled free VRAM. This is a screening rule, not a stability guarantee.
- Capacity mode uses accumulation one; effective batch changes. Controlled mode keeps effective batch four. These speed tests do not compare learning quality.
- After capacity search, missed fitting sizes in the final doubling interval are timed too. The recommendation is the fastest tested candidate with headroom; smaller sizes outside that interval are not exhaustively scanned.
- Compare GPUs with the same dataset, model, LoRA rank, precision and attention settings. A100 40 GB and 80 GB variants should be recorded as separate runs.
- Each linked report has per-batch timings and memory. Its sibling benchmark_metadata.json, benchmark.json, and model_info.json inside candidate folders preserve the exact settings and GPU details.

## Updating this file

Use the local launcher for automatic cloud execution, report download and workspace refresh:

```bash
.venv/bin/python benchmark_cloud_session.py \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519'
```

Get POD_IP and POD_PORT from RunPod → Pod → Connect → SSH over exposed TCP. The launcher selects this Mac's existing private key.
Keep the Mac awake and connected. This command does not stop the rented pod. Stop it in RunPod after synchronization.
For an existing run or an interrupted download, use the same command with `--fetch-only`; it does not run training.
The download includes all saved benchmark reports, JSON measurements, configs and logs on the attached volume; adapter weights are excluded.
Timestamped report folders and a combined report are saved in outputs/gpu-benchmarks. This workspace file is rebuilt automatically from the downloaded JSON.
For report folders already downloaded, the standalone refresh command remains available:

```bash
.venv-training/bin/python summarize_gpu_benchmarks.py \
  --root outputs/gpu-benchmarks \
  --output GPU_BATCH_COMPARISON.md
```

Historical training memory usage alone does not establish maximum batch capacity.
