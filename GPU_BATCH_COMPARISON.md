# GPU batch capacity and training throughput

This table collects measured OpenVLA LoRA batch benchmarks. Each row is one run; different datasets or settings remain separate.

**Throughput** means training examples processed per second after warmup. One example is one camera image, its task instruction, and seven target action values—not one trajectory.
**Examples/second = effective batch ÷ seconds per completed parameter update.** Effective batch = microbatch × gradient accumulation (single GPU).
Measured update time includes data loading, image/text processing, forward/backward, optimizer, and ordinary training logging. Model loading, final validation, and checkpoint saving are excluded.

The speed columns describe the fastest tested candidate passing the 4 GiB memory-headroom screen. That candidate may differ from the largest fitting batch.
A fitting batch completed 25 real updates. The short-run capacity applies to tested inputs/settings; confirm it on the final dataset and a longer trial.

| GPU | ECC current | VRAM GiB | Dataset | Mode | Fitting capacity | First OOM | Suggested microbatch / accumulation | Seconds/update | Examples/s | Peak allocated / reserved GiB | Run / details |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NVIDIA A40 | Unknown | 47.4 | training_v1 | capacity | 9 (boundary found) | 10 | 8 / 1 | 2.140 | 3.74 | 40.25 / 42.71 | [20261011_054257_777818](outputs/gpu-benchmarks/20261011_054257_777818/BATCH_BENCHMARK.md) |
| NVIDIA H100 80GB HBM3 | Unknown | 79.2 | training_v2 | capacity | 18 (boundary found) | 19 | 16 / 1 | 1.606 | 9.96 | 65.55 / 71.64 | [20261011_071323_382964](outputs/gpu-benchmarks/20261011_071323_382964/BATCH_BENCHMARK.md) |
| NVIDIA RTX A6000 | Unknown | 47.4 | training_v1 | capacity | 9 (boundary found) | 10 | 4 / 1 | 5.295 | 0.76 | 27.59 / 28.42 | [20261011_095911_435567](outputs/gpu-benchmarks/20261011_095911_435567/BATCH_BENCHMARK.md) |

## Cost for the same amount of training

Estimates below process **10,000 training examples**, using the recommended microbatch and measured throughput for each run.
Time = examples ÷ examples/second. Cost = time in hours × hourly GPU rate.
Rates are the user-confirmed prices for these particular pod runs, saved in `training/gpu_benchmark_rates.json`; they are not live quotes and are not assumed for later runs.

| GPU | Run | GPU rate USD/hour | Examples/s | Time for 10,000 examples | Training cost USD |
| --- | --- | --- | --- | --- | --- |
| NVIDIA A40 | 20261011_054257_777818 | $0.59 | 3.74 | 44 min 35 sec | $0.44 |
| NVIDIA H100 80GB HBM3 | 20261011_071323_382964 | $3.99 | 9.96 | 16 min 44 sec | $1.11 |
| NVIDIA RTX A6000 | 20261011_095911_435567 | $0.59 | 0.76 | 220 min 37 sec | $2.17 |

Lowest estimated training cost among these priced runs: **NVIDIA A40**, **$0.44 per 10,000 examples**.
**NVIDIA H100 80GB HBM3** processes examples **2.67× faster** than NVIDIA A40, but costs **2.54× as much per example**.

These are training-only estimates, excluding deployment, model loading, validation, checkpoint saving, idle time, and storage charges. Total billed cost includes those additional items.
Different microbatches/effective batches and datasets were benchmarked. Equal examples processed does not guarantee equal model quality; a convergence comparison should match effective batch and training settings.
Future report refreshes retain the confirmed rates. Record each new run’s actual hourly rate before estimating its cost.

## NVIDIA A40 — 47.4 GiB

### Run 20261011_054257_777818

Dataset: `/workspace/panda-training/training_v1`. Model: `/workspace/openvla-7b`.
Mode: capacity; PyTorch: 2.2.2+cu121; LoRA rank: 32; quantization: False; image augmentation: False.
Total benchmark time: **11 min 42 sec** (sum of batch-test wall times).
GPU ECC: **Unknown** (current); pending mode: **Unknown**.
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

## NVIDIA H100 80GB HBM3 — 79.2 GiB

### Run 20261011_071323_382964

Dataset: `/workspace/panda-training/training_v2`. Model: `/workspace/openvla-7b`.
Mode: capacity; PyTorch: 2.2.2+cu121; LoRA rank: 32; quantization: False; image augmentation: False.
Total benchmark time: **13 min 10 sec**.
GPU ECC: **Unknown** (current); pending mode: **Unknown**.
Training uses BF16 autocast and eager attention. See the saved candidate configs for all settings.
Capacity finding: **18 (boundary found)**.
Fastest candidate with headroom: **microbatch 16, accumulation 1**, 9.96 examples/s.

| Microbatch | Accumulation | Phase | Status | Seconds/update | Examples/s | Peak allocated / reserved GiB | Headroom screen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | capacity | completed | 0.303 | 3.31 | 18.06 / 23.37 | True |
| 2 | 1 | capacity | completed | 0.404 | 4.95 | 21.34 / 23.13 | True |
| 4 | 1 | capacity | completed | 0.469 | 8.52 | 27.65 / 28.65 | True |
| 8 | 1 | capacity | completed | 0.880 | 9.09 | 40.35 / 43.09 | True |
| 16 | 1 | capacity | completed | 1.606 | 9.96 | 65.55 / 71.64 | True |
| 17 | 1 | speed_refinement | completed | 1.744 | 9.75 | 68.76 / 74.37 | False |
| 18 | 1 | capacity | completed | 1.716 | 10.49 | 71.93 / 77.72 | False |
| 19 | 1 | capacity | out_of_memory | — | — | — | — |
| 20 | 1 | capacity | out_of_memory | — | — | — | — |
| 24 | 1 | capacity | out_of_memory | — | — | — | — |
| 32 | 1 | capacity | out_of_memory | — | — | — | — |

## NVIDIA RTX A6000 — 47.4 GiB

### Run 20261011_095911_435567

Dataset: `/root/robot-openvla/.benchmark-local/inputs/training_v1`. Model: `/root/robot-openvla/.benchmark-local/inputs/openvla-7b`.
Mode: capacity; PyTorch: 2.2.2+cu121; LoRA rank: 32; quantization: False; image augmentation: False.
Total benchmark time: **35 min 05 sec**.
GPU ECC: **Unknown** (current); pending mode: **Unknown**.
Training uses BF16 autocast and eager attention. See the saved candidate configs for all settings.
Capacity finding: **9 (boundary found)**.
Fastest candidate with headroom: **microbatch 4, accumulation 1**, 0.76 examples/s.
Input storage: **global_volume** at `/workspace`; training reads staged inputs from **container_disk**.
Input staging: **7.11 seconds**, excluded from total benchmark time above. Model staging read source: `local_pinned_download_cache`.
Pod CPU: AMD EPYC 7543 32-Core Processor; CPU quota (`cpu.max`): `680000 100000`.
CPU and GPU snapshot diagnostics: `batch_host_context.json` in this run’s output folder. Throughput describes this deployment and can depend on host resources and software as well as the GPU.


| Microbatch | Accumulation | Phase | Status | Seconds/update | Examples/s | Peak allocated / reserved GiB | Headroom screen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | capacity | completed | 1.585 | 0.63 | 18.19 / 26.76 | True |
| 2 | 1 | capacity | completed | 2.811 | 0.71 | 21.25 / 26.85 | True |
| 4 | 1 | capacity | completed | 5.295 | 0.76 | 27.59 / 28.42 | True |
| 8 | 1 | capacity | completed | 9.869 | 0.81 | 40.25 / 42.86 | False |
| 9 | 1 | capacity | completed | 11.481 | 0.78 | 43.33 / 46.22 | False |
| 10 | 1 | capacity | out_of_memory | — | — | — | — |
| 12 | 1 | capacity | out_of_memory | — | — | — | — |
| 16 | 1 | capacity | out_of_memory | — | — | — | — |

## Reading the results

- “Boundary found” means adjacent sizes were tested: N fitted and N+1 ran out of memory. Intermediate untested sizes are inferred using approximately monotonic memory demand.
- “Boundary not established” means the search is incomplete or its ceiling fitted. It is a lower bound, not a measured maximum.
- Suggested batches leave at least 4 GiB by peak reserved and sampled free VRAM. This is a screening rule, not a stability guarantee.
- ECC reports the current NVIDIA error-correction mode; pending mode is shown in each run section. Unknown means it was not recorded or could not be queried. Unavailable (N/A) is not treated as disabled. The benchmark does not change ECC settings.
- Capacity mode uses accumulation one; effective batch changes. Controlled mode keeps effective batch four. These speed tests do not compare learning quality.
- After capacity search, speed refinement tests missing integer sizes from the largest tested power-of-two batch passing the headroom screen up to fitting capacity. Earlier runs refined only the final capacity interval and may have missed sizes below a fitting batch without headroom. Skipping smaller sizes is a search-cost heuristic, not a guarantee that powers of two are faster. Recommendations describe the fastest tested candidate with headroom.
- Compare GPUs with the same dataset, model, LoRA rank, precision and attention settings. A100 40 GB and 80 GB variants should be recorded as separate runs.
- Total benchmark time in each run section covers setup/loading and all batch tests, excluding pod deployment, report downloading and the supervisor shutdown window. Older runs use summed batch-test wall times, which omit between-test overhead.
- Each linked report has per-batch timings and memory. Its sibling benchmark_metadata.json, benchmark.json, and model_info.json inside candidate folders preserve the exact settings and GPU details.

## Updating this file

Use the local launcher for automatic cloud execution, report download and workspace refresh:

```bash
.venv/bin/python benchmark_cloud_session.py \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519'
```

Get POD_IP and POD_PORT from RunPod → Pod → Connect → SSH over exposed TCP. The launcher selects this Mac's existing private key.
Keep the Mac awake and connected. By default a detached cloud supervisor requests a pod stop after completion, failure, or its 60-minute outer limit; the Mac downloads reports and also requests a stop. It uses the RUNPOD_API_KEY secret already configured on the pod. Confirm stopped status in RunPod.
Use `--keep-pod-running` to disable that stop. For an existing run or interrupted download, use `--fetch-only`; add `--stop-after-fetch` to stop after retrieval. No benchmark is run in fetch-only mode.
If the Mac disconnects, the cloud supervisor continues and saved reports can be fetched later. If stop requests fail, stop the pod manually in RunPod.
The download includes all saved benchmark reports, JSON measurements, configs and logs on the attached volume; adapter weights are excluded.
Timestamped report folders and a combined report are saved in outputs/gpu-benchmarks. This workspace file is rebuilt automatically from the downloaded JSON.
For report folders already downloaded, the standalone refresh command remains available:

```bash
.venv-training/bin/python summarize_gpu_benchmarks.py \
  --root outputs/gpu-benchmarks \
  --output GPU_BATCH_COMPARISON.md
```

Historical training memory usage alone does not establish maximum batch capacity.
