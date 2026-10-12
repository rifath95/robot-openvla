# GPU inference comparison

One request = one RGB image + instruction → one seven-value action. Inference batch size is 1.

Timings are seconds. The first 5 recorded requests per trial are excluded as warmup. Short trials without remaining samples show missing timings. These are observed simulator runs, not identical-input hardware benchmarks; model/checkpoint, instruction, network, Mac, storage and software differences can affect results. Rejected/held actions remain prediction requests.

## Full cycle and estimated cost for 100 steps

| GPU | Run | Measured/all requests | Full cycle s | Request RTT s | Request p95 s | Model prediction s | USD/hour | 100 cycles s | 100 active cycles USD | 100 cycles + preparation/loading USD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NVIDIA A40 | simulator_trial | 0/5 | — | — | — | — | 0.59 | — | — | — |
| NVIDIA A40 | remote_loop_20261010_060952_21db409c | 95/100 | 1.464 | 0.960 | 1.196 | 0.319 | 0.59 | 146.4 | 0.0240 | 0.0345 |
| NVIDIA A40 | remote_loop_20261010_072519_5186f692 | 95/100 | 1.416 | 0.911 | 0.981 | 0.299 | 0.59 | 141.6 | 0.0232 | 0.0301 |
| NVIDIA A40 | remote_loop_20261010_133609_315d8199 | 82/87 | 1.630 | 1.123 | 1.341 | 0.331 | 0.59 | 163.0 | 0.0267 | 0.0276 |
| NVIDIA A40 | remote_loop_20261011_003638_4fe3675c | 95/100 | 1.561 | 1.053 | 1.323 | 0.321 | 0.59 | 156.1 | 0.0256 | 0.0265 |
| NVIDIA A40 | remote_loop_20261011_021037_83d8c4a3 | 40/45 | 1.450 | 0.944 | 1.034 | 0.308 | 0.59 | 145.0 | 0.0238 | 0.0248 |
| NVIDIA A40 | remote_loop_20261011_023950_e87092ab | 57/62 | 1.865 | 1.357 | 2.508 | 0.346 | 0.59 | 186.5 | 0.0306 | 0.0315 |
| NVIDIA A40 | remote_loop_20261011_130030_90592976 | 95/100 | 1.411 | 0.905 | 1.030 | 0.295 | 0.59 | 141.1 | 0.0231 | 0.0240 |
| NVIDIA A40 | remote_loop_20261011_130522_3943abad | 95/100 | 1.421 | 0.920 | 1.045 | 0.292 | 0.59 | 142.1 | 0.0233 | 0.0243 |
| NVIDIA RTX A5000 | Not measured | 0 | — | — | — | — | 0.27 | — | — | — |
| NVIDIA RTX A6000 | Not measured | 0 | — | — | — | — | 0.59 | — | — | — |

Cost = estimated seconds ÷ 3600 × user-confirmed hourly rate. 100-cycle estimates extrapolate the measured mean, including partial trials; they do not assert task success. Setup-inclusive cost adds recorded model preparation and loading once. It excludes image deployment, adapter merge/export unless covered by recorded preparation, other setup, manual pauses, idle time, shutdown delays and storage charges. Rates are comparison inputs, not historical billing records. Multiple trials can reuse one model loading. Missing measurements stay missing.

## Remote request breakdown — averages in seconds

| Run | Request RTT | Receive body | Decode image | Processor + GPU transfer | Model prediction | Server total | Remaining server overhead | RTT minus server total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simulator_trial | — | — | — | — | — | — | — | — |
| remote_loop_20261010_060952_21db409c | 0.960 | 0.014 | 0.010 | 0.043 | 0.319 | 0.386 | 0.000 | 0.575 |
| remote_loop_20261010_072519_5186f692 | 0.911 | 0.007 | 0.009 | 0.027 | 0.299 | 0.342 | 0.000 | 0.569 |
| remote_loop_20261010_133609_315d8199 | 1.123 | 0.009 | 0.011 | 0.092 | 0.331 | 0.443 | 0.000 | 0.680 |
| remote_loop_20261011_003638_4fe3675c | 1.053 | 0.006 | 0.009 | 0.111 | 0.321 | 0.446 | 0.000 | 0.607 |
| remote_loop_20261011_021037_83d8c4a3 | 0.944 | 0.007 | 0.007 | 0.029 | 0.308 | 0.351 | 0.000 | 0.593 |
| remote_loop_20261011_023950_e87092ab | 1.357 | 0.204 | 0.010 | 0.120 | 0.346 | 0.680 | 0.000 | 0.677 |
| remote_loop_20261011_130030_90592976 | 0.905 | 0.007 | 0.006 | 0.025 | 0.295 | 0.333 | 0.000 | 0.572 |
| remote_loop_20261011_130522_3943abad | 0.920 | 0.006 | 0.008 | 0.030 | 0.292 | 0.336 | 0.000 | 0.584 |

Server total is inside request RTT, not added to it. RTT minus server total is residual network/tunnel/client overhead, not pure network latency. Receive-body time can itself include waiting for uploaded bytes. Processor and GPU transfer are measured together. Model prediction includes vision encoding, autoregressive action-token generation and decoding to seven values.

## Client request stages — averages in seconds

| Run | Encode (outside RTT) | Connect (outside RTT) | Send | Wait for headers | Read response |
| --- | --- | --- | --- | --- | --- |
| simulator_trial | — | — | — | — | — |
| remote_loop_20261010_060952_21db409c | 0.001 | 0.001 | 0.001 | 0.957 | 0.003 |
| remote_loop_20261010_072519_5186f692 | 0.001 | 0.001 | 0.001 | 0.910 | 0.000 |
| remote_loop_20261010_133609_315d8199 | 0.001 | 0.001 | 0.001 | 1.122 | 0.000 |
| remote_loop_20261011_003638_4fe3675c | 0.001 | 0.001 | 0.001 | 1.050 | 0.003 |
| remote_loop_20261011_021037_83d8c4a3 | 0.001 | 0.001 | 0.001 | 0.943 | 0.000 |
| remote_loop_20261011_023950_e87092ab | 0.001 | 0.001 | 0.001 | 1.356 | 0.000 |
| remote_loop_20261011_130030_90592976 | 0.001 | 0.001 | 0.001 | 0.904 | 0.000 |
| remote_loop_20261011_130522_3943abad | 0.001 | 0.001 | 0.001 | 0.919 | 0.000 |

Send, wait for response headers and read response are included in request RTT. Sending may buffer bytes locally; it is not a separately measured network upload duration. Waiting includes server processing and network delays.

## Local simulator breakdown — averages in seconds

| Run | Input capture/save | IK | Movement | Result capture/save | Local subtotal | Remaining cycle overhead |
| --- | --- | --- | --- | --- | --- | --- |
| simulator_trial | — | — | — | — | — | — |
| remote_loop_20261010_060952_21db409c | 0.022 | 0.001 | 0.446 | 0.025 | 0.494 | 0.010 |
| remote_loop_20261010_072519_5186f692 | 0.022 | 0.001 | 0.446 | 0.025 | 0.495 | 0.009 |
| remote_loop_20261010_133609_315d8199 | 0.023 | 0.001 | 0.446 | 0.027 | 0.497 | 0.010 |
| remote_loop_20261011_003638_4fe3675c | 0.022 | 0.001 | 0.446 | 0.029 | 0.498 | 0.010 |
| remote_loop_20261011_021037_83d8c4a3 | 0.023 | 0.001 | 0.445 | 0.028 | 0.497 | 0.010 |
| remote_loop_20261011_023950_e87092ab | 0.022 | 0.001 | 0.446 | 0.028 | 0.498 | 0.010 |
| remote_loop_20261011_130030_90592976 | 0.022 | 0.001 | 0.447 | 0.026 | 0.495 | 0.011 |
| remote_loop_20261011_130522_3943abad | 0.021 | 0.001 | 0.446 | 0.024 | 0.492 | 0.010 |

Full cycle = request RTT + local subtotal + remaining cycle overhead, using records with all required fields. Remaining overhead includes client encoding/connection work and recording outside named stages. Movement may be deliberately paced by the live viewer. Pauses or interruptions within a recorded cycle can affect timing. Each column averages available measurements only; different coverage can prevent exact subtotal agreement.

## Setup and comparison context

| Run | VRAM GiB | ECC | Model / checkpoint | Load s | Preparation s | Storage | PyTorch / Transformers | Precision / attention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simulator_trial | 44.432 | Not recorded | /workspace/openvla-7b | 63.043 | 0.055 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261010_060952_21db409c | 44.432 | Not recorded | /workspace/openvla-7b | 64.261 | 0.049 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261010_072519_5186f692 | 44.432 | Not recorded | /workspace/openvla-7b | 41.930 | 0.048 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261010_133609_315d8199 | 44.424 | Not recorded | /workspace/panda-training/runs/lora_20261010_130927_064468/checkpoint_000020 | 5.265 | 0.049 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261011_003638_4fe3675c | 47.402 | Not recorded | /workspace/panda-training/sessions/panda-train-20261010-233038-932/runs/training/checkpoint_000500 | 5.498 | 0.048 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261011_021037_83d8c4a3 | 44.422 | Not recorded | /workspace/panda-training/sessions/panda-train-20261010-233038-932/runs/training/checkpoint_001000 | 6.027 | 0.050 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261011_023950_e87092ab | 44.424 | Not recorded | /workspace/panda-training/sessions/panda-train-20261010-233038-932/runs/training/checkpoint_001000 | 5.844 | 0.047 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261011_130030_90592976 | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261011-090041-1584/runs/training/checkpoint_003000 | 5.384 | 0.045 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |
| remote_loop_20261011_130522_3943abad | 44.432 | Not recorded | /workspace/panda-training/sessions/panda-train-20261011-090041-1584/runs/training/checkpoint_005000 | 5.866 | 0.038 | nfs4 | 2.2.2+cu121 / 4.40.1 | torch.bfloat16 / eager |

Model loading includes processor setup, checkpoint reading, construction and GPU transfer; it does not isolate storage-to-VRAM time. Historical ECC is not inferred from unrelated training pods.

## Refreshing this report

New local simulator trials refresh this report automatically after saving their run report. No cloud connections or predictions are made by report generation.

```bash
.venv/bin/python -m analysis.summarize_gpu_inference
```

Hourly rates are stored in `inference/gpu_inference_rates.json`. Raw measurements remain in `outputs/<run>/startup_info.json` and `step_*/execution.json`. Each run also keeps `RUN_REPORT.md`.
