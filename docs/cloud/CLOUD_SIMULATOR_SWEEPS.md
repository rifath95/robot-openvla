# Detached cloud simulator sweeps

The new `cloud.cloud_sweep_session` mode runs MuJoCo, rendering, inference, and trial recording on a dedicated GPU pod. Your Mac launches it and optionally watches/fetches results. Mac sleep, closing the terminal, or losing SSH does not cancel the worker. The cloud supervisor requests stopping after completion, worker failure, or a runtime limit.

The ordinary interactive simulator workflow stays available. This mode runs sequential trials (inference batch one). OpenVLA's upstream generation wrapper explicitly rejects larger batches; batching was deferred rather than replacing its decoder. Physics runs on CPU; headless EGL rendering and model inference use the pod GPU.

## Resume the interrupted color-transfer sweep

Checkpoint 50 and checkpoint 100 each have 60 recorded trials. Those 120 outcomes, including task failures, remain unchanged. The new worker executes only checkpoint 300 and 500, for 120 remaining trials. Recorded results retain their original execution mode; new rows and startup metadata identify cloud execution. Linux/EGL rendering can differ slightly from Mac rendering; this change is recorded and timing measurements should not be pooled as identical Mac-to-cloud measurements.

Use a separate A40 training-template pod, the same network volume, and the existing RunPod stop secret. For EGL, set template environment variable `NVIDIA_DRIVER_CAPABILITIES=all` before deployment so graphics driver libraries are exposed. The launcher prepares a separate pinned MuJoCo environment and missing EGL/GL runtime libraries automatically; it does not change the model Python dependencies. No Docker rebuild is required.

After receiving the current SSH address, run locally:

```bash
.venv/bin/python -m cloud.cloud_sweep_session \
  --ssh 'ssh root@HOST -p PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --resume outputs/scripted_color_transfer_20261013_100305_2a5212fb
```

Wait for **`Detached cloud sweep started`**. Before that message, setup still requires your Mac connection. After it appears, you can disconnect or put your Mac to sleep. Ctrl+C then ends only the local watcher. Add `--launch-only` if you want the launcher to return immediately after starting the supervisor.

If you stay connected, the watcher downloads images, actions, physics states, trial replay launchers, startup metadata, reports, and logs into the same local folder. Old trial images are retained. The supervisor gives an active download watcher up to 30 minutes to finish before requesting stopping; an acknowledgment ends this window immediately. If the Mac is disconnected, there is no required download window: results stay on the volume and the pod is stopped without waiting for a Mac.

## Storage and model switching

- Persistent results: `/workspace/panda-training/scripted-evaluations/<sweep-name>/`.
- Original Mac trials stay on the Mac; only their compact result records are imported. New cloud trial images/actions/states are copied to the volume before recording each trial as finished.
- Base weights and selected checkpoints are copied once to `/root/panda-sweep-cache/<sweep-name>/`. Source weights on the volume are never edited.
- Each checkpoint loads the staged base into GPU memory, loads its adapter, merges in memory, and runs its trials. It writes no full merged model to disk. The server exits and releases its GPU memory before the next checkpoint loads.
- The launcher fetches repository code once and creates a private checkout. It fetches only the pinned Panda assets from MuJoCo Menagerie. No pulls occur between trials or checkpoints. The exact commit is saved in supervisor and execution-segment metadata.

Individual trials write on container disk while running, then publish their completed artifacts to the volume. A hard interruption can lose the unfinished trial's latest temporary files, but already published trial results remain. On restart, unrecorded trials can be retried. A local or cloud record of an unsuccessful task is a measured outcome and is not retried merely because it failed.

## Timeouts

For both cloud and Mac-driven sweeps:

```text
inner limit = estimated whole-process duration + 1 hour
outer limit = inner limit + 1 hour
```

The default remaining-work estimate uses observed sweep wall time per trial, conservatively floored at 60 seconds per trial, plus 10 minutes for preparation and 2 minutes per checkpoint for model switching. For this 120-trial continuation, that is an estimated **2 h 14 min**, inner **3 h 14 min**, outer **4 h 14 min**. This is an unbatched conservative estimate, not a requirement to keep the GPU running that long. Actual cloud trials may be faster because they remove the Mac network round trip. Override only the estimate with `--estimated-seconds`; both one-hour buffers still apply.

The supervisor's elapsed time includes weight staging, the headless render check, model loading/merging, trials, and artifact publication. Initial repository and dependency setup precede the supervisor, is individually bounded, and requires the Mac to remain connected. It does not include pod deployment time. The inner cap stops sweep work; the outer cap bounds the supervisor and any result-download grace. API/network failure can still prevent a stop request from succeeding; check Stopped in RunPod.

## Fetch after disconnection

Resume a pod with the same volume and run:

```bash
.venv/bin/python -m cloud.cloud_sweep_session \
  --ssh 'ssh root@HOST -p PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --resume outputs/scripted_color_transfer_20261013_100305_2a5212fb \
  --fetch-only
```

This fetches and merges results without starting another sweep. If the old cloud supervisor is still active in its download window, acknowledgment lets it stop the pod. If the pod was already stopped and you resumed it solely to fetch, stop that resumed pod manually afterward. To continue an interrupted cloud sweep, fetch its newer records first, then launch again with `--resume`.

Cloud-only simulator timings are excluded from `GPU_INFERENCE_COMPARISON.md`, whose simulator cycles measure the Mac/cloud workflow. Cloud sweep timings remain in trial logs and reports.

## Existing Mac-driven scripted mode

`cloud.scripted_color_transfer_session` and the other scripted launchers still run physics locally. They now arm a detached cloud watchdog with a heartbeat every 15 seconds and a 3-minute disconnect grace. If heartbeats stop, the watchdog requests a pod stop. Their model weights are also staged once and merged in GPU memory. Git is pulled only once at sweep launch, and checkpoint setup no longer pulls again. Interactive `cloud.cloud_session` behavior is unchanged.

The cloud-only mode treats a Mac disconnection as permission to continue. The Mac-driven mode treats lost contact as a reason to stop, because it cannot continue its simulator without the Mac.

## Validation before paid deployment

44 local checks cover timeout calculations, missed heartbeats, worker failure, worker termination, checkpoint/seed/target mapping, preserved measured failures, weight copy integrity, GPU merge call order, result archive validation, old-image preservation, and unchanged interactive paths. A local end-to-end fixture verified that every transmitted image matched the demonstration and that recorded expert actions still touched the correct cube. The first deployment still needs to validate Linux EGL rendering and the GPU in-memory merge; this occurs in the same supervised sweep before its remaining trials, rather than a separate paid job.

## Cross-platform scene precision

The first cloud continuation stopped before model loading because an exact coordinate comparison rejected NumPy 1.26.4 vs Mac NumPy 2.5.3 rounding. The largest measured difference was 1.11e-16 metres. The check now permits up to 1e-12 metres of arithmetic rounding, still rejects changed coordinates, and feeds the saved plan coordinates directly into each resumed simulator trial. Ordinary interactive scene initialization is unchanged. The launcher returns a failure status when the cloud worker fails, after downloading diagnostics and leaving stopping to the supervisor.

## Loopback port allocation

The next deployment passed EGL rendering and loaded/merged the model in 17.63 seconds, then stopped because port 8001 was already occupied. Cloud sweep servers now bind an OS-assigned available loopback port (`--port 0`) before model loading, retain that socket, and atomically publish the selected port after loading. The worker uses that exact port for health checks and predictions. This avoids fixed-port conflicts and probe/rebind races. The ordinary interactive server still defaults to port 8000.
