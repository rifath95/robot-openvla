# Detached training with automatic pod stop

Use this for the expanded-dataset **5,000-update A40 run**. Training runs on the pod inside `tmux`, so
closing VS Code or sleeping the Mac does not disconnect the training process.
The wrapper requests **stop**, not termination, after success, failure, or its
**five-hour outer wall limit**. This is a dedicated training pod: stopping it ends all
processes on that pod.

## 1. Prepare before launching

Push the latest code to GitHub. Reuse the existing **training Docker template**;
no image rebuild or dataset re-upload is needed. Attach
`robot-openvla-network-volume-A40` at `/workspace` in CA-MTL-1.

In RunPod **Credentials → API Keys**, create a key named `panda-training-stop`
with permission to read your pod and stop it (a read-only key will not work).
If scoped permissions are offered, allow the pod GET and stop POST operations.
This is a RunPod management API key, not an SSH key or S3 key.

Store the key through **RunPod → Secrets**, for example as `panda-training-stop`.
In the training template's environment variables, add:

| Key | Value |
| --- | --- |
| `RUNPOD_API_KEY` | Use the key/secret picker to select that secret |

Keep `PUBLIC_KEY` as already configured. `RUNPOD_POD_ID` is provided by RunPod;
do not paste a previous pod's ID into the template. Do not put the management key
in GitHub, this document, terminal commands, or chat. Add the secret before
launching; editing a running pod can reset its container disk.

Official API: [RunPod stop endpoint](https://docs.runpod.io/pods/manage-pods#stop-a-pod).

## 2. Connect from the Mac

Get the new endpoint from **Pod → Connect → SSH over exposed TCP**. In a local
Mac terminal:

```bash
ssh -F /dev/null -o IdentitiesOnly=yes \
  -i ~/.ssh/id_ed25519_maxalderone -p POD_PORT root@POD_IP
```

Replace `POD_IP` and `POD_PORT` with the current pod's values. The next commands
run in the **cloud terminal**, where the prompt starts with `root@...`.

## 3. Update code and start the detached job

For a fresh pod:

```bash
git clone https://github.com/rifath95/robot-openvla.git /root/robot-openvla
cd /root/robot-openvla
```

If that directory already exists, use instead:

```bash
cd /root/robot-openvla
git pull --ff-only
```

Install the small terminal utility only if needed; all training dependencies are
already in the image:

```bash
command -v tmux || (apt-get update && apt-get install -y tmux)
bash scripts/train_detached.sh training/pick_place_v2_5000.json /workspace/panda-training/training_v2
```

For the first **20-update automatic-stop test**, run this instead of the last
line above:

```bash
bash scripts/train_detached.sh training/smoke.json
```

The smoke test evaluates a 64-example validation sample; completing that sample
is accepted for this test. The 5,000-update configuration requires full final validation. Each launch creates a new session directory.

The script verifies API read access and the attached network volume before
launching. It prints the exact session name, attach command, log path, and status
path. No Python training parameters need to be pasted: it uses
`training/pick_place_v2_5000.json` with the expanded dataset and original base model.
This profile requires the A40 and a compatible saved benchmark; it rejects a
missing recommendation instead of silently using a fallback. The reviewed A40
result selects microbatch 8 and accumulation 1. Allow at least 3 GiB free on the
network volume for five adapter checkpoints and logs; the launcher checks this.
The no-argument command still selects the historical v1/1,000-update profile.

| Setting | This run |
| --- | --- |
| Dataset | 250 trajectories: 176 train / 37 validation / 37 test |
| Optimizer updates | 5,000 |
| Effective batch | 8 training steps per update |
| Training examples presented | 40,000, about 2.9 passes over 13,812 train examples |
| Full validation | Before training (update 0), then every 1,000 updates |
| Checkpoints | Every 1,000 updates |
| Inner training limit | 14,400 seconds (4 hours) |
| Outer supervisor limit | 18,000 seconds (5 hours) |

Repeated/shuffled sampling does not guarantee each example appears equally often.
Validation and test trajectories are excluded from gradient updates.
The update-0 evaluation writes the initial validation loss to `metrics.jsonl`,
TensorBoard, and `loss.png` before the first training update. It uses the same
validation split as later evaluations, with no gradients or weight updates.
For adapter continuation, this baseline evaluates the loaded adapter. Initial
validation counts toward both runtime limits; no update-0 checkpoint is saved.

Use the printed `tail -F .../console.log` command to see initial progress. Ctrl+C
stops **tail**, not the training job. Wait until actual training updates appear
and check `status.json` before leaving it unattended. The launcher message alone
is not proof that the worker started successfully.

## 4. Disconnect or reconnect

You may close the SSH terminal or sleep the Mac after starting the job.
To view the cloud terminal, use the exact printed command, for example:

```bash
tmux -L SESSION_NAME attach -t train
```

Replace `SESSION_NAME` with the launcher output. To detach, press **Ctrl+B**, then
**D**. Pressing Ctrl+C inside the worker's tmux pane can interrupt the supervisor;
use detach to leave training running. On reconnection to a still-running pod,
list session sockets/directories via:

```bash
ls -td /workspace/panda-training/sessions/panda-train-* | head
```

Stopping the pod ends tmux too. It cannot reconnect to a session on a stopped or
terminated pod. Persistent saved results can be read after attaching the volume
to a running pod again.

## 5. Saved results and stopping behavior

Each job gets a unique directory:

```text
/workspace/panda-training/sessions/panda-train-TIMESTAMP-PID/
  console.log
  status.json
  runs/training/
    metrics.jsonl
    loss.png
    experiment.json
    model_info.json
    tensorboard/
    batch_selection.json
    checkpoint_001000/
    checkpoint_002000/
    checkpoint_003000/
    checkpoint_004000/
    checkpoint_005000/
```

Checkpoints exist only if training reached those updates. Training starts a fresh
adapter and leaves the original base model and smoke checkpoint unchanged.
`status.json` records the exit code, final update, missing artifacts, full-final-
validation status, and stop-request state. A zero process exit alone does not
mean all 5,000 updates completed: the training runtime guard may finish earlier.

The supervisor closes the training log, writes status, flushes filesystem writes,
then calls the RunPod stop API. It tries up to three times on API/network errors.
The management credential is removed from the training subprocess environment
and is never included in commands or output files.

The inner **four-hour training limit** begins after model/data setup and checks
between updates. On expiry, the trainer saves the current completed update and
ends; it does not wait for the next 1,000-update checkpoint. Validation and saving
can extend this inner limit. The outer **five-hour hard limit** starts when the
cloud supervisor starts, not when you deploy the pod. It includes training setup,
validation, and saving;
on expiry the child process group is terminated, then the pod stop is requested.
A forced timeout or training failure may leave only earlier checkpoints or an
incomplete checkpoint. The wrapper records this as a failure, not success.

**Automatic stopping is a request, not a guarantee.** A host crash, killed
supervisor, unavailable API, invalid/insufficient key, or storage failure can
prevent it. API read preflight cannot prove stop permission. Check the RunPod
portal afterward: an accepted stop request is not proof the GPU has stopped.
The wrapper deliberately never deletes your pod or network volume. Persistent
storage charges continue after compute stops.

The detached wrapper and automatic stop were verified in earlier 20-update and
1,000-update cloud runs. The new profile and adapter continuation have local
checks, including a real CPU LoRA save/reload and another training update.
The new 5,000-update run has not been launched yet.

## 6. Continue training from a saved adapter later

The default launch above starts a fresh LoRA adapter from the original base.
To continue learned adapter weights, supply a third argument:

```bash
bash scripts/train_detached.sh training/pick_place_v2_5000.json \
  /workspace/panda-training/training_v2 \
  /workspace/panda-training/sessions/OLD_SESSION/runs/training/checkpoint_005000
```

Replace `OLD_SESSION` with the saved session name. Choose an existing complete
checkpoint; it can also be an earlier checkpoint if training stopped early.
This creates a new output session and trains for **another** configured 5,000
updates (unless a time limit ends it first). Change the configuration to select
a different number of additional updates. The base weights and source checkpoint
are read without overwriting them.

This is **continuation from adapter weights with a fresh optimizer**, not exact
training-state resume: AdamW momentum, shuffled data position, elapsed time, and
update numbering restart. Checkpoints do not contain optimizer state. LoRA
rank/dropout and the original base path must match. The source adapter path and
continuation mode are saved in `experiment.json` for the new run.
