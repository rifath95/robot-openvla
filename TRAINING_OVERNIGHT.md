# Detached training with automatic pod stop

Use this for the 1,000-update run. Training runs on the pod inside `tmux`, so
closing VS Code or sleeping the Mac does not disconnect the training process.
The wrapper requests **stop**, not termination, after success, failure, or its
90-minute wall limit. This is a dedicated training pod: stopping it ends all
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
bash scripts/train_detached.sh
```

The script verifies API read access and the attached network volume before
launching. It prints the exact session name, attach command, log path, and status
path. No Python training parameters need to be pasted: it uses
`training/pick_place_1000.json` with the existing dataset/base model.

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
    checkpoint_000250/
    checkpoint_000500/
    checkpoint_000750/
    checkpoint_001000/
```

Checkpoints exist only if training reached those updates. Training starts a fresh
adapter and leaves the original base model and smoke checkpoint unchanged.
`status.json` records the exit code, final update, missing artifacts, full-final-
validation status, and stop-request state. A zero process exit alone does not
mean all 1,000 updates completed: the training runtime guard may finish earlier.

The supervisor closes the training log, writes status, flushes filesystem writes,
then calls the RunPod stop API. It tries up to three times on API/network errors.
The management credential is removed from the training subprocess environment
and is never included in commands or output files.

The inner one-hour training limit begins after model/data setup and checks between
updates. The outer **90-minute hard limit** includes setup, validation, and saving;
on expiry the child process group is terminated, then the pod stop is requested.
A forced timeout or training failure may leave only earlier checkpoints or an
incomplete checkpoint. The wrapper records this as a failure, not success.

**Automatic stopping is a request, not a guarantee.** A host crash, killed
supervisor, unavailable API, invalid/insufficient key, or storage failure can
prevent it. API read preflight cannot prove stop permission. Check the RunPod
portal afterward: an accepted stop request is not proof the GPU has stopped.
The wrapper deliberately never deletes your pod or network volume. Persistent
storage charges continue after compute stops.

This mechanism has local mocked tests; a live stop request must be checked on
the first cloud run. No real pod was stopped during those tests.
