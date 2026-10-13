# Compare touch checkpoints while training continues

Training instruction: **touch the red cube.** The launcher uses that exact
instruction by default with `--task touch`.

An inference pod can attach the same network volume as the training pod and
read its completed checkpoints. Use the training template for adapter merging.
The inference pod creates its merged model on its own container disk and does
not write to the training session's checkpoint or report directories. Storage
bandwidth is shared, so loading a model may briefly slow the trainer's reads or
writes. Stopping one pod does not stop the other. No trainer changes are needed.

The launcher checks that all required checkpoint JSON metadata is readable,
including `training_config.json`, which the current trainer saves after the
adapter and processor. It also verifies that the safetensors file has its full
header and expected data length. A missing/partial checkpoint is rejected before
the previous inference server is stopped. Wait for the trainer's
`Saved adapter checkpoint: ...` line before evaluating a new checkpoint.

Use checkpoint **0** as the untrained baseline. Its initialized LoRA adapter
leaves pretrained weights unchanged, while supplying the same touch-dataset
normalization statistics used by later checkpoints. A base-model trial with
Bridge statistics would not be a comparable action-decoding baseline.

## Trial sequence

For each checkpoint (0, 10, 50, 100, and subsequent completed saves):

1. Run greedy decoding on the chosen scene seeds.
2. End the local session with Ctrl+C or q, releasing its SSH tunnel while keeping
   the model loaded.
3. Run the same seeds with `--sample-actions`.
4. Repeat sampled trials if needed to assess variability.

The touch task reproduces the collection's fixed downward-facing closed-gripper
arm start and camera. `--scene-seed` identifies the collector's predeclared slot:
seeds 0, 1, 2 correspond to training demonstrations train_001, train_002, train_003.
Seed 4 is validation_001; seed 5 is test_001. Seeds 50 onward use new reproducible
cube positions not in the current 50-trajectory collection. Known training scenes
are useful for the first learning check; held-out and new positions are separate
generalization checks.

The trial stops on verified positive-force finger–cube contact, preserving the
successful final state and recording `task_result` in its summary and execution
JSON. Otherwise it runs up to the action/time/rejection cap or a user stop. No
expert commands or coordinates are fed into the model; cube coordinates are
used only for scene initialization and measuring the task result.

Example local Mac command, replacing the SSH and session placeholders:

```bash
caffeinate -i .venv/bin/python -m cloud.cloud_session \
  --ssh 'INFERENCE_POD_SSH_COMMAND' \
  --adapter /workspace/panda-training/sessions/TRAINING_SESSION/runs/training/checkpoint_000000 \
  --task touch --instruction 'touch the red cube.' \
  --scene-seeds 0 1 2 --steps 50 --server-runtime-seconds 21600
```

For the sampling trial, use the same command with:

```bash
--sample-actions --temperature 1.0 --top-p 1.0
```

Greedy/sampled trials of the same checkpoint share one updated resident model;
changing checkpoints reloads the model. Each trial saves its own images, raw and
executed actions, motion frames, scene seed, checkpoint provenance, sampling
settings, timings, report and replay under `outputs/remote_loop_*`. The Mac must
remain connected and awake for live simulator trials; the detached training job
continues independently if the Mac disconnects.

## Local verification

The touch simulator was exercised with the recorded expert actions as mock
predictions (no cloud model). It reached actual finger contact and stopped at
step 19, matching train_001's demonstration length. Every initial/input image
matched the corresponding demonstration PNG exactly (19/19), and every saved
post-action qpos matched exactly (maximum difference 0). This checks scene,
controller, image capture and contact termination; it does not measure a model's
learned performance.

Tests cover incomplete checkpoint rejection, readiness before server replacement,
touch-task initialization and contact termination, sampling transmission and
recording, existing cube/bowl simulator behavior, and resident-server reuse.
