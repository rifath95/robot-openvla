# Phase 3: prepare, fine-tune, and evaluate OpenVLA

The local preparation milestone packages all 50 expert trajectories into a
portable TFDS/RLDS dataset at `datasets/panda_pick_place/training_v1`. Local
preparation has passed, and the first 20-update cloud training test has
completed. Generated datasets remain outside Git; upload the bundle
separately from pushing code.

| Split | Trajectories | Training examples |
|---|---:|---:|
| Training | 36 | 2,595 |
| Validation | 7 | 507 |
| Test | 7 | 507 |

Each example uses the RGB image immediately before an accepted action, the
instruction, and the seven values actually commanded. Episode boundaries are
retained. The export contains images directly in TFRecords, not Mac file paths;
simulation replay files and physics diagnostics stay in the original recordings.
Each episode also stores its final after-image as an RLDS terminal observation,
with no outgoing action. Those 50 terminal records are excluded from the 3,609
training examples, so the final demonstrated action is preserved.
This follows the [RLDS terminal-observation convention](https://github.com/google-research/rlds#dataset-format).

## What training code we use

The official [OpenVLA fine-tuning script](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/vla-scripts/finetune.py)
is preserved under `training/upstream`, together with its MIT license. Its
revision and SHA256 are in `training/upstream/PIN.json`. The full package is
fetched at that revision when building the training image.

`training/integrate_upstream.py` checks the original script's hash and generates
the integration. It keeps the official model, processor, action tokenizer, batch
transform, LoRA attachment (`all-linear`), loss, optimizer and backpropagation.
The project changes are:

- Read our explicitly split RLDS dataset through `training/panda_rlds.py` instead
  of registering it as an Open-X mixture. Training repeats/shuffles encoded images;
  validation is finite and uses the same training normalization. Test data never
  enters the training or validation loops.
- Use eager attention, avoiding a flash-attention compilation for this initial test.
- Log locally and run periodic validation with gradients disabled.
- Save adapters during training; merge a full model once afterward.
- Count completed optimizer updates correctly and stop at the requested count.
- Check a runtime guard between updates. Model loading happens before that guard
  starts; validation/checkpointing can extend it. It never terminates a rented pod.

Normalization uses training-only 1st/99th percentiles for the six motion values,
clipped to [-1,1]. The absolute gripper command stays [0,1]. Roll/pitch ranges are
small but nonzero and are preserved; genuinely degenerate dimensions normalize
to zero and unnormalize to their fixed value. Adapter checkpoints carry these
statistics under `panda_grasp_v1`; do not use Bridge statistics for this model.

## Local preparation and checks

Already prepared bundles do not need rebuilding. To create a new bundle after
collecting a new experiment, use a new output directory:

```bash
python3.11 -m venv .venv-training
.venv-training/bin/python -m pip install -r requirements-training.txt
.venv-training/bin/python prepare_training_dataset.py \
  --manifest datasets/panda_pick_place/panda_expansion_20261010_104720/combined_manifest.json \
  --output-dir datasets/panda_pick_place/training_v1
```

Validate all source images/labels and real processor batches without loading model weights:

```bash
.venv-training/bin/python check_training_inputs.py \
  --processor .cache-openvla/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f
.venv-training/bin/python -m unittest test_training_preparation -v
```

The processor path is the existing local model snapshot. On another machine,
point it at an exported OpenVLA model directory containing the tokenizer and
processor files. The seven-billion-parameter weights are not loaded by this check.
Results are in the bundle's `verification.json` and `input_check.json`. SHA256SUMS
lets the upload command check that the cloud copy matches.

## Build the training image before renting the GPU

Docker is not installed on this Mac. A separate GitHub Actions workflow is
provided; the existing inference image and workflow remain usable.

1. Commit and push the source changes. The dataset is ignored by Git intentionally.
2. Open the GitHub repository's **Actions** tab.
3. Select **Build OpenVLA training image**, then **Run workflow**.
4. Wait for its build and dependency checks to pass.
5. In GitHub Packages, make the new `robot-openvla-training` package public,
   as with the inference package, or supply registry authentication in RunPod.
6. Create a RunPod template using
   `ghcr.io/rifath95/robot-openvla-training:<GITHUB_COMMIT_SHA>`.

Get the exact SHA/tag from the completed build or package page. Do not include
`docker pull` in the template's container image field. Configure TCP port **22**,
60 GB container disk, and the existing `PUBLIC_KEY` startup variable containing
your Mac's public key. Attach the network volume containing `/workspace/openvla-7b`
at `/workspace`; the template's legacy volume-disk size can remain zero.

The Linux image build and first CUDA smoke test have passed. Batch size 1 with
accumulation 4 reached a peak PyTorch allocation of approximately 18.01 GiB on
the tested pod. Other hardware or settings still require their own measurement.

## Upload once after the pod is ready

Run in the **local Mac workspace terminal**:

```bash
.venv/bin/python upload_training_dataset.py \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519'
```

Replace the quoted SSH command with **SSH over exposed TCP** from the pod's
**Connect** page. The helper selects this Mac's actual
`id_ed25519_maxalderone` private key, as our inference launcher does. It refuses
an unmounted `/workspace` or an existing destination, and verifies checksums
after copying. The dataset destination is `/workspace/panda-training/training_v1`.
For future pods attached to the same network volume, skip the upload if that
verified bundle is already present.

## Start the short training test

Connect to the pod using the same direct TCP SSH command, with the real private
key filename. In the resulting **cloud terminal**, run:

```bash
git clone https://github.com/rifath95/robot-openvla.git /root/robot-openvla
cd /root/robot-openvla
/opt/openvla/bin/python -m pip check
/opt/openvla/bin/python train_panda.py \
  --dataset /workspace/panda-training/training_v1 \
  --model /workspace/openvla-7b
```

If the checkout already exists, replace the clone with
`git -C /root/robot-openvla pull --ff-only`. The Docker image already contains
training dependencies and the pinned upstream source; do not install them again.
Model weights must already have a complete export at `/workspace/openvla-7b`.

`training/smoke.json` specifies **20 optimizer updates**, microbatch 1,
accumulation 4 (effective batch 4), LoRA rank 32, learning rate 0.0001, and no
image augmentation or quantization. This is a setup test, not the final training
schedule. Every 10 updates it evaluates a fixed **64-example validation subset**
and saves an adapter. The metrics explicitly record whether validation covered
the complete split. For a longer experiment, use a separate reviewed config and
set `validation_max_batches` to **0** to evaluate all 507 validation examples.

## Next run: 1,000 updates with full validation

The first smoke test passed. Its review and the evidence limits are in
[training/SMOKE_REVIEW.md](training/SMOKE_REVIEW.md). Commit and push the new
configuration before the next run. **The existing training Docker image can be
reused; no rebuild or dataset upload is needed.** Attach
`robot-openvla-network-volume-A40` at `/workspace`, select the training template,
and use the new pod's direct TCP SSH command.

In the cloud terminal, clone the latest repository as above (or pull if it is
already present), then run:

```bash
cd /root/robot-openvla
/opt/openvla/bin/python train_panda.py \
  --dataset /workspace/panda-training/training_v1 \
  --model /workspace/openvla-7b \
  --config training/pick_place_1000.json
```

This starts a **fresh LoRA adapter from the original pretrained model** in a new
run directory; it does not continue or overwrite the smoke checkpoint. Settings
are microbatch 1, accumulation 4, rank 32, learning rate 0.0001, and a 512-example
encoded-image shuffle buffer. It evaluates **all 507 validation examples** and
saves adapters at updates 250, 500, 750 and 1,000. Approximately 4,000 examples
are sampled over the run (about 1.54 training-dataset passes).

Allow roughly **35–45 minutes after setup**, based on the smoke test, with
additional uncertainty for model loading and storage/checkpoint speed. The
one-hour training guard remains a process limit, not a pod billing limit.
Choose a checkpoint based on full validation performance before testing robot
behavior. Leave the test split out of checkpoint selection.

For training while the Mac sleeps, follow [TRAINING_OVERNIGHT.md](TRAINING_OVERNIGHT.md)
instead of running the command directly in SSH. It starts a tmux supervisor,
stores logs/results on the network volume, and requests an automatic pod stop
after training or failure. A management API secret is required. This route puts
training results inside `panda-training/sessions/SESSION_NAME/runs/training/`;
use the checkpoint path from that session for later adapter evaluation.

The previous smoke metrics remain on the network volume. On the next pod, you
can download the small result files into a **Mac terminal** without copying its
adapter weights:

```bash
mkdir -p outputs/phase3_smoke_results
scp -i ~/.ssh/id_ed25519_maxalderone -P POD_PORT \
  root@POD_IP:/workspace/panda-training/runs/lora_20261010_130927_064468/{loss.png,metrics.jsonl,model_info.json,experiment.json} \
  outputs/phase3_smoke_results/
```

Substitute the new pod's IP and port. These files let us inspect the exact earlier
metrics and parameter counts; only its plot and final console metrics have been
reviewed so far.

## See plots and save results

The launcher prints the new run directory under
`/workspace/panda-training/runs/lora_<timestamp>`. It contains:

- `metrics.jsonl`: training loss, action-token accuracy, normalized action L1
  error, validation loss/count, elapsed time, allocated and peak GPU memory.
- `loss.png`: training/validation curves, refreshed at each validation and finish.
- `experiment.json`: source revision, validation settings and runtime guard.
- `model_info.json`: GPU type/VRAM, exact total and trainable parameter counts.
- `tensorboard/`: scalar events for live plots.
- `checkpoint_000010/` and `checkpoint_000020/`: adapters, processor, normalization
  statistics and training configuration.

View these files using the remote VS Code window, or copy them with SCP. To
download a plot into the local workspace after substituting the actual printed
run-folder name, pod IP and port:

```bash
scp -i ~/.ssh/id_ed25519_maxalderone -P POD_PORT \
  root@POD_IP:/workspace/panda-training/runs/RUN_FOLDER/loss.png \
  outputs/phase3_loss.png
```

The training console also prints each metric update. Lower teacher-forced loss
does not establish pick-and-place success; a later live simulator evaluation
will measure grasp, lift, placement, IK rejections and success rate.

For live browser plots, in a second cloud SSH terminal run:

```bash
/opt/openvla/bin/tensorboard --logdir /workspace/panda-training/runs/RUN_FOLDER/tensorboard --host 127.0.0.1 --port 6006
```

In a separate **Mac** terminal, substitute the current pod IP and port:

```bash
ssh -F /dev/null -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519_maxalderone \
  -p POD_PORT -N -L 6006:127.0.0.1:6006 root@POD_IP
```

Keep the tunnel open and visit `http://127.0.0.1:6006` in your Mac browser.

## Merge and live evaluation after training

The saved 20-update smoke adapter can be evaluated before longer training.
Launch the **training template**, attach `robot-openvla-network-volume-A40` at
`/workspace`, and use its current SSH over exposed TCP command. Push the latest
launcher changes first; no Docker rebuild or dataset upload is needed.

In a **local Mac terminal** in this repository:

```bash
.venv/bin/python cloud_session.py \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519' \
  --adapter /workspace/panda-training/runs/lora_20261010_130927_064468/checkpoint_000020
```

Replace `POD_IP` and `POD_PORT` using RunPod → your running pod → Connect →
SSH over exposed TCP. The launcher chooses this Mac's existing private key.
It clones/updates cloud code, merges the saved adapter with the base model onto
container disk, loads the merged model, creates the tunnel, and opens the paused
100-step simulator. Space starts/pauses/resumes; closing the viewer ends the
trial. Enter repeats a trial with the model still loaded. Ctrl+C ends the session;
stop or terminate the pod separately in RunPod.

No training updates happen during this evaluation. The simulator accepts the
trained `panda_grasp_v1` action statistics and retains movement limits and IK
rejection handling. Outputs include images, actions, timings, replay, and the
adapter checkpoint path in startup metadata. A 20-update model may still fail to
pick or place; this is a qualitative behavior check, not evidence of convergence
or a held-out task benchmark.

Each launcher invocation makes a separate merged copy under
`/root/panda-evaluation-*/model`; repeated trials within one session reuse it.
Merging and loading take additional time. The base model and adapter remain
unchanged on the network volume.

For manual merging, the helper creates a full model, attaching
`panda_grasp_v1` statistics. Its output is roughly another
15 GB; use container disk or a volume with sufficient free space, rather than
assuming a 30 GB volume holding the base model can also hold a merged copy.

```bash
/opt/openvla/bin/python -m training.merge_adapter \
  --base-model /workspace/openvla-7b \
  --adapter /workspace/panda-training/runs/RUN_FOLDER/checkpoint_000020 \
  --output /root/panda-finetuned
```

Then the existing server can load it with `--model-dir /root/panda-finetuned`
and `--unnorm-key panda_grasp_v1`. Persistent checkpoint adapters let us recreate
the merged model after terminating a pod. The launcher supports live exploratory
evaluation; controlled held-out task evaluation is still pending.

## Completion criteria

Phase 3 is complete after reproducible training, saved plots/checkpoints, and
live model-driven evaluation on held-out pick-and-place scenarios, compared with
the pretrained baseline. Record actual performance, including limited success;
perfect performance is not required to complete and assess this experiment.
Stop or terminate the rented pod in RunPod after testing. Exiting training does
not stop GPU billing; external storage persists and has its own charge.
