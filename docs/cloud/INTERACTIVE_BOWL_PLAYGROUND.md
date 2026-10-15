# Interactive bowl checkpoint testing

Use a separate A40 pod with the training image and the same network volume.
60 GB container disk is sufficient when at least 22 GiB is actually free. There
is no dataset download or global-volume requirement for manual testing.
The running automated sweep uses its own pod and checkout.

## First fetch the training reports

From the Mac repository folder, substitute the new pod's SSH command:

```bash
caffeinate -i .venv/bin/python -m cloud.interactive_bowl_session \
  --ssh 'ssh root@HOST -p PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --reports-only
```

This finds the exact 600-demonstration colored bowl training session from its
dataset provenance, downloads reports to `outputs/training/phase3_training/`,
and leaves the pod running. It does not stage weights or load a GPU model.
Use `--training-session panda-train-color-bowl-...` if multiple matching runs exist.

## Then open the simulator controls

```bash
caffeinate -i .venv/bin/python -m cloud.interactive_bowl_session \
  --ssh 'ssh root@HOST -p PORT -i ~/.ssh/id_ed25519_maxalderone'
```

An ordinary local browser window shows the live MuJoCo camera and clickable
controls together. MuJoCo and physics run on the Mac; prediction runs on the GPU.
It does not require a separate native MuJoCo viewer window. The terminal also
prints the localhost URL so the page can be reopened.

- Checkpoint buttons: **0, 1,000, 2,000, 4,000, 6,000**. Checkpoint zero is the
  freshly initialized adapter baseline, with this dataset's action normalization.
- Choose **1, 2, 3 or 4 cubes** and distinct colors. Red/blue/green/yellow were
  trained; black/purple/orange/white/pink/brown/cyan were not.
- Choose the target color; the instruction box initially fills the corresponding
  bowl-placement instruction. Edit it freely: the exact text is sent to OpenVLA.
  Outcome scoring follows the target selector, even if the text asks otherwise.
- **Apply / reset same positions** applies scene/color/target selections, restoring
  the same seed's initial arm, cubes and bowl. **New positions** chooses a new seed.
  Count/color selections must be applied before running the new scene.
- **Play** runs at most 120 predictions from the current pose, stopping on placement,
  tracking failure, three rejected actions or your stop request.
- **Act once** predicts and executes one action, retaining the resulting pose.
- **Stop** cancels the trial, discards late predictions, and holds the current pose.
  Play/Act can continue; Reset starts over. Changing checkpoint resets the same
  layout so the next model starts from its initial pose.
- Optional sampling uses temperature 1 and top-p 1; greedy decoding is the default.

Instruction and controls are HTML outside the unmodified 640×480 model image.
Only the existing MuJoCo camera observation is sent to the prediction endpoint.
The current grasp-point controller, IK retries, 1 cm / .05 radian bounds and 200
physics steps per action are reused. Simulation stays frozen during inference.

## Model staging and lifetime

On first startup, the base and all five checkpoints are copied and checked on
**container disk**, under `/root/panda-interactive-bowl`. GPU loads read these
copies. Adapters merge into the base in VRAM using the existing inference backend;
no merged model is written to the network volume. Switching checkpoint first
releases the old model and loads the new one. All trials with the same checkpoint
reuse its loaded model. Weight staging is reused when reopening local controls
on the same container. Stop/redeployment can erase container disk.

Closing the browser requests a local trial stop. Ctrl+C ends local controls and
the tunnel; both leave the cloud model resident. **This manual workflow does not
auto-stop the pod. Stop it in RunPod when finished.** A model server has a 12-hour
readiness runtime; selecting the checkpoint again reloads an expired server.
Keep the Mac awake and connected while actively controlling it.

Cloud host log: `/root/panda-interactive-bowl/host.log`.
Per-checkpoint logs: `/root/panda-interactive-bowl/checkpoint_NNNNNN.log`.
Local recordings: `outputs/simulator_runs/interactive/bowl_DATE_TIME/trial_.../`.
Each contains the typed instruction, checkpoint provenance, before/after images,
actions, physics states, motion frames and a `RUN_REPORT.md` with direct Replay
and Action Replay buttons through the shared local replay launcher.

Local checks cover all four scene counts rendering and initializing, real
controller execution, instruction preservation, cancelled-response rejection,
same-model reuse and checkpoint identity checks. A real GPU switch and the full
Mac-to-cloud path still need their first check on the new pod.
