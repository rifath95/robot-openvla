# Continue the colored bowl model from 6,000 to 12,000 updates

This is the explicitly requested single-pod alternative to the usual two-stage
sweep delivery: continue training, evaluate, upload and verify on the same pod,
then request a pod stop. Mac connectivity is unnecessary after detached startup.

## Training

Parent: `panda-train-color-bowl-20261015-094416-4344d31f`, checkpoint `006000`.
The complete parent adapter is checked against the 600-demonstration dataset
provenance, copied onto container disk and checked again before loading.
Its optimizer state was not saved: **this continues learned adapter weights with
a fresh optimizer and data stream**, rather than restoring an exact interrupted
optimizer state.

- Same pinned dataset, `Maxalderone/robot-openvla-panda-color-place-bowl` at
  `0bf1faf4036cbfd5e3c7282111f1d9a54b2d020f`.
- All 600 demonstrations / 43,344 action examples used for training.
- 6,000 additional optimizer updates, batch 8, accumulation 1, learning rate
  0.0001, LoRA rank 32, dropout 0; unquantized, no image augmentation.
- **Only three new checkpoints: cumulative 8,000, 10,000, 12,000.** The internal
  additional-run save schedule is 2,000, 4,000, 6,000. Metrics, TensorBoard,
  checkpoint folders and saved metadata use cumulative update numbers. The
  trainer performs 6,000 updates, not an additional 12,000.
- No validation split. Existing checkpoints are retained unchanged.
- New adapter checkpoints and reports go to the network volume; dataset, base,
  starting adapter and later sweep recordings go to container disk.
- Three new checkpoints are uploaded to `Maxalderone/robot-openvla-checkpoints`
  and verified at a pinned commit **before** starting the long sweep. Copies stay
  on the network volume after the pod stops.

## Matched sweep: 600 trials

Each of the three checkpoints receives exactly the prior colored bowl sweep's
200 trials, using its frozen coordinates, color rotations, instruction text,
camera, controller and success criteria. The original checkpoints and baseline
are not rerun. Their separate archives provide the comparison.

| Scene | Colors | Layouts per model | Target requests per layout | Trials per model |
|---|---|---:|---:|---:|
| Single seen | red | 20 | 1 | 20 |
| Single unseen | black | 20 | 1 | 20 |
| Two, red present | red, black | 20 | 2 | 40 |
| Two, red absent | black, purple | 20 | 2 | 40 |
| Four, red present | red, green, black, purple | 10 | 4 | 40 |
| Four, red absent | green, black, purple, orange | 10 | 4 | 40 |
| **Total per model** | | | | **200** |

Every request starts from a fresh reset. Greedy decoding; up to 120 actions.
Instruction: `pick up the COLOR cube and place it in the blue bowl.`
Success requires prior lift ≥8 cm, cube settled within 4.5 cm of bowl center
and 4 mm of floor resting height, and hand ≥8 cm away. Wrong placements and
controller failures remain visible; infrastructure failures end the chain.

Training saw red/blue/green/yellow and at most two cubes. Black/purple/orange
are unseen fine-tuning colors; four-cube scenes are also outside training.

Only one inference model is resident at a time. Its base and selected adapter
load from the staged container copies, merge in VRAM, and stay loaded throughout
that checkpoint's 200 trials. Training processes have exited before inference.

## Estimate, storage and deadlines

- Training updates previously took 3 h 23 min; allow about **3½–4 h** for training
  and loading/saves. Setup, 600 sequential trials, packing and verified delivery
  depend on rollout lengths and network speed. **22 h is the conservative whole-job
  planning estimate**, not a guaranteed duration.
- Whole-chain **inner limit: 25 h** (22 h estimate + requested 3 h buffer).
- Independent **outer limit: 30 h** (inner + requested 5 h buffer), including a
  bounded recovery attempt and shutdown. The separate trainer runtime cap is 7 h.
- Use an **A40**, training template, **200 GB container disk** with at least
  **110 GiB free after image startup**.
- Same network volume: at least **5 GiB free**, retaining new checkpoints/reports.
- Attach the recovery global volume at **`/workspace-global`**, with approximately
  **50 GB available** for raw/packed sweep recovery if HF delivery fails.
- Allow roughly **25 GiB additional HF capacity** for sweep archives and the
  three new adapters. Actual sweep size varies. Check available quota before launch,
  especially while the earlier sweep is also uploading.
- Set `NVIDIA_DRIVER_CAPABILITIES=all`; retain the existing RunPod stop secret.
  Have the HF **write** token ready for the hidden prompt. Do not paste it into
  command text or chat. No new Docker build is required.

## Launch, once the pod is ready

Run in the **pod terminal**:

```bash
if [ -d /root/robot-openvla/.git ]; then
  git -C /root/robot-openvla pull --ff-only
else
  git clone https://github.com/rifath95/robot-openvla.git /root/robot-openvla
fi
cd /root/robot-openvla
/opt/openvla/bin/python -u -m cloud.color_bowl_continuation --launch
```

Wait for `Detached colored bowl continuation chain started. Mac may disconnect.`
Startup prints the exact session, training/sweep watchers, persistent directory
and budgets. Source selection, dataset hashes and all three cumulative update
numbers are recorded in provenance. Each stage's duration is written to
`TIMINGS.md` and `stage_timings.jsonl`.

HF destination: `Maxalderone/robot-openvla-simulator-results`, under the printed
`scripted_color_bowl_continue_...` prefix. Use `data_tools.download_sweep_archive`
for aggregate progress/resume, followed by `data_tools.restore_sweep_archive` for
the usual report and replay layout. Final training provenance and timings are
included in the verified sweep archive.

After verified delivery the supervisor requests pod stop. Failure or timeout also
requests stop, after a bounded recovery attempt when recordings exist. Confirm
**Stopped** in RunPod; an API/network failure can prevent automatic stopping.

Local verification covers learned adapter reload/trainability on a tiny CPU
model, cumulative metrics/checkpoint numbering, exact 600-trial inventory,
portable pack/restore, stage failure ordering and existing supervisor paths.
A complete continuation on the real 7B GPU model is first exercised on the pod.
