# Phase 4 bowl-placement dataset expanded to 50

Expanded October 14, 2026 after the first ten were reviewed. The original ten
recordings remain in the same sequentially numbered dataset. All forty added
predeclared scenes succeeded and passed independent deterministic action replay;
no failed positions were substituted. Collection and checks run locally, without
cloud compute or changes to the separately running lift training.

| Split | Trajectories | Examples / transitions |
|---|---:|---:|
| Training | 34 | 2,446 |
| Validation | 8 | 596 |
| Test | 8 | 586 |
| Total | 50 | 3,628 |

Instruction: **`pick up the red cube and place it in the blue bowl.`**

The initial downward arm pose, open gripper, fixed cube orientation, blue bowl,
bowl camera and updated pose-holding controller are unchanged. Cube and bowl
positions vary within the same reachable workspace, separated by 18–32 cm.
The expert approaches, descends, grasps, lifts, transfers, releases above the
bowl rim and retreats. There are no idle startup or settling labels.

Every recording and replay verified prior lift, final bowl containment, settled
cube velocity and separation from the empty hand. The [audit](assets/place_bowl_50_audit.json)
contains per-episode placement and replay errors. All 100 direct Replay and
Action Replay links are present in `datasets/panda_place_bowl/WATCH_TRAJECTORIES.md`,
ordered as training, validation, test. Keep the local replay launcher running.

## Upload-ready bundle and proposed pilot

The complete portable bundle is `datasets/panda_place_bowl/training_place_bowl_v1`.
It contains all 50 episodes, embedded observations/actions/instructions, fixed
splits, training-only action statistics, verification records and checksums.
All source images, instructions and actions matched the export, and official
processor checks passed on all three splits (37 input tokens, eight supervised
tokens). The bundle is about 452 MB. The separate lift datasets and the old
bowl datasets remain intact.

Upload to a new private HF repository from the local terminal:

```bash
caffeinate -i .venv-training/bin/hf upload-large-folder \
  Maxalderone/robot-openvla-panda-place-bowl \
  datasets/panda_place_bowl/training_place_bowl_v1 \
  --repo-type dataset --private --num-workers 4
```

No upload or training has been launched. The downloader accepts
`--bundle training_place_bowl_v1 --revision COMPLETED_UPLOAD_COMMIT`.
A proposed fresh-pretrained A40 pilot is prepared in
`training/place_bowl_50_500.json`: 500 updates, batch 8, accumulation 1,
learning rate 0.0001, LoRA rank 32, checkpoints and full validation at
0, 50, 100, 300 and 500. The inner limit is 90 minutes and outer 150 minutes.
Select checkpoints with actual bowl-placement trials alongside validation loss.

## Container staging for parallel training

The bowl profile enables `stage_inputs_on_container`. Download and verify the
dataset on the network volume first. The detached trainer then copies the base
model and complete dataset to `/root/panda-training-cache/SESSION/training/`,
verifies copied bytes and dataset checksums, and reads those copies for model
loading and examples. The dataset basename is preserved for profile matching.
The initial copy still reads the shared volume; subsequent input reads are local.

Checkpoints, plots, metrics, TensorBoard events, logs and `input_staging.json`
remain under `/workspace/panda-training/sessions/SESSION/runs/training/`.
Original model and dataset paths and any dataset HF source record are recorded
in staging provenance. Staged checkpoints retain the original base-model path
for later continuation, despite the temporary loading path.

Staging runs inside the supervised child, covered by the outer wall limit and
automatic stop on failure. A disconnected Mac does not cancel copying. The inner
trainer budget retains its existing start point; staging consumes outer time.
Container space must cover all input bytes plus 2 GiB reserve; 60 GB is sufficient
for this base model and dataset. Existing network-volume free-space checks remain.

This setting is enabled for the next bowl run only. The running lift job is not
modified or restarted. Other profiles retain their current input locations until
the setting is explicitly enabled. No Docker dependency changes are required.
