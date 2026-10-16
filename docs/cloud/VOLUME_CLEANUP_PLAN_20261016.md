# Network-volume cleanup proposal — October 16, 2026

This is a plan, not an execution receipt. No deletion, upload or service restart
has been executed for this cleanup. The user wants commands to run themselves
after discussing the targets. Do not use the old `archive_training_checkpoints`
launcher on the running sweep pod: it backs up more than this selected set and
auto-stops the pod.

## Keep on the volume

- `/workspace/openvla-7b`: base model, about 15 GiB.
- `sessions/panda-train-color-bowl-20261016-084536-c17080be`: uninterrupted
  checkpoints 6,000/8,000/10,000/12,000, about 6.1 GiB, including final full
  training state. Required by the active sweep and the new playground default.
- All training reports, configuration, provenance and HF verification receipts.
- Active sweep reports, container recordings/staging and global recovery mount.

The playground's local default now uses the uninterrupted profile. Original
colored-bowl checkpoints are no longer required by new playground launches.
An existing old playground host is not automatically replaced or stopped.

## Archive selected earlier models before removing volume weights

| Task / training session | HF checkpoint(s) to preserve | Evidence |
|---|---|---|
| Lift, 300 demonstrations: `panda-train-20261014-095425-1136` | 1,000 | 16/20 successful lifts, best observed in its matched sweep |
| Red/blue touch: `panda-train-20261013-074359-896` | 100 | 15/20 correct-only contacts, best observed in its sweep |
| Four-color touch: `panda-train-20261013-110217-1084` | 300 and 1,000 | 300 best on 4–6-cube evaluation; 1,000 best on two-cube evaluation |

HF inventory at revision `9d3230d730351b858e56d5a4b72bc57ae18f1328` has
none of these selected weights at their `sessions/.../runs/training/` paths
in `Maxalderone/robot-openvla-checkpoints`. The only other visible model repo
is the separately archived single-cube-touch checkpoint 300. Do not treat the
missing selected backups as already verified. Preserve adapters, processors,
action statistics, training metadata/provenance and relevant reports, then
download/check the uploaded files and retain receipts before deleting sources.
After verified archival, remove all volume checkpoint directories for these
three older training sessions, leaving their reports. Keeping both four-color
candidates costs about 1 GiB on HF and preserves the two useful evaluation cases.

## Remove other checkpoint weights, preserving reports

- Original colored-bowl run `panda-train-color-bowl-20261015-094416-4344d31f`:
  about 2.3 GiB, all five checkpoints already HF-backed. Its checkpoint zero
  remains available from pinned HF for reproducing training initialization.
- Red-only bowl run `panda-train-chain-20261014-174530-4acde087`: about 2.3 GiB,
  verified checkpoint backup already exists.
- Fresh-optimizer continuation `panda-train-color-bowl-cont-20261015-151409-140b3657`:
  about 1.4 GiB, verified checkpoint backup already exists.
- 50-demonstration lift pilot `panda-train-20261014-064135-2604`: about 2.3 GiB,
  superseded by the archived best 300-demonstration lift checkpoint.
- 50-demonstration bowl pilot `panda-train-20261014-104416-1195`: about 2.3 GiB,
  superseded by later bowl training.
- Remaining single-cube-touch checkpoint folder in
  `panda-train-20261013-034928-1786`: separately archived checkpoint 300;
  remaining volume folder is small. Preserve reports.

Red/blue and four-color touching also used 50 demonstrations; the selected
best models above are the task-history exceptions to archive on HF, rather
than keeping their checkpoints on the network volume.

## Remove old data copies and completed sweep transfer material

- `training_four_color_touch_v1`, `training_lift_v1`, `training_lift_v2_300`,
  `training_place_bowl_v1`: about 2.3 GiB combined; uploaded datasets already exist
  on HF. Keep adjacent source/provenance JSON files.
- `scripted-evaluations/scripted_color_lift_models_20261014_144240_51b35ee5`:
  about 1.8 GiB, downloaded/restored and reviewed on the Mac.
- `reports/hf-archives/scripted_color_lift_models_20261014_144240_51b35ee5/packed-parts`:
  about 1.7 GiB of duplicate transfer packages. Retain the parent receipt/logs.

All paths above are relative to `/workspace/panda-training` unless absolute.
After cleanup, the large retained items should be the base plus current
uninterrupted checkpoints (about 21 GiB combined), with reports and small
metadata in addition. Directory-size figures are approximate and include some
reports that will remain. No active sweep input or output is a deletion target.
