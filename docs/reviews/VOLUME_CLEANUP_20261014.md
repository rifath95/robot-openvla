# Network-volume cleanup — October 14, 2026

User authorized cleanup and chose to finish the existing Mac download of the latest 1,080-trial sweep, skipping HF upload for this sweep.

Volume usage changed from **29 GiB used / 8.9 GiB free** to **24 GiB used / 14 GiB free**, on a 38 GiB usable volume (rounded `df -h` readings).

## Removed

| Items | Size |
|---|---:|
| Older sweep PNG images and NPZ states/motion frames: 60,510 files | 2.89 GiB |
| Four redundant adapter weight files | 1.80 GiB |
| Old `training_color_touch_v1` dataset: 50 red/blue demonstrations | 109 MiB |

The older sweeps are `scripted_color_transfer_20261013_100305_2a5212fb` and `scripted_four_color_models_20261013_132943_e69718c3`. Both were downloaded and reviewed locally. Their complete local copies remain in `outputs/`; compact cloud reports, action JSON, summaries and provenance remain on the volume. `ARTIFACT_STORAGE.json` in each cloud folder points to its local copy. Cloud image/state replay now requires restoring those binaries from the local copy.

Removed adapter weights:

- Single-cube checkpoint 300 from `panda-train-20261013-034928-1786`: archived on HF at revision `9d9e2b22098ab794f2a6161a4ad7d4b223a110ac`; [restore guide](../training/TOUCH_CHECKPOINT_ARCHIVE.md).
- Red/blue checkpoints 50, 100, 300 from `panda-train-20261013-074359-896`. Checkpoint metadata and training reports remain; the removed adapter weights were not newly archived during this cleanup.

The old red/blue dataset is redundant because the current 300-demonstration four-color dataset includes the original 50 demonstrations. It also remains available from its existing HF dataset repository.

## Retained

- Pretrained `/workspace/openvla-7b` model.
- Four-color checkpoints **50, 100, 300, 500, 1,000** from `panda-train-20261013-110217-1084`.
- Historical red/blue checkpoint **500**.
- The current 300-demonstration `training_four_color_touch_v1` dataset.
- All images, states, actions and reports for the latest 1,080-trial sweep `scripted_multi_color_models_20261013_164548_eeda379b`. Its Mac download was still in progress and was excluded from cleanup.
- Training curves, metrics, TensorBoard events, benchmark reports, diagnostics and checkpoint metadata.

Cloud receipt: `/workspace/panda-training/reports/cleanup_20261014.json`. Local receipt: `outputs/volume-cleanups/cleanup_20261014.json`.

## Future workflow preferences

Before another paid sweep pod is deployed, remind the user to choose direct Mac download or cloud-to-HF archiving. Prepare the chosen flow before deployment so the GPU does not need to wait for a slow Mac download. The reminder is saved in the root `AGENTS.md` and [cloud sweep guide](../cloud/CLOUD_SIMULATOR_SWEEPS.md).

For authorized routine cleanup, use the existing record of completed downloads/reviews rather than repeating expensive file-by-file verification without a specific concern. Keep artifacts whose download is incomplete.
