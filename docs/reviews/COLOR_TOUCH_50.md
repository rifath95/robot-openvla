# Red/blue touch dataset expanded to 50 demonstrations

The original ten reviewed demonstrations are preserved. Forty were added, giving 25 paired scenes: one red-target and one blue-target demonstration from exactly the same initial scene per pair. The new positions are independently seeded draws within the established touch region, conditioned on at least 12 cm separation between cubes. Positions are declared before collection; failed demonstrations are never replaced by easier positions. All planned demonstrations succeeded.

| Split | Trajectories | Red targets | Blue targets | Action examples |
|---|---:|---:|---:|---:|
| Training | 34 | 17 | 17 | 609 |
| Validation | 8 | 4 | 4 | 136 |
| Test | 8 | 4 | 4 | 142 |
| Total | 50 | 25 | 25 | 887 |

Each scene pair remains entirely within its assigned split. Instructions are exactly `touch the red cube.` and `touch the blue cube.` Initial robot states, gripper opening, cube orientations and camera geometry are fixed as in the reviewed demonstrations. Both cubes appear in every scene.

## Verification

- All 50 action sequences passed continuous physics replay and reached the instructed cube.
- Wrong-cube contact is rejected during recording and replay verification.
- First observation PNG bytes are identical between the red/blue demonstrations of every pair.
- Initial robot positions and velocities match across all 50 trajectories.
- Original episode fingerprints match, and all 1,180 files in the reviewed ten remain unchanged.
- All action labels have nonzero translation, zero rotation and a closed-gripper value.
- Every saved before/after image was validated at 640×480 pixels, with no instruction overlay burned into the images.
- Maximum recorded-versus-replayed joint-position difference: 4.885e-15.
- Sixteen scene, replay and expansion-plan regression checks passed.

## Review

- [All trajectories in training, validation, test order](../../datasets/panda_color_touch/WATCH_TRAJECTORIES.md)
- [All 25 paired scenes, with starting and red/blue ending images](../../datasets/panda_color_touch/REVIEW_ALL_PAIRS.jpg)
- [Selected new pairs](../../datasets/panda_color_touch/NEW_PAIRS_PREVIEW.jpg)
- [Full audit](../../datasets/panda_color_touch/REVIEW_AUDIT.json)
- [Dataset details](../../datasets/panda_color_touch/README.md)

Example new demonstration:

```bash
.venv/bin/python datasets/panda_color_touch/train/train_007/replay_actions.py
```

The paired blue-target demonstration is `train_008`. The viewer displays the instruction in the upper right; camera images remain plain RGB. Space starts replay.

Dataset expansion is complete. No portable training bundle, Hugging Face upload, training configuration or cloud inference for this task has been launched. Dataset contents stay local and are ignored by Git; code and this review are tracked.
