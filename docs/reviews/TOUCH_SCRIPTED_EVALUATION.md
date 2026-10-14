# Scripted touch evaluation review — 2026-10-13

## Result

Checkpoint 300 achieved verified finger–cube contact at all 20 predeclared unseen cube positions. It is the strongest candidate in this sweep by the touch-task criterion. Clean completions were tied at 17/20 for checkpoints 100, 300 and 500, so this is not evidence that checkpoint 300 is superior on every controller metric.

| Checkpoint | Cube contact | Contact rate | Clean completion | Contact with tracking failure | No contact |
|---|---:|---:|---:|---:|---:|
| 50 | 11/20 | 55% | 10/20 | 1 | 9 |
| 100 | 18/20 | 90% | 17/20 | 1 | 2 |
| 300 | 20/20 | 100% | 17/20 | 3 | 0 |
| 500 | 19/20 | 95% | 17/20 | 2 | 1 |

These are observed rates for 20 positions, not guarantees for arbitrary new scenes. All trials used the same initial arm configuration, camera and cube orientation; only cube XY varied inside the collection bounds. No sampling was used. Each checkpoint received exactly the same positions, seeds 1000–1019. Positions were verified not to equal any of the 50 collected demonstration locations.

## Verification

All 80 trial folders have matching checkpoint provenance, scene seeds and the trained instruction `touch the red cube.` Every prediction reports `do_sample: false`. The maximum difference between an action's ending qpos and the following action's starting qpos was exactly zero across the sweep. There were no launcher errors or incomplete trials. The user confirmed the pod stopped successfully; this review used local files only.

Positive contact is the existing task criterion: finger–cube contact with positive normal force. It does not require centered contact, holding contact or leaving the cube unmoved. Maximum cube displacement in any trial was about 48.2 mm at checkpoint 50, 24.9 mm at 100, 25.6 mm at 300 and 22.1 mm at 500; some touches therefore nudged/pushed the cube appreciably. A gentler contact task would need an additional displacement criterion. Clean completion additionally means the existing action tracking check did not reject the final movement. This review preserves both measures rather than relabeling recorded Failed trials as clean successes.

## Failure details

Every Failed trial ended after an executed action with position tracking error greater than 5 mm. No failures were API errors or IK rejections. Seven failed trials also reached the cube: checkpoint 50 seed 1007; checkpoint 100 seed 1015; checkpoint 300 seeds 1003, 1008 and 1015; checkpoint 500 seeds 1003 and 1008. Their final executed action and contact remain in `execution.json`, although the completed-action count excludes that action.

Checkpoint 50 missed nine positions. Images show examples descending beside the cube rather than correcting sufficiently before reaching table height. At checkpoint 100 the remaining noncontact cases were seeds 1003 and 1016. Checkpoint 300 reached both. Checkpoint 500 missed seed 1000, which earlier checkpoints reached. Therefore performance does not improve monotonically with every additional training update.

The shared seed-1003/1008/1015 tracking failures at later checkpoints show that task attainment and controller tracking must be assessed separately. Contact can obstruct a commanded descent. The recorded threshold error alone does not prove a particular collision cause, and the controller threshold was not changed for this review.

## Speed and training interpretation

Mean action count among contact trials: checkpoint 50 19.18; checkpoint 100 18.44; checkpoint 300 18.80; checkpoint 500 18.89. This is conditional on reaching contact, so checkpoint 50's number excludes its nine misses.

Across all recorded executed actions, mean GPU prediction time was 0.345 seconds and mean image-to-next-state cycle time was 1.269 seconds. These are descriptive end-to-end measurements from this headless scripted sweep, including startup requests, not a separately warmed inference benchmark. Headless physics execution is not paced by viewer animation, so avoid treating its cycle time as directly equivalent to prior live-window trial timings.

Training validation loss was lowest at update 40 and rose afterward, while touch performance in this sweep improved substantially from checkpoint 50 to 300. Teacher-forced action-token loss and actual closed-loop task success rank checkpoints differently here. Fine-tuning has learned useful behavior that transfers to new cube locations within the tested setup.

## Recommendation

Use checkpoint 300 as the current touch-task candidate. Before claiming general reliability, evaluate it on another predetermined unseen-position set, without choosing positions based on success. There is no evidence here that training longer than 500 updates is needed. Once satisfied with touch reliability, progress to a separately defined grasp task; this sweep does not establish grasping, lifting or arbitrary language instruction following.

No inference, controller, model or training code was changed during the review.

## Saved artifacts

- [Automatic per-trial report](../../outputs/sweeps/scripted_touch_20261013_052521_7a043028/REPORT.md)
- [Raw sweep results](../../outputs/sweeps/scripted_touch_20261013_052521_7a043028/results.json)
- [Reviewed measurements](../../outputs/sweeps/scripted_touch_20261013_052521_7a043028/review_measurements.json)
- [Selected first/middle/final images](../../outputs/sweeps/scripted_touch_20261013_052521_7a043028/review_montage.jpg)

Local detailed outputs are ignored by Git. This review and the updated GPU inference comparison are committed to GitHub.
