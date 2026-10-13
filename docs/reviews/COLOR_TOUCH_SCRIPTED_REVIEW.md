# Red/blue scripted simulator review — October 13, 2026

Sweep: `scripted_color_touch_20261013_080412_95b3a48c`.
Training session: `panda-train-20261013-074359-896`.

Reviewed all 80 saved trials locally, including checkpoint provenance, instruction payloads, generation settings, executed actions, contact results, state continuity and selected start/end images. No cloud access or inference/controller changes were needed for this review.

## Results

| Checkpoint | Correct-only contact | Wrong-cube contact | No contact | Clean correct completion | Both instructions correct in scene |
|---|---:|---:|---:|---:|---:|
| 50 | 7/20 (35%) | 1/20 | 12/20 | 7/20 | 2/10 |
| 100 | **15/20 (75%)** | 1/20 | 4/20 | **14/20** | **6/10** |
| 300 | 14/20 (70%) | 1/20 | 5/20 | 13/20 | 5/10 |
| 500 | 14/20 (70%) | 1/20 | 5/20 | 14/20 | 4/10 |

No both-cube or unmeasured outcomes occurred. A Completed process does not mean task success: each checkpoint has one wrong-cube trial that ended normally after contact. Correct contact accompanied by a tracking failure is included in the contact count, but excluded from clean completion.

| Checkpoint | Red target correct | Blue target correct |
|---|---:|---:|
| 50 | 3/10 | 4/10 |
| 100 | 6/10 | 9/10 |
| 300 | 5/10 | 9/10 |
| 500 | 6/10 | 8/10 |

Checkpoint 100 is the strongest observed candidate, although its lead over 300/500 is only one trial out of 20. This small sample does not establish a statistically reliable general ranking. Blue requests performed better in this particular scene set; more counterbalanced scenes are needed to distinguish color preference from geometry/reachability effects.

## Did it use the instruction?

Yes, the saved results show instruction-dependent behavior. All ten scene pairs at every checkpoint have byte-identical starting camera images. Every paired red/blue request produces a different first translation action: 10/10 pairs at each checkpoint. Mean first-action vector difference is approximately 11.14 mm at 50, 10.66 mm at 100, 9.74 mm at 300 and 9.52 mm at 500.

At checkpoint 100, scenes 1000, 1002, 1004, 1007, 1008 and 1009 reached the respective requested cube for both instructions. One of these contacts (blue request at 1004) also triggered a tracking failure. The paired evidence is stronger than testing isolated scenes, because image/starting state are held fixed while the color instruction changes.

This supports that the fine-tuned policy learned to use the red/blue instruction to guide movement. It does not establish recognition of untrained colors or retention of all pretrained language abilities; no black/purple color-transfer test was performed.

## Failures

All 28 runtime Failed trials ended with a greater-than-5-mm position tracking error after an executed movement. No API errors, IK-rejection endings or missing results were recorded. Two of these trials achieved correct contact: checkpoint 100, seed 1004 blue (8.338 mm tracking error), and checkpoint 300, seed 1008 blue (9.534 mm).

The 26 no-contact trials are all tracking-failure endings. Selected images show the hand near the intended cube but offset, or descending to the table away from a cube. No-contact does not by itself mean the model selected the wrong color; it can be a motion/alignment error.

Wrong-cube outcomes:

- Checkpoints 50, 100 and 500: seed 1001, requested red, first contact blue.
- Checkpoint 300: seed 1006, requested red, first contact blue.

Seed 1001 places the blue cube nearer the initial arm, with red farther away. In checkpoint 100 the first red request predicts a positive-Y translation, whereas the blue request predicts almost no Y translation; the trajectories are not identical despite both ultimately touching blue. Wrong-cube contact therefore establishes a task failure, but does not by itself distinguish incorrect recognition from collision/trajectory interference.

For every consecutive step in every trial, the preceding saved end qpos equals the following saved start qpos exactly. There is no reset/bounce caused by restoring old states between commands.

## Fairness and timing

- Exactly the same ten predeclared scenes, seeds 1000–1009, were used for all four checkpoints.
- Both individual cube positions in every scene were verified different from all collected dataset positions, even allowing a color exchange.
- All 80 trial checkpoint paths, instructions, target colors and actual scene positions match the evaluation plan.
- Every prediction echoed the correct instruction and `do_sample: false`.
- Setup/model preparation plus trials: **2,836.09 seconds = 47 minutes 16 seconds**.
- Summed individual trial elapsed times: **2,242.47 seconds = 37 minutes 22 seconds**.
- Mean actions among correct-only contact trials: 18.43 at checkpoint 50, 18.87 at 100, 18.43 at 300 and 18.21 at 500. These are conditional on contact, not averages counting failed trials as fast successes.

The default headless sweep uses a fixed initial arm, cube orientations and camera; these results measure transfer to new object positions within that setup, rather than broad scene generalization. The local summary confirms the sweep completed without launcher errors. It does not independently confirm the later RunPod stop; check the portal separately.

## Recommendation

Use checkpoint 100 as the current red/blue candidate. Additional training from 100 to 500 did not improve the overall contact result in this sweep, which is consistent with the training review's rising validation loss after checkpoint 50, though validation alone did not select checkpoint 100.

Before expanding training or claiming color transfer, inspect the repeated seed-1001 interference and the near-target no-contact cases. Use a further predetermined, counterbalanced scene set to verify reliability and distinguish target selection from precision control. Do not simply relax the tracking threshold or equate process completion with success.

## Saved artifacts

- [Automatic checkpoint/color and per-trial report](../../outputs/scripted_color_touch_20261013_080412_95b3a48c/REPORT.md)
- [Raw sweep results](../../outputs/scripted_color_touch_20261013_080412_95b3a48c/results.json)
- [Reviewed measurements](../../outputs/scripted_color_touch_20261013_080412_95b3a48c/review_measurements.json)
- [Selected paired-scene images](../../outputs/scripted_color_touch_20261013_080412_95b3a48c/review_montage.jpg)
- [Training-loss review](COLOR_TOUCH_TRAINING_REVIEW.md)

Detailed trial artifacts stay local under the output folder. This review and the automatically updated GPU inference comparison are pushed to GitHub.
