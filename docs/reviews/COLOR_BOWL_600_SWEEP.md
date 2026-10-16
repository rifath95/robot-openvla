# 600-demo colored bowl model: 800-trial review

Reviewed October 16, 2026. Source: `outputs/sweeps/scripted_color_bowl_models_20261015_112315_46cfdb52/`.
Training session: `panda-train-color-bowl-20261015-094416-4344d31f`.
600 demonstrations, including one- and two-cube scenes, trained for 6,000 updates.
Training colors: red, blue, green, yellow. Black, purple and orange were not in this fine-tuning dataset.

All 800 frozen trial identities validate. All 800 restored summaries agree with the saved final outcomes. No infrastructure failures or launcher error were reported. Controller failures remain in the denominator and overlap task outcomes.
Computed aggregates: [review data](assets/color_bowl_600_sweep_review.json).

## Checkpoints

| Checkpoint | Trials | Target lifted ≥8 cm | Correct placements | Placement rate | Controller failures |
|---|---:|---:|---:|---:|---:|
| 1,000 | 200 | 19 | 8 | 4.0% | 93 |
| 2,000 | 200 | 57 | 20 | 10.0% | 21 |
| 4,000 | 200 | 95 | 58 | 29.0% | 39 |
| 6,000 | 200 | 107 | 80 | 40.0% | 28 |

Success requires prior lift ≥8 cm, settled requested cube within 4.5 cm of bowl center and 4 mm of resting height, and hand ≥8 cm away. No wrong/both-cube placements were recorded. Checkpoint 6,000 is the best overall checkpoint in this sweep, pending the separate continuation sweep.

On the 200 paired tasks, 6,000 gains 48 successes over 4,000 and loses 26; both succeed on 32. An exact two-sided paired discordance test gives p≈0.014 (exploratory, without adjustment for checkpoint selection). Improvement is not uniform: single-cube qualifying lifts decrease from 30 to 28, while completed placements increase from 16 to 23.

## Checkpoint 6,000 by scene

| Scene | Correct placements | Rate | Earlier red-only model at 3,000 |
|---|---:|---:|---:|
| One red cube | 12/20 | 60.0% | 10/20 |
| One black cube | 11/20 | 55.0% | 4/20 |
| Two cubes: red and black | 18/40 | 45.0% | 9/40 |
| Two cubes: black and purple | 14/40 | 35.0% | 9/40 |
| Four cubes with red | 11/40 | 27.5% | 0/40 |
| Four cubes without red | 14/40 | 35.0% | 0/40 |
| Total | 80/200 | 40.0% | 32/200 |

Two-cube aggregate: 32/80 (40%); four-cube aggregate: 25/80 (31.25%).
The older and newer sweeps use matching task identities; the requested cube's initial position matches exactly in all 200 comparisons. The new model gains 58 successes and loses 10 relative to the old best checkpoint. This supports the combined expanded-data/training setup, but does not isolate the causal contribution of dataset size, colors, multiple objects, or update count.
The historical base scored 0/200, but used old 300-demo action statistics; it was not rerun with the newer statistics.

## Color retention and selection

At 6,000, the first qualifying lift was of the requested cube in all 79 multi-cube trials that achieved a qualifying lift; 81/160 multi-cube trials had none. All 49 multi-cube trials lifting an unseen-color target lifted the requested color first. These are conditional measures of selection among achieved lifts, not 100% task/color accuracy across all trials.

| Requested color | Correct placements | Rate |
|---|---:|---:|
| Red (seen) | 24/50 | 48.0% |
| Green (seen) | 9/20 | 45.0% |
| Black (unseen) | 33/80 | 41.25% |
| Purple (unseen) | 9/40 | 22.5% |
| Orange (unseen) | 5/10 | 50.0% |

Unseen targets overall: 47/130 placements (36.2%), including 36/110 in multi-cube scenes. Scene difficulties and sample sizes differ, so these rates do not rank intrinsic color understanding. Blue and yellow were trained but not requested in this sweep.

Seven two-cube layouts succeeded for both requested colors after separate resets: red/black seeds 1001, 1005, 1008, 1015, 1017; black/purple seeds 1013 and 1018. For black/purple seed 1013, the initial input images are byte-identical, yet changing the instruction selects and places each requested cube. Before/lift/final frames were visually inspected for that pair and a successful four-cube orange request. This is strong behavioral evidence of retained unseen-color grounding. It does not quantify pretraining versus fine-tuning contributions; fine-tuning can adapt how pre-existing visual/language features drive actions.

## Remaining limitations

At 6,000: 80 correct placements, 27 target lifted but not placed, and 93 no qualifying lift. Of 28 controller failures, 27 overlap no-lift trials. All 27 lifted-but-not-placed trials finish outside the required bowl-center radius; additional overlapping failures include height tolerance (10), unsettled cube (2), and hand proximity (4). A red single-cube example at seed 1003 looks near/in the bowl from the rendered view but fails the geometric center requirement. Do not infer success from appearance alone.

No wrong cube was lifted at 4,000 or 6,000. Earlier checkpoints each recorded one wrong-cube lift. Grasping, transport, placement geometry and controller tracking remain the main observed limitations. Compare the restored continuation checkpoints before choosing further training or data changes.

## Timing

Simulator stage: 10 h 7 min 26 s. Sum of per-trial times: 10.04 h. Packing/upload/verification: 22 min 3 s. Recorded setup-to-final-report stages span about 10 h 31 min; this excludes any provisioning or final shutdown delay.
