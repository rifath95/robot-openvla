# Uninterrupted colored bowl training: 800-trial sweep

Reviewed October 17, 2026. Sweep:
`scripted_color_bowl_models_20261016_121931_8f24640a`.
Training session: `panda-train-color-bowl-20261016-084536-c17080be`.

The best observed checkpoint from this run is **6,000: 61/200 placements
(30.5%)**. Additional uninterrupted training did not improve this suite:
checkpoint 12,000 reaches **40/200 (20%)**. The original separately trained
checkpoint 6,000 remains the strongest observed model at **80/200 (40%)**.
Keeping optimizer state continuous did not produce the hoped-for gain in this
experiment. This does not establish that optimizer resets are beneficial or
that they caused the earlier continuation's performance drop.

Computed aggregates: [review measurements](assets/uninterrupted_color_bowl_sweep_review.json).

## Completed trials and comparison controls

- All 800 planned trial identities validate: four checkpoints, 200 tasks each.
  Every restored summary agrees with the saved placement/outcome/cube records.
  No missing trials, launcher error or infrastructure failures were found.
- Sampling is off in all 800 trials; automatic or manual playground recovery
  is not part of the sweep. Every trial permits 120 action attempts and starts
  from a fresh reset. Instructions match the requested color.
- The planned layouts and actual initial cube positions match the earlier
  comparison suite. Scene XML and controller fingerprints match the original
  6,000-update sweep; MuJoCo is 3.14.0.
- Both runs use an A40, bfloat16, eager attention and in-VRAM adapter merging.
  PyTorch/Transformers versions, instruction processing, action normalization
  key and all action statistics match at all four checkpoints.
- Original versus new checkpoint-6,000 first images are byte-identical in
  **198/200** matched trials. In that identical-image subset, the original model
  succeeds 79 times versus 61 for the new model. Both sweeps rendered in the
  cloud; the Mac-versus-cloud rendering difference does not explain this overall
  comparison. This audit does not claim pixel-identical images at every later step.
- Executed actions report physics frozen during prediction. Recorded controller
  failures remain included in task-success denominators.

## Checkpoint results

| New checkpoint | Requested cube lifted ≥8 cm | Correct placements | Success | Lifted target, not placed | No lift | Wrong/both placements | Controller failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6,000 | 90/200 | **61/200** | **30.5%** | 29 | 108 | 0 | 29 |
| 8,000 | 74/200 | 54/200 | 27.0% | 20 | 125 | 1 | 36 |
| 10,000 | 94/200 | 52/200 | 26.0% | 42 | 105 | 1 | 26 |
| 12,000 | 74/200 | 40/200 | 20.0% | 34 | 125 | 0 | 23 |

Columns are separate measures and need not sum to 200: controller failures
overlap task outcomes; wrong qualifying lifts are separate from target lifts.
There are two `wrong_cube_lifted` final outcomes at 6,000 and one at 12,000.
All correct placements completed cleanly.

| Matched scene group | Trials/model | 6,000 | 8,000 | 10,000 | 12,000 |
|---|---:|---:|---:|---:|---:|
| One red cube | 20 | 11 | 13 | 12 | 12 |
| One black cube | 20 | 9 | 10 | 8 | 4 |
| Two cubes, red present | 40 | 16 | 12 | 14 | 10 |
| Two cubes, red absent | 40 | 11 | 11 | 8 | 11 |
| Four cubes, red present | 40 | 7 | 5 | 5 | 0 |
| Four cubes, red absent | 40 | 7 | 3 | 5 | 3 |

The decline is strongest with four cubes: 14/80 at 6,000, falling to 3/80 at
12,000. One-red-cube success is roughly maintained. This is not a uniform loss
of the ability to grasp/place in every scene.

## Compare with the original run and fresh-optimizer continuation

| Update | Original run / its fresh-optimizer continuation | New uninterrupted run |
|---|---:|---:|
| 6,000 | **80/200 (40.0%)** | 61/200 (30.5%) |
| 8,000 | 55/200 (27.5%) | 54/200 (27.0%) |
| 10,000 | **64/200 (32.0%)** | 52/200 (26.0%) |
| 12,000 | **64/200 (32.0%)** | 40/200 (20.0%) |

The new run was already different at update 6,000, before the point where the
earlier experiment reset its optimizer. Exact original checkpoint-zero LoRA
matrices were restored, but all original RNG/input-stream states were not
recorded. The training streams/execution trajectories cannot be assumed
identical. These are comparisons between realized runs, not a controlled
same-checkpoint branch that isolates optimizer-state effects.

In the new run, 12,000 gains 15 successes and loses 36 compared with its own
6,000; 25 successes are shared. Relative to the original 6,000, new 12,000 gains
13 and loses 53, with 27 shared successes. New 12,000 versus the earlier
12,000 continuation gains 20 and loses 44. These are substantial trajectory
changes rather than the same failures occurring everywhere.

Exact two-sided paired binomial tests on discordant tasks give p=0.0046 for
new 6,000 versus new 12,000, and p=0.0037 for the two 12,000 models. These are
exploratory, unadjusted comparisons on a suite reused to select checkpoints;
related scene/target trials are not independent samples of arbitrary future
tasks. They do not establish universal model rankings.

Training loss decreased from 0.9887 to 0.6100 and teacher-forced token accuracy
rose from 69.6% to 79.9% over the corresponding 1,000-update windows. That
improvement did not translate into higher closed-loop placement success.
This is consistent with a fit/rollout mismatch; proven overfitting, color
forgetting or an optimizer-reset cause cannot be inferred from these runs alone.

## Color knowledge and remaining failure modes

| Checkpoint | Correct first qualifying lift / multi-cube trials with any lift | Multi-cube trials with no qualifying lift | Unseen-target placements /130 |
|---|---:|---:|---:|
| 6,000 | 65/67 (97.0%) | 93/160 | 33 (25.4%) |
| 8,000 | 48/49 (98.0%) | 111/160 | 33 (25.4%) |
| 10,000 | 68/69 (98.6%) | 91/160 | 27 (20.8%) |
| 12,000 | 51/52 (98.1%) | 108/160 | 23 (17.7%) |

The lift-choice measure is conditional on reaching an 8 cm lift, not overall
task success and not an evaluation of every near-contact choice. Black, purple
and orange were absent from fine-tuning. The high correct-lift fraction remains
consistent with usable color knowledge, but is not evidence that motor execution
is reliable or that every piece of knowledge has been preserved. Blue/yellow
are trained colors but not requested by this suite. Orange has only 10 trials
per model; zero orange placements at some checkpoints is not proof of forgetting.

At 12,000, 34 requested cubes were lifted but not placed. Of these, 31 finished
outside the bowl-center radius; 26 failed resting-height tolerance, 21 had the
hand too close, and 6 were unsettled. These failures overlap. At 10,000 all 42
lifted-but-unplaced targets finished outside the bowl-center radius. Transport,
release and clearance remain significant weaknesses alongside failure to lift.

Visual checks of new checkpoint 12,000:

- Two cubes, red present, seed 1008, red requested: the final image shows red
  back on the table away from the bowl after a prior qualifying lift, with the
  hand open above it. The saved result is `target_lifted_not_placed`, not success.
- Two cubes, red absent, seed 1001, purple requested: purple is inside the bowl
  and the hand has retreated; the physical placement criteria report success.

## Timing, delivery and recommendation

Simulator stage: **10 h 24 min 2 s**. Packing/upload/verification:
**22 min 50 s**. Recorded launch-to-final-report span: approximately
**10 h 48 min 17 s**, within the 12-hour planning budget, 15-hour inner and
18-hour outer limits. Mean trial durations range from 42.3 to 49.5 seconds.
HF delivery report confirms 800/800 trials saved and verified before pod stop;
the archive restored and validated locally. The compressed manifest totals
about 18.67 GiB.

Retain **original checkpoint 6,000** as the strongest observed reference; its
volume copy was intentionally removed after HF backup and can be restored from
HF for comparison. Use **new checkpoint 6,000** as the best candidate from the
uninterrupted run. The playground currently offers only the new run's models;
this review does not silently change its default or access a pod.

Do not assume more same-setting training will help. Before another long run,
compare the best original/new checkpoints on fresh matched scenes and inspect
grasp/transport/release failures. A shorter lower-learning-rate pilot or targeted
failure demonstrations are more informative next experiments. Test recovery
separately and label interventions; this sweep measured greedy policy behavior.

Full local evidence: `outputs/sweeps/scripted_color_bowl_models_20261016_121931_8f24640a/`.
Open its `REPORT.md` for direct replay links to individual trials.
