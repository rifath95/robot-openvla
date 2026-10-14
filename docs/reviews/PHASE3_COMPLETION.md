# Phase 3 completion: Panda fine-tuning and color-guided touching

**Status: completed October 13, 2026.** Phase 3 demonstrated that OpenVLA can be adapted to our simulated Franka Panda, execute a learned closed-loop touch skill, follow color instructions, and generalize to colors absent from fine-tuning. The final checkpoint achieved 95% correct contact on seen colors and 82.5% on unseen colors in the final controlled simulator sweep.

The phase originally targeted pick-and-place. That objective was narrowed after unsuccessful grasp/lift experiments to establish a simpler, measurable adaptation result. Completion therefore means the training/evaluation pipeline and color-guided touch milestone are demonstrated. It does **not** mean the original pick-and-place objective has been achieved.

## Starting point and approach

Phases 1 and 2 established MuJoCo simulation, Panda end-effector control, local OpenVLA inference and a Mac-to-cloud observation/action loop. Phase 3 added our own demonstration data, Panda action normalization, LoRA adaptation and controlled evaluation against the pretrained model.

The model receives the current camera image and instruction, and predicts seven action values: position deltas, rotation deltas and a gripper command. The controller executes the bounded action through physics, captures the next image and repeats. No image-history input or new model architecture was introduced. We integrated the official OpenVLA training path and changed our data, controller, launchers and evaluation tools around it.

## Experiments and what they taught us

| Stage | What we did | Observed result and lesson |
|---|---|---|
| Initial pick-and-place | Collected 50 scripted demonstrations; ran 20-update smoke training and 1,000-update LoRA training. | Training, validation and checkpoint saving worked. Some rollouts approached/gripped near the cube, but reliable lift/place was not established. |
| Broader pick-and-place | Expanded to 250 demonstrations with wider starting poses and object positions; trained for 5,000 updates. | Dataset and training expansion alone did not establish task success. Lower loss was insufficient evidence of successful robot behavior. |
| Visible bowl destination | Added a plate, then bowl, adjusted camera/gripping and collected 300 bowl demonstrations; trained for 5,000 updates. | A visible goal avoided the relative-destination ambiguity, but learned rollouts still stalled. It did not solve all phase-transition or control problems. |
| Controller and supervision diagnosis | Checked saved actions/images, fixed pose-hold/bounce behavior, replayed recorded actions through physics, and compared teacher-forced errors with normal generated actions. | Action replay verified that expert actions could execute the task. It did not prove that a stateless image policy could infer the script's hidden phase. Repeated hold/open startup labels could leave a policy in an unchanged observation. |
| Smaller bowl pilot | Revised demonstrations and ran a 30-trajectory, 500-update pilot with frequent checkpoints. | More movement appeared, including orientation attempts, but reviewed trials did not complete pickup or placement. |
| Single-cube touch | Simplified to fixed starting arm pose, varying cube XY, and instruction `touch the red cube.` Expanded from 10 to 50 demonstrations. | Checkpoint 300 contacted the cube at all 20 predeclared new positions; 17/20 completed without a tracking failure. This established a learned touch skill. |
| Red/blue selection | Added two visible cubes and paired demonstrations with different target instructions; trained on 50 demonstrations. | Checkpoint 100 reached the requested cube in 15/20 trials. Identical initial images with changed instructions produced successful selection of different cubes in paired scenes. |
| Reserved-color transfer | Evaluated the red/blue-trained checkpoints on additional colors. | The earlier checkpoint 500 reached the requested black/purple/orange/white cube in 23/40 trials. This provided initial evidence of generalization beyond the fine-tuning colors. |
| Four-color expansion | Collected 300 demonstrations using red, blue, green and yellow; trained fresh from the pretrained model for 1,000 updates. | The final 480-trial sweep showed much stronger task success and unseen-color generalization, with checkpoint 1,000 the best observed rollout model. |

These stages are experiments with different datasets and task scopes, not one continuous training run. Earlier failures remain part of the record. The documented startup-label problem is a demonstrated issue, not proof that it explained every grasp/lift failure.

Evidence: [initial smoke](../training/SMOKE_REVIEW.md), [broader data](../datasets/BROAD_PANDA_DATASET.md), [v2 trials](V2_LOCAL_AUDIT.md), [bowl stall audit](BOWL_STALL_CAUSE_AUDIT.md), [bowl pilot](BOWL_V2_CHECKPOINT_TRIALS.md), [single-cube sweep](TOUCH_SCRIPTED_EVALUATION.md), [red/blue sweep](COLOR_TOUCH_SCRIPTED_REVIEW.md), and [earlier reserved-color sweep](UNSEEN_COLOR_TRANSFER_REVIEW.md).

## Final dataset and training

Task: two differently colored cubes on a table; instruction `touch the COLOR cube.` Initial arm pose, gripper state, camera and cube orientation are fixed; cube positions and color assignments vary.

The final collection has 150 paired scenes, demonstrated twice per scene, covering all six pairs among red, blue, green and yellow. Both target demonstrations for a scene stay in the same split. The original 50 red/blue demonstrations were preserved byte-for-byte; 250 new demonstrations expanded the collection to 300. All new demonstrations passed action replay and source-input checks.

| Split | Trajectories | Image/instruction/action examples |
|---|---:|---:|
| Training | 200 | 3,544 |
| Validation | 50 | 879 |
| Test | 50 | 888 |

The test split was held out from training and validation. The final simulator sweep used separately seeded positions; its rollout counts below are not results on the 888 test examples.

| Setting | Final run |
|---|---|
| Base model | `openvla/openvla-7b`; fresh LoRA adaptation |
| Training session | `panda-train-20261013-110217-1084` |
| Dataset bundle | `training_four_color_touch_v1` |
| GPU | NVIDIA A40 |
| Microbatch / accumulation | 8 / 1; effective batch 8 |
| Optimizer updates / example exposures | 1,000 / 8,000; approximately 2.26 passes worth of training examples |
| Learning rate / LoRA rank | 0.0001 / 32 |
| Quantization / image augmentation / LoRA dropout | Off / off / 0 |
| Saved checkpoints | 50, 100, 300, 500, 1,000 |
| Full validation | Updates 0, 50, 100, 300, 500, 1,000; 879 examples each |
| Recorded supervised run time | 48 minutes 12 seconds, including setup/training/validation/checkpoint work |
| Approximate GPU charge at $0.59/hour | $0.47 for that recorded interval; deployment and other pod time excluded |

Hugging Face dataset: `Maxalderone/robot-openvla-panda-four-color-touch`. The [dataset review](FOUR_COLOR_TOUCH_300.md), [training guide](../training/FOUR_COLOR_TOUCH_TRAINING.md) and [training review](FOUR_COLOR_TOUCH_TRAINING_REVIEW.md) record preparation, configuration, metrics and timings.

## Final evaluation and achievements

All 480 trials completed: pretrained base plus five checkpoints, four color pairs, ten scenes per pair and two instructions per scene. Both instructions received identical starting images. All six models also received byte-identical initial images for each scene/target input. Color assignment alternated across physical positions. Sampling was off.

Seen pairs: red/blue and green/yellow. Reserved unseen pairs: black/purple and orange/white. These four reserved color names/appearances were not demonstrated in fine-tuning.

| Model | Seen correct contact / 40 | Unseen correct contact / 40 | All correct contact / 80 |
|---|---:|---:|---:|
| Pretrained base | 5 (12.5%) | 7 (17.5%) | 12 (15.0%) |
| 50 | 16 (40.0%) | 4 (10.0%) | 20 (25.0%) |
| 100 | 23 (57.5%) | 24 (60.0%) | 47 (58.8%) |
| 300 | 36 (90.0%) | 29 (72.5%) | 65 (81.2%) |
| 500 | 37 (92.5%) | 30 (75.0%) | 67 (83.8%) |
| **1,000** | **38 (95.0%)** | **33 (82.5%)** | **71 (88.8%)** |

![Final evaluation results](assets/four_color_model_sweep_results.png)

At checkpoint 1,000, 34 unseen-color trials touched a cube: 33 touched the requested cube and one touched the wrong cube, giving 97.1% choice accuracy among contacts. Six made no contact. All 71 correct contacts across seen/unseen colors completed cleanly. Both target instructions succeeded in 14/20 unseen paired scenes: with the same initial image, changing the instruction changed which cube was successfully touched.

Compared with the earlier red/blue checkpoint 500 on the same reserved color pairs, unseen success increased from 57.5% to 82.5%, and wrong contacts decreased from six to one. Dataset size, palette, normalization and training duration changed, so this comparison does not isolate the contribution of any one change.

**Interpretation:** fine-tuning taught the Panda touch behavior and how to use color instructions for that behavior. It supplied no demonstrations naming the reserved colors. Successful instruction-dependent selection of those colors is strong evidence that existing pretrained color knowledge remained usable after adaptation. The experiment does not measure the internal source of each prediction or isolate the contributions of pretrained representations and generalization learned during adaptation.

Checkpoint 1,000 is the best observed rollout model in this sweep, despite checkpoint 100 having the lowest teacher-forced validation loss. Closed-loop task evaluation was essential for selecting the demonstrated model. The [full sweep review](FOUR_COLOR_MODEL_SWEEP_REVIEW.md) records per-color outcomes, statistical assumptions, comparison caveats and audits.

## Infrastructure delivered

- Replay-verified demonstration collection, scene-level split planning, RLDS packaging and training-only normalization.
- Hugging Face dataset transfer with checksums, plus a documented checkpoint archive/restore workflow and provenance metadata.
- LoRA training with saved metrics, loss curves, validation, checkpoint schedules and timings.
- Cross-GPU training capacity/throughput and inference latency/cost reports, including software stack and ECC details.
- Resident inference for repeated interactive trials, separate scripted sweeps, resume support and a cloud-only simulator path that continues while the Mac disconnects.
- Detached training/sweep supervisors with inner/outer runtime limits, persistent artifacts and automatic pod-stop requests. API/network failures can prevent stopping; recorded artifacts and portal status remain the evidence.
- Repository organization, documentation and targeted checks of the moved entry points and cloud workflows.

## Saved artifacts and reproduction

| Artifact | Location |
|---|---|
| Final demonstrations and replay commands | `datasets/panda_four_color_touch/`; `WATCH_TRAJECTORIES.md` |
| Portable local dataset | `datasets/panda_four_color_touch/training_four_color_touch_v1/` |
| Training configuration | `training/four_color_touch_300_1000.json` |
| Downloaded training reports/curves | `outputs/training/phase3_training/panda-train-20261013-110217-1084/` |
| Final checkpoint on the network volume | `/workspace/panda-training/sessions/panda-train-20261013-110217-1084/runs/training/checkpoint_001000` |
| Sweep profile | `inference/sweep_profiles/four_color_all_models.json` |
| Downloaded 480-trial evidence | `outputs/sweeps/scripted_four_color_models_20261013_132943_e69718c3/` |
| Tracked final review and plot | `docs/reviews/FOUR_COLOR_MODEL_SWEEP_REVIEW.md`; `docs/reviews/assets/four_color_model_sweep_results.png` |

Generated demonstrations, raw outputs and model weights are excluded from Git. Reports and selected plots are committed. Local report downloads do not include adapter weights; those remain on the volume unless separately archived. Storage paths describe the completion snapshot and may change after future cleanup.

Follow the [final training guide](../training/FOUR_COLOR_TOUCH_TRAINING.md) and [model sweep guide](../cloud/FOUR_COLOR_MODEL_SWEEP.md) to reproduce the experiment. Historical pick-and-place instructions remain in the [original Phase 3 training guide](../training/PHASE3_TRAINING.md).

## Limits and handoff to Phase 4

This is a simulated touch result, not grasping, lifting, placement or real-robot success. Contact need not be centered or gentle and may move the cube. Initial arm pose, camera and cube orientation are fixed; variation is limited to the declared scene workspace and color assignments. Ten scenes per pair limit precision, and reusing this suite to choose checkpoints makes fresh evaluation desirable before broader claims.

Phase 4 started October 14 with a [ten-demonstration single-cube grasp/lift pilot](LIFT_CUBE_FIRST_TEN.md), then [expanded to 50 verified demonstrations](LIFT_CUBE_50.md) after review. Fine-tuning has not started for this task. Preserve the Phase 3 model and evaluation as a reference, and measure both the new skill and retention of color-guided touching.
