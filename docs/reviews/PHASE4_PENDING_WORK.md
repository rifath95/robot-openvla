# Phase 4: potential future explorations

**Status: proposed, not started. Updated October 17, 2026.** All completed work
in this repository, including lift and bowl experiments previously labeled
Phase 4, is now part of [Phase 3](PHASE3_COMPLETION.md).

## Research direction

Explore whether the same pretrained OpenVLA adapts more successfully in a
different simulation/robot environment, potentially LIBERO, with a larger
existing demonstration dataset from Hugging Face. Compare task completion and
retention of instruction/color knowledge with our Panda/MuJoCo findings.
Neither the environment nor a particular dataset has been selected. More data
or a different environment is a hypothesis to test, not a promised improvement.

## Proposed sequence

1. Select an environment and compatible demonstration dataset. Check robot
   embodiment, camera views, instruction coverage, action semantics, controller,
   normalization and reset behavior before training.
2. Reproduce expert action execution and test the pretrained model as a baseline.
3. Run a small fine-tuning pilot before committing to a long training job.
4. Evaluate actual closed-loop success on held-out layouts/instructions, with
   seen and unseen attributes where the dataset permits a meaningful test.
5. Compare checkpoints throughout training, including an earlier stopping point.
   Preserve full training state and make reports independently downloadable.

Carry forward the simulator audits, replay, provenance, cloud delivery and
runtime safeguards developed in Phase 3. Keep recovery-assisted demos separate
from unassisted model evaluations.

## Historical notes now belonging to Phase 3

The notes below describe completed Phase 3 preparation under its former phase
label. They are retained for context, not a pending Phase 4 deployment plan.

## Bowl initialization fix implemented

Implemented October 14, 2026 after the color-lifting sweep started.
Both cube and bowl are positioned at the declared layout before settling the
robot, with forward kinematics refreshed. Previously the cube stayed at its
home location while the relocated bowl could intersect it. Contact inspection
confirmed cube/bowl overlap in the intermediate state for seeds 1003 and 1008.
All 301 collection candidates and all 20 prior sweep layouts pass
settling and cube-position checks. Future bowl sweeps preflight every declared
layout before loading model weights. Controller, geometry and camera are unchanged.
Saved demonstrations replay their stored initial physics state and remain compatible.

In `scripted_place_bowl_models_20261014_111236_f42bb9bf`, seeds **1003 and 1008**
failed before any model prediction, for every checkpoint, with `Starting state
is non-finite or still moving after settling`. These were scene-initialization
failures, not evidence of policy failure. The next policy sweep should re-evaluate
these two unchanged layouts after the fix; the four successful checkpoint-500
placements remain valid. Initialization audit:
[321 layouts](assets/place_bowl_initialization_audit.json).

## Historical color-selection and lifting evaluation

Use lift checkpoints 500 and 1000 from the 300-trajectory run, fine-tuned from
pretrained OpenVLA on a single red cube. No new color training was performed for that experiment.
The completed 160-trial sweep tested two cubes, sometimes neither red. Correct/wrong lift
and no lift are measured separately. See
[launch and transfer instructions](../cloud/COLOR_LIFT_SWEEP.md).
