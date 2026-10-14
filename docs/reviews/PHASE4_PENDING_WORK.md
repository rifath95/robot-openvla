# Phase 4 follow-up work

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

## Current color-selection and lifting evaluation

Use lift checkpoints 500 and 1000 from the 300-trajectory run, fine-tuned from
pretrained OpenVLA on a single red cube. No new color training is performed.
The 160-trial plan tests two cubes, sometimes neither red. Correct/wrong lift
and no lift are measured separately. See
[launch and transfer instructions](../cloud/COLOR_LIFT_SWEEP.md).
