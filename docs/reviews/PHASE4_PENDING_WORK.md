# Phase 4 follow-up work

## Deferred bowl initialization fix

User requested October 14, 2026: fix this when returning to bowl placement,
not during preparation of the color-lifting sweep.

In `scripted_place_bowl_models_20261014_111236_f42bb9bf`, seeds **1003 and 1008**
failed before any model prediction, for every checkpoint, with `Starting state
is non-finite or still moving after settling`. These are scene-initialization
failures, not evidence of policy failure. Check object reset/settling order and
collision state, retain the declared layouts rather than replacing them with
easier positions, and preflight all bowl layouts before expensive model loading.
Preserve the current controller and replay compatibility. Re-evaluate these two
layouts after the fix; the four successful checkpoint-500 placements remain valid.

## Current color-selection and lifting evaluation

Use lift checkpoints 500 and 1000 from the 300-trajectory run, fine-tuned from
pretrained OpenVLA on a single red cube. No new color training is performed.
The 160-trial plan tests two cubes, sometimes neither red. Correct/wrong lift
and no lift are measured separately. See
[launch and transfer instructions](../cloud/COLOR_LIFT_SWEEP.md).
