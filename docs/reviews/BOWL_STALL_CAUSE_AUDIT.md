# Why the bowl policy stalls: local source and rollout audit

Audited 2026-10-13. Uses the existing 200 training demonstrations, bundle statistics,
actual cached OpenVLA processor, saved training configuration, and three
checkpoint-5,000 rollouts. No new cloud inference, training or dataset replacement
was performed. Measurements: `outputs/diagnostics/bowl_stall_label_audit.json`.

## Confirmed immediate mechanism

The repeated policy output is exactly the decoded token sequence for the expert
label `[0, 0, 0, 0, 0, 0, 1]`: zero arm movement and open gripper. Quantization
makes it look like tiny nonzero deltas, including X +0.0211 mm and Y -0.0140 mm,
and gripper 0.996078. It is not an arbitrary corrupt seven-value vector.

| Checkpoint-5,000 trial seed | Exact zero/open token pattern | Completed actions |
| --- | ---: | ---: |
| 0 | 10 | 10 |
| 1 | 6 | 8 |
| 2 | 5 | 6 |

Its translation norm is about 0.0253 mm, below the IK tolerance of 0.1 mm. After
physical settling it produces effectively no motion; the gripper is already open.
The late seed-0 input images are byte-for-byte identical, as are late inputs in
seeds 1 and 2. Deterministic generation then repeats the same action. Thus the
loop is a stable hold, not IK rejecting substantial movement or resetting poses.

## Confirmed dataset-design issue

Every one of the 200 training trajectories begins with `open_for_pickup` and the
same expert label `[0, 0, 0, 0, 0, 0, 1]`. This is unconditional in
`data_tools/bowl_demonstrations.py`: the scripted planner issues an opening command
even when the starting gripper is already fully open. It subsequently advances
to `clear_start` (110 trajectories) or `orient_for_grasp` (90), using its internal
execution phase rather than requiring task-visible progress from that first action.

Fifty training starts explicitly have a fully open gripper. Their first action
has no intended arm or gripper change. In the original controller's recordings,
it nevertheless creates small transient motion: median net grasp-point shift
0.275 mm and rotation 0.033 degrees. Subsequent labels can request substantial
movement or rotation after this incidental change. Before/after images for these
starts are similar: median RGB difference after a simple 224-pixel resize is
0.154/255; median cosine similarity of actual fused processor pixel tensors is
0.99959. **None of these original tensor pairs is literally identical.** Whole-
image similarity includes a large fixed background and is not a learned visual
embedding measure. These facts support weak observable phase distinction, not a
mathematical impossibility theorem about all original images.

The corrected controller removes most of the incidental motion, making an already-
open startup command even less likely to produce a visible change. For a stateless
policy, this creates a problematic supervision pattern: initial task scenes are
labelled hold/open, but the script proceeds to a different command after that hold
without an explicit phase input. Successful complete action replay establishes
physical executability; it does not establish that each action is a well-defined
function of the model's available observation.

This is a concrete startup-supervision defect and the strongest source-level
explanation for the observed initial hold loop. Attribution of *all* policy
failures to it would require a corrected-data/model counterfactual, which has not
been trained or tested.

## Other findings and limits

The full zero/open token sequence appears in 600 of 22,311 training examples
(2.69%): 200 opening, 200 dropping and 200 settling steps. It does not dominate
the global dataset. Individual near-zero X and Y bins are common (33.1% and
47.2%), partly because many scripted phases are vertical or rotational. That
frequency alone does not explain a context-specific stall.

The run processed 40,000 training examples (5,000 updates × 8), equivalent to
1.79 training-population passes. That is an exposure count, not proof that more
updates would fix the problem. The 512-example shuffle buffer gives limited
mixing, but `.repeat()` still revisits the finite dataset; no evidence here shows
that opening/approach examples were filtered out or shifted to the wrong images.
Prior full bundle checks verified image/action alignment and exclusion of terminal
dummy actions. The deployed processor matched training prompts/pixels on all 52
diagnostic samples, and normalization matched the bowl statistics.

The 52-example checkpoint-2,000 diagnostic also found poor approach and rotation
predictions on familiar images, and larger errors without teacher-forced earlier
tokens. Those errors remain relevant even after repairing startup supervision.
They prevent claiming that the startup defect is the sole cause or that removing
it guarantees successful placement. The collection/current controller mismatch
is independently demonstrated by train_001 action replay.

## Repair to validate before more training

Use one controller consistently for collection and inference. The expert's first
command should be a state-dependent, useful command, rather than an unconditional
opening hold. For example, omit opening when already open, or safely combine
opening with the first necessary clearance/reorientation movement. Preserve opening
when it changes the gripper state and matters to the task; do not indiscriminately
delete zero-translation grasp/drop examples.

Generate a small new pilot under the chosen controller, verify action replay and
inspect initial image/label pairs before regenerating the collection. Require
meaningful observation change for redundant setup transitions, and check initial-
state generated actions as part of evaluation rather than relying solely on the
aggregate teacher-forced loss. This is a recommended corrective experiment; the
existing dataset and trained adapters have not been edited by this audit.
