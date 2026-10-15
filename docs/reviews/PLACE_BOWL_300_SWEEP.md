# Single-red-cube bowl model: complete 1,000-trial review

Reviewed October 15, 2026. This is the **300-demo red-only model**, trained for
3,000 updates, not the newer 600-demo four-color model trained for 6,000 updates.

Source: `outputs/sweeps/scripted_bowl_chain_20261014_174530_4acde087/`.
Training: `panda-train-chain-20261014-174530-4acde087`.
All 300 demonstrations were training data, each with one red cube and a blue bowl.
The updated controller and corrected scene initialization were used for this sweep.

All 1,000 trial identities pass the frozen sweep-plan validator. All restored
trial summaries exist and their final outcomes match `results.json`. No trial
reported an infrastructure failure. Controller failures remain in the denominator.
Computed aggregates are in [the review data](assets/place_bowl_300_sweep_review.json).

## Best observed checkpoint: 3,000

| Checkpoint | Trials | Requested cube lifted ≥8 cm | Correct placements | Placement rate | Controller failures |
|---|---:|---:|---:|---:|---:|
| Pretrained base | 200 | 0 | 0 | 0.0% | 4 |
| 500 | 200 | 34 | 10 | 5.0% | 57 |
| 1,000 | 200 | 32 | 16 | 8.0% | 18 |
| 2,000 | 200 | 54 | 25 | 12.5% | 21 |
| 3,000 | 200 | 49 | 32 | 16.0% | 19 |

Success requires a prior lift ≥8 cm, the cube settled within 4.5 cm of the bowl
center and 4 mm of resting height, and the hand ≥8 cm away. All 32 successes at
checkpoint 3,000 ended cleanly. No model placed a wrong cube in the bowl.

Checkpoint 3,000 has the most completed tasks. Checkpoint 2,000 lifted the target
more often (54 versus 49), but converted fewer of those lifts into placement:
25/54 = 46.3%, versus 32/49 = 65.3% at 3,000. This conditional comparison suggests
better completion after a lift at 3,000; it is not a paired causal measurement.

The difference between 2,000 and 3,000 is modest: on the same 200 task instances,
3,000 gains 22 successes and loses 15; both succeed on 10. An exact two-sided
paired discordance test gives p≈0.324. Treat 3,000 as the best observed checkpoint
on this suite, rather than an established universal winner.

## Where checkpoint 3,000 succeeds

| Scene group | Correct placements | Rate |
|---|---:|---:|
| One red cube | 10/20 | 50.0% |
| One black cube | 4/20 | 20.0% |
| Two cubes: red and black, both targets requested | 9/40 | 22.5% |
| Two cubes: black and purple, both targets requested | 9/40 | 22.5% |
| Four cubes with red present | 0/40 | 0.0% |
| Four cubes with red absent | 0/40 | 0.0% |

The model learned usable pick-and-place behavior and can apply it to an unseen
cube color, including when red is absent. It is far from a dependable bowl policy.
Adding four cubes remains particularly difficult: 3,000 lifts the requested cube
in only 3/80 four-cube trials and completes none. Earlier checkpoints produce a
few four-cube successes (500: 4/80; 2,000: 2/80), so improvement is not uniform
across scene types.

## Retained color grounding

Only **red** was seen in this model's fine-tuning demonstrations. Black, purple,
green and orange were unseen cube colors for this run. Green is seen in the
newer four-color training, but that is a different model.

At checkpoint 3,000, the first cube lifted ≥8 cm was the requested cube in all
**29/29 multi-cube trials that produced such a lift**: 26/26 two-cube and 3/3
four-cube trials. No distractor reached the qualifying lift threshold in its
200 trials. This is conditional precision, not 100% color-selection accuracy:
the large majority of multi-cube trials never produce a qualifying lift, and
these figures do not measure first contact or all failed approaches.

More direct instruction-conditioned evidence comes from paired resets:

- Two-cube group `two_red_absent`, seed **1013**, checkpoint **3000**: black and
  purple requests have byte-identical initial camera images. Asking for black
  completes black placement; asking for purple completes purple placement.
  Representative initial, lifted and final images were visually inspected.
- In the red/black group, seed **1016**, changing the requested target likewise
  produces successful placement of each requested color from identical images.
- Across 40 two-cube layouts, five support correct lifts for **both** requested
  targets, and two support full placement for both. Four of those lift pairs
  contain red/black, and one contains black/purple. Their initial image pairs
  were checked for equality.

This supports retained, usable grounding for unseen color words, particularly
black and purple, together with transfer of the learned movement skill. It does
not establish robust recognition of every tested color or scene. Checkpoint
3,000 places black 10/80 times and purple 6/40 times across their scene groups;
green 0/20 and orange 0/10. Green and orange occur only in the difficult four-cube
groups, so those failures do not isolate color recognition from scene difficulty.
No naive chance p-value is assigned to the selectively successful lift subset.

## Remaining bottlenecks

Of checkpoint 3,000's 200 trials:

- **32:** correct placement.
- **17:** target lifted ≥8 cm but not placed.
- **151:** no cube achieved a qualifying lift. Some include smaller lifts or
  failed grasps; “no lift” in the generated report means below the 8 cm threshold.

Its 19 controller failures overlap these outcome categories: 17 no-qualifying-lift
trials and two lifted-but-unplaced trials. Excluding them would still leave only
32/181 placements (17.7%), so controller rejection does not explain most failures.

None of the 17 lifted-but-unplaced targets ends within the 4.5 cm bowl radius.
These are not merely successes penalized because the hand failed to retreat.
Four remain ≥5 cm above their starting height at the final state; the others
also fail the full placement criteria. A representative red-target lift failure
and a four-cube orange-target failure were visually inspected alongside the
successful black/purple pair.

## Implication for the newer experiment

Use checkpoint 3,000 as the main single-red-cube completion baseline, retaining
2,000 for context. The matched 600-demo colored model sweep is the relevant next
comparison: it explicitly teaches target selection with two cubes, whereas this
model saw only one cube. Compare matching scene groups, target requests and
controller-failure-inclusive success rates. The 6,000→12,000 continuation should
then be compared with that colored model's checkpoint 6,000 on the same scenes.

The existing generated `REPORT.md` contains the complete table and direct replay
links for all 1,000 recordings. Examples to locate there:

- `checkpoint_003000_two_red_absent_2c_seed_1013_black`
- `checkpoint_003000_two_red_absent_2c_seed_1013_purple`
- `checkpoint_003000_two_red_present_2c_seed_1016_red`
- `checkpoint_003000_two_red_present_2c_seed_1016_black`
