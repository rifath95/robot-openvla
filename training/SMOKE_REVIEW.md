# First cloud LoRA test — October 10, 2026

The 20-update test completed and saved its final adapter at:

```text
/workspace/panda-training/runs/lora_20261010_130927_064468/checkpoint_000020
```

The run is stored on `robot-openvla-network-volume-A40`, mounted at `/workspace`.
The downloaded plot is `outputs/phase3_smoke_loss.png`. The full JSON metrics,
model information and checkpoints remain on the network volume; they have not
been downloaded or independently inspected locally. This review uses the plot
and the final terminal metrics supplied by the user.

| Measurement | Result | Evidence |
|---|---:|---|
| Optimizer updates completed | 20 | Terminal progress and final checkpoint |
| Initial training loss | Approximately 10.6 | Read from the plot |
| Final training loss | 2.881129 | Terminal, update 20 |
| Validation loss at update 10 | Approximately 6.4 | Read from the plot |
| Validation loss at update 20 | 3.793934 | Terminal |
| Validation examples evaluated per check | 64 / 507 | Terminal; incomplete split |
| Final action-token accuracy | 0.464286 | Terminal; training batches, not task success |
| Final normalized action L1 error | 0.154062 | Terminal; training batches |
| Peak PyTorch allocated GPU memory | 18.007344 GiB | Terminal; not total device usage |
| Elapsed at final validation | 60.079335 seconds | Excludes model setup and final save |

Training loss is a mean over four microbatches per optimizer update. Its
fluctuations are expected with small batches and changing examples. The
validation line connects only two measurements, not continuous evaluations.
Both measured losses decrease, supporting a longer controlled experiment.
There is not enough evidence to establish overfitting behavior, full validation
performance, convergence, or model-driven grasp/placement success.

## Next controlled run

Use `training/pick_place_1000.json`:

- 1,000 optimizer updates, microbatch 1, accumulation 4: 4,000 sampled training
  examples, approximately 1.54 passes through the 2,595-example training split.
  The repeating shuffled stream is not organized as explicit epochs.
- Preserve rank 32 and learning rate 0.0001 from the successful test. Keep image
  augmentation and quantization disabled to isolate the effect of longer training.
- Increase the encoded-image shuffle buffer from 128 to 512 for broader mixing.
- Evaluate all 507 validation examples and save adapters at updates 250, 500,
  750 and 1,000. The test split stays unused for training/checkpoint selection.
- Start a fresh adapter from the original pretrained model. This does not resume
  the 20-update adapter or overwrite its directory.
- Retain a one-hour training guard, checked between updates. Loading precedes
  this guard; validation/saving can extend it, and it does not stop pod billing.

The smoke run suggests roughly 1.6–1.8 seconds per optimizer update and about
13 seconds for 64 validation examples. Extrapolating that to 1,000 updates and
four complete validation passes suggests approximately 35–45 minutes after
model/dataset setup, with storage, checkpoint writes and GPU/CPU differences
adding uncertainty. This is a planning estimate, not a measured longer-run time.

Afterward, inspect the complete validation curve, choose a checkpoint using
validation performance, and evaluate actual robot behavior on held-out scenarios.
If validation worsens while training improves, prefer an earlier checkpoint and
reassess the schedule. Do not select a checkpoint using the test split.
