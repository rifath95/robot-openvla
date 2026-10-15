# Documentation index

**Phases 1–3 are complete; Phase 4 has trained lift/bowl pilots, expanded datasets, and a prepared overnight training/evaluation chain.** Start with the
[Phase 3 completion summary](reviews/PHASE3_COMPLETION.md) for achieved results
and limitations, and [repository layout](REPOSITORY_LAYOUT.md) for code locations.

For result folders and replay locations, see the [local outputs layout](OUTPUT_LAYOUT.md).

## Simulation and controls

- [Manual cube-to-bowl scene](simulation/BOWL_SCENE.md)
- [Local control and camera checks](simulation/CONTROL_CALIBRATION.md)
- [Local investigation and bounded recovery](simulation/CONTROL_FINDINGS.md)
- [Offline run playback](simulation/REPLAY.md)
- [Click ▶ Replay in a local sweep REPORT.md](simulation/REPLAY_BUTTONS.md)

## Cloud sessions and inference

- [Checkpoint 6000 → 12000 continuation, 600 matched trials and verified HF delivery](cloud/COLOR_BOWL_CONTINUATION_CHAIN.md)
- [Interactive bowl playground: checkpoint buttons, scene colors and typed instructions](cloud/INTERACTIVE_BOWL_PLAYGROUND.md)
- [Prepared 800-trial colored bowl comparison, HF delivery and automatic stop](cloud/COLOR_BOWL_600_SWEEP.md)

- [Prepared single-pod bowl training, 1,000-trial sweep, verified HF backup and global recovery](cloud/BOWL_TRAIN_SWEEP_CHAIN.md)

- [Default two-stage workflow: simulator pod, then cheaper HF transfer pod](cloud/TWO_STAGE_SWEEP_WORKFLOW.md)
- [Archive completed sweep results to Hugging Face, then stop the pod](cloud/HF_SWEEP_ARCHIVE.md)
- [Prepared multi-cube color selection sweep: 4, 5 and 6 cubes](cloud/MULTI_COLOR_MODEL_SWEEP.md)
- [Detached cloud simulator sweeps, resume and disconnect protection](cloud/CLOUD_SIMULATOR_SWEEPS.md)
- [Four-color model sweep: pretrained base and every checkpoint](cloud/FOUR_COLOR_MODEL_SWEEP.md)
- [One-command cloud sessions](cloud/CLOUD_QUICKSTART.md)
- [Cloud GPU inference setup](cloud/CLOUD_SETUP.md)
- [Camera-to-server connection test](cloud/CONNECTION_TEST.md)
- [One cloud OpenVLA prediction, displayed on the Mac](cloud/REMOTE_INFERENCE.md)
- [Repeated cloud control with a live local simulator](cloud/REMOTE_LOOP.md)
- [Separate automatic touch-checkpoint evaluation](cloud/SCRIPTED_TOUCH_EVALUATION.md)
- [Scripted paired red/blue checkpoint evaluation](cloud/SCRIPTED_COLOR_TOUCH_EVALUATION.md)
- [Unseen-color transfer evaluation and chance comparison](cloud/COLOR_TRANSFER_EVALUATION.md)
- [One cloud-predicted robot movement](cloud/REMOTE_SINGLE_ACTION.md)
- [RunPod connection guide: Mac camera → cloud OpenVLA → Mac action](cloud/RUNPOD_CONNECTION_GUIDE.md)

## Demonstrations and datasets

- [600 mixed one/two-cube color-conditioned bowl-placement demonstrations](reviews/COLOR_PLACE_BOWL_600.md)

- [300 all-training bowl-placement demonstrations](reviews/PLACE_BOWL_300.md)

- [First ten single-cube grasp/lift demonstrations](reviews/LIFT_CUBE_FIRST_TEN.md)
- [Expanded 50-demonstration lift dataset and recommended pilot](reviews/LIFT_CUBE_50.md)
- [First ten paired red/blue touch demonstrations](reviews/COLOR_TOUCH_FIRST_TEN.md)
- [Expanded 50-demonstration red/blue touch dataset](reviews/COLOR_TOUCH_50.md)
- [300-demonstration four-color touch dataset](reviews/FOUR_COLOR_TOUCH_300.md)
- [Broad starting-pose demonstrations](datasets/BROAD_PANDA_DATASET.md)
- [Panda convention and expert demonstrations](datasets/PANDA_DEMONSTRATIONS.md)
- [Expanded dataset: training_v2](datasets/TRAINING_V2.md)

## Fine-tuning

- [Prepared 600-demo mixed colored bowl training and checkpoint backup](training/COLOR_PLACE_BOWL_600_TRAINING.md)

- [Red/blue touch training profile and launch](training/COLOR_TOUCH_TRAINING.md)
- [Four-color touch training profile and launch](training/FOUR_COLOR_TOUCH_TRAINING.md)
- [Archived touch checkpoint 300 and restore instructions](training/TOUCH_CHECKPOINT_ARCHIVE.md)
- [A40 cube-to-bowl training profile and launch](training/BOWL_TRAINING.md)
- [Blackwell training and benchmark image](training/BLACKWELL_TRAINING_SETUP.md)
- [Phase 3: historical pick-and-place training guide](training/PHASE3_TRAINING.md)
- [First cloud LoRA test — October 10, 2026](training/SMOKE_REVIEW.md)
- [Measure training microbatch size on the A40](training/TRAINING_BATCHES.md)
- [Detached training with automatic pod stop](training/TRAINING_OVERNIGHT.md)
- [Cube-to-bowl dataset and trajectory replay](datasets/PANDA_BOWL_DATASET.md)

## GPU benchmarking

- [Global-volume inputs for GPU benchmarks](benchmarks/GLOBAL_BENCHMARK_SETUP.md)
- [GPU batch capacity and training throughput](benchmarks/GPU_TRAINING_COMPARISON.md)
- [GPU inference latency, simulator timings, and 100-step cost](benchmarks/GPU_INFERENCE_COMPARISON.md)

## Audits and findings

- [Multi-cube color retention and checkpoint comparison](reviews/MULTI_CUBE_MODEL_SWEEP_REVIEW.md)
- [October 14 volume cleanup and retained checkpoints/results](reviews/VOLUME_CLEANUP_20261014.md)
- [Phase 3 completion: experiments, achievements and remaining limits](reviews/PHASE3_COMPLETION.md)
- [Four-color sweep: base, all checkpoints, and stronger unseen-color generalization](reviews/FOUR_COLOR_MODEL_SWEEP_REVIEW.md)
- [Red/blue wrong-cube and alignment failure diagnosis](reviews/COLOR_TOUCH_FAILURE_DIAGNOSIS.md)
- [Red/blue paired simulator sweep review](reviews/COLOR_TOUCH_SCRIPTED_REVIEW.md)
- [Red/blue touch 500-update training review](reviews/COLOR_TOUCH_TRAINING_REVIEW.md)
- [Four-color touch 1,000-update training review](reviews/FOUR_COLOR_TOUCH_TRAINING_REVIEW.md)
- [Completed unseen-color sweep of red/blue-trained checkpoints](reviews/UNSEEN_COLOR_TRANSFER_REVIEW.md)
- [Touch-cube checkpoint trials and training curves](reviews/TOUCH_CHECKPOINT_TRIALS.md)
- [80 scripted touch trials at unseen cube positions](reviews/TOUCH_SCRIPTED_EVALUATION.md)
- [Check expert images through the cloud inference pipeline](reviews/EXPERT_EVALUATION.md)
- [Compare teacher forcing and normal generation](reviews/MATCHED_EVALUATION.md)
- [Phase 3: approach learning after 1,000 updates](reviews/PHASE3_LEARNING_REVIEW.md)
- [Phase 3: training and inference consistency](reviews/PIPELINE_AUDIT.md)
- [Local audit after Panda v2 fine-tuning](reviews/V2_LOCAL_AUDIT.md)

- [Phase 4 lift dataset expanded to 300](reviews/LIFT_CUBE_300.md): preserved pilot demonstrations, split counts, verification and training setup.

- [Phase 4 bowl placement: first ten](reviews/PLACE_BOWL_FIRST_TEN.md): blue bowl interior, fixed lift start, physical placement and replay checks.

- [Phase 4 bowl placement expanded to 50](reviews/PLACE_BOWL_50.md): verified examples, upload bundle and proposed pilot configuration.
- [300-demo red-only bowl model: complete 1,000-trial review](reviews/PLACE_BOWL_300_SWEEP.md): checkpoint comparison, matched color requests and remaining placement failures.
