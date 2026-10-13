# Documentation index

Start with [repository layout](REPOSITORY_LAYOUT.md) for code locations.

## Simulation and controls

- [Manual cube-to-bowl scene](simulation/BOWL_SCENE.md)
- [Local control and camera checks](simulation/CONTROL_CALIBRATION.md)
- [Local investigation and bounded recovery](simulation/CONTROL_FINDINGS.md)
- [Offline run playback](simulation/REPLAY.md)

## Cloud sessions and inference

- [One-command cloud sessions](cloud/CLOUD_QUICKSTART.md)
- [Cloud GPU inference setup](cloud/CLOUD_SETUP.md)
- [Camera-to-server connection test](cloud/CONNECTION_TEST.md)
- [One cloud OpenVLA prediction, displayed on the Mac](cloud/REMOTE_INFERENCE.md)
- [Repeated cloud control with a live local simulator](cloud/REMOTE_LOOP.md)
- [Separate automatic touch-checkpoint evaluation](cloud/SCRIPTED_TOUCH_EVALUATION.md)
- [Scripted paired red/blue checkpoint evaluation](cloud/SCRIPTED_COLOR_TOUCH_EVALUATION.md)
- [One cloud-predicted robot movement](cloud/REMOTE_SINGLE_ACTION.md)
- [RunPod connection guide: Mac camera → cloud OpenVLA → Mac action](cloud/RUNPOD_CONNECTION_GUIDE.md)

## Demonstrations and datasets

- [First ten paired red/blue touch demonstrations](reviews/COLOR_TOUCH_FIRST_TEN.md)
- [Expanded 50-demonstration red/blue touch dataset](reviews/COLOR_TOUCH_50.md)
- [Broad starting-pose demonstrations](datasets/BROAD_PANDA_DATASET.md)
- [Panda convention and expert demonstrations](datasets/PANDA_DEMONSTRATIONS.md)
- [Expanded dataset: training_v2](datasets/TRAINING_V2.md)

## Fine-tuning

- [Red/blue touch training profile and launch](training/COLOR_TOUCH_TRAINING.md)
- [Archived touch checkpoint 300 and restore instructions](training/TOUCH_CHECKPOINT_ARCHIVE.md)
- [A40 cube-to-bowl training profile and launch](training/BOWL_TRAINING.md)
- [Blackwell training and benchmark image](training/BLACKWELL_TRAINING_SETUP.md)
- [Phase 3: prepare, fine-tune, and evaluate OpenVLA](training/PHASE3_TRAINING.md)
- [First cloud LoRA test — October 10, 2026](training/SMOKE_REVIEW.md)
- [Measure training microbatch size on the A40](training/TRAINING_BATCHES.md)
- [Detached training with automatic pod stop](training/TRAINING_OVERNIGHT.md)
- [Cube-to-bowl dataset and trajectory replay](datasets/PANDA_BOWL_DATASET.md)

## GPU benchmarking

- [Global-volume inputs for GPU benchmarks](benchmarks/GLOBAL_BENCHMARK_SETUP.md)
- [GPU batch capacity and training throughput](benchmarks/GPU_TRAINING_COMPARISON.md)
- [GPU inference latency, simulator timings, and 100-step cost](benchmarks/GPU_INFERENCE_COMPARISON.md)

## Audits and findings

- [Red/blue wrong-cube and alignment failure diagnosis](reviews/COLOR_TOUCH_FAILURE_DIAGNOSIS.md)
- [Red/blue paired simulator sweep review](reviews/COLOR_TOUCH_SCRIPTED_REVIEW.md)
- [Red/blue touch 500-update training review](reviews/COLOR_TOUCH_TRAINING_REVIEW.md)
- [Touch-cube checkpoint trials and training curves](reviews/TOUCH_CHECKPOINT_TRIALS.md)
- [80 scripted touch trials at unseen cube positions](reviews/TOUCH_SCRIPTED_EVALUATION.md)
- [Check expert images through the cloud inference pipeline](reviews/EXPERT_EVALUATION.md)
- [Compare teacher forcing and normal generation](reviews/MATCHED_EVALUATION.md)
- [Phase 3: approach learning after 1,000 updates](reviews/PHASE3_LEARNING_REVIEW.md)
- [Phase 3: training and inference consistency](reviews/PIPELINE_AUDIT.md)
- [Local audit after Panda v2 fine-tuning](reviews/V2_LOCAL_AUDIT.md)
