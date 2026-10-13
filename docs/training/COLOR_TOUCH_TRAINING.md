# Red/blue touch fine-tuning

Profile: `training/color_touch_50_500.json`.

Start from pretrained `/workspace/openvla-7b`, using the verified bundle `/workspace/panda-training/training_color_touch_v1`. No old adapter is supplied. The 50 demonstrations contain 34 training trajectories (609 actions), 8 validation trajectories (136 actions), and 8 test trajectories (142 actions). Each split balances red and blue targets; paired scenes stay within one split.

- 500 updates; microbatch 8; accumulation 1; effective batch 8.
- Learning rate 0.0001; LoRA rank 32; fixed configured batch policy on NVIDIA A40.
- Save checkpoints at **50, 100, 300 and 500**. No checkpoint 0.
- Full validation at **0, 50, 100, 300 and 500**, covering all 136 validation actions each time. The initial measurement starts the validation curve; it does not save a checkpoint.
- 4,000 training example exposures, about 6.57 passes worth over the 609 training actions. Test examples are not used for training or validation.
- Inner runtime cap: 3,600 seconds; outer cap: 5,400 seconds including setup.
- Require at least 4 GiB free on the volume for adapter checkpoints and reports.

Prepare this profile before launching a pod. Once a training-template A40 pod is running with the existing volume and RunPod stop secret, run in the **Mac repository terminal**, substituting its current SSH address:

```bash
caffeinate -i .venv/bin/python -m cloud.training_cloud_session \
  --ssh 'ssh root@HOST -p PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --config training/color_touch_50_500.json
```

The launcher validates the selected bundle and starts detached cloud training. The Mac may disconnect once the detached supervisor is confirmed started. If connected at completion, it downloads reports automatically. Checkpoints and reports stay on the volume before the supervisor requests a pod stop. Confirm stopped status in RunPod.

For a future volume that needs the bundle again, the download helper now supports the pinned HF revision:

```bash
/opt/openvla/bin/python -m cloud.download_training_dataset --bundle training_color_touch_v1
```

Repository: `Maxalderone/robot-openvla-panda-color-touch`; revision: `48d4da5e692b3f13c774e8cf547a8668115fae09`. This dataset has already been downloaded and verified on the user's current volume.
