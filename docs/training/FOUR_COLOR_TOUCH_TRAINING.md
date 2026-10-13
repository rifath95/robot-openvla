# Four-color touch fine-tuning

Profile: `training/four_color_touch_300_1000.json`. Start from pretrained `/workspace/openvla-7b`; no adapter is supplied. The verified dataset is `/workspace/panda-training/training_four_color_touch_v1`.

| Split | Trajectories | Action examples |
| --- | ---: | ---: |
| Training | 200 | 3,544 |
| Validation | 50 | 879 |
| Test | 50 | 888 |

The dataset contains red, blue, green, and yellow cube targets, with two cubes visible per scene. Both instructions for each scene remain in the same split. Test examples are held out from training and validation.

- 1,000 optimizer updates; microbatch 8; accumulation 1; effective batch 8. The configured batch is fixed on NVIDIA A40 rather than automatically substituted.
- Learning rate 0.0001; LoRA rank 32; dropout 0; no quantization or image augmentation.
- 8,000 training example exposures, approximately 2.26 passes worth of the 3,544 training examples (shuffled sampling, not an exact epoch count).
- Checkpoints at 50, 100, 300, 500, and 1,000.
- Full validation at 0, 50, 100, 300, 500, and 1,000: 879 examples each, or 5,274 validation exposures if completed.
- Inner runtime limit 2 hours; outer supervised runtime limit 3 hours including training setup. These start within the training launch, not at pod deployment or dataset transfer.
- Require at least 4 GiB free on the volume for five adapter checkpoints and reports. Existing datasets/checkpoints are retained.

## Prepare before deployment

Complete the upload to `Maxalderone/robot-openvla-panda-four-color-touch`, then use a training-template A40 pod attached to the existing network volume. Include the Hugging Face read secret (`HF_TOKEN`) and RunPod stop secret (`RUNPOD_API_KEY`). Do not launch training in the pod currently running the simulator sweep, since either workflow can stop its own pod.

## Download on the pod

After SSH connection, run:

```bash
if [ -d /root/robot-openvla/.git ]; then
  git -C /root/robot-openvla pull --ff-only
else
  git clone https://github.com/rifath95/robot-openvla.git /root/robot-openvla
fi
cd /root/robot-openvla
/opt/openvla/bin/python -m cloud.download_training_dataset \
  --bundle training_four_color_touch_v1 \
  --revision COMPLETED_UPLOAD_SHA
```

Use the completed dataset upload's commit SHA. Once the upload is confirmed complete, `--revision main` can also be used: the downloader resolves it to a single SHA before reading files and records the exact SHA in `/workspace/panda-training/training_four_color_touch_v1_hf_source.json`. It verifies all checksums before publishing the dataset. This command does not start training or stop the pod.

## Launch from the Mac

After the download reports `Verified downloaded bundle` or `Existing bundle verified`, run in a separate local Mac repository terminal:

```bash
caffeinate -i .venv/bin/python -m cloud.training_cloud_session \
  --ssh 'ssh root@HOST -p PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --config training/four_color_touch_300_1000.json
```

The launcher installs tmux if needed, checks the API, storage, dataset, and configured GPU, then starts the detached supervisor. The Mac can disconnect after the supervisor has started. Training, checkpoints, metrics, validation, curves, and logs remain on the network volume. If the Mac remains connected, reports are downloaded to `outputs/phase3_training/<session>/` before the stop acknowledgment. If disconnected, the supervisor still requests stopping, and reports can be fetched after resuming the pod. Confirm Stopped in RunPod; API/network failure can prevent stopping.

The exact start/finish times, updates, validation timings, and selected batch are recorded in the run reports. No training run is launched by creating this profile.
