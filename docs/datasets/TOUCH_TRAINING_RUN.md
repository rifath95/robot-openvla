# Touch-cube short training run

Profile: `training/touch_cube_50_500.json`.

- Dataset: `training_touch_v1`, camera `touch_camera`.
- 34 training demonstrations / 604 actions; 8 validation / 148 actions;
  8 test / 152 actions. Test remains held out.
- Start from the unchanged pretrained base with freshly initialized LoRA adapters.
- User-selected microbatch 8, accumulation 1, effective batch 8.
- Fixed profile batch policy: automatic benchmark selection does not substitute
  another size. This is an explicit choice, not a new capacity benchmark.
- 500 updates: 4,000 sampled training examples, about 6.62 training-set passes'
  worth; samples are not guaranteed equal exposure.
- Learning rate 0.0001, LoRA rank 32, no quantization or image augmentation.
- Checkpoint and full validation at 0, 10, 20, ..., 100, 200, 300, 400, 500.
- Update 0 is baseline validation and initialized adapter saving, before learning.
- Inner limit 1 hour; outer detached supervisor limit 90 minutes.
- Requires an A40 and at least 9 GiB free network-volume space for fifteen
  adapter checkpoints (roughly 7 GB of weights) and reports. Actual memory use
  remains dependent on the A40's exposed VRAM; the fixed batch is not a claim
  that it passes the benchmark headroom screen on every A40 deployment.

HF source: `Maxalderone/robot-openvla-panda-touch`, verified completed revision
`4db62eb5225e9d02df450565f1987eff65dd71c4`. All 11 files are present; three large
file SHA256 hashes and seven metadata file checksums match the local bundle,
and the uploaded checksum manifest matches. Upload took approximately 63 seconds.
Portable local bundle: `datasets/panda_touch_cube/training_touch_v1`, about 115 MB.

On the cloud pod with the training template, HF read secret, RunPod stop secret
and network volume attached, update/clone the repository and download:

```bash
cd /root/robot-openvla
git pull --ff-only
/opt/openvla/bin/python -m cloud.download_training_dataset --bundle training_touch_v1
df -h /workspace
```

If reusing an inference pod, stop its model server before training so it does
not occupy the GPU alongside the trainer. Leave the persistent base, datasets
and adapters intact.

After the bundle passes checksum verification and storage is sufficient, launch
from the Mac repository terminal:

```bash
caffeinate -i .venv/bin/python -m cloud.training_cloud_session \
  --ssh 'CURRENT_POD_SSH_COMMAND' \
  --config training/touch_cube_50_500.json
```

The startup output must report microbatch 8, accumulation 1 and effective batch 8.
The trainer records timings, losses and GPU memory usage. The cloud supervisor
saves persistent checkpoints/reports, provides a bounded download window and
requests a pod stop. A connected Mac automatically downloads reports; a sleeping
or disconnected Mac can fetch them from the volume later.

No cloud training has been started by preparing this profile. The touch-task
simulator inference configuration will be prepared separately for checkpoint
comparison after training. Begin with greedy decoding to assess learned behavior;
sampling remains an optional, separately recorded inference experiment.
