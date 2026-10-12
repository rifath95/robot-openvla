# A40 cube-to-bowl LoRA training

Profile: `training/pick_bowl_5000.json`. This starts from the untouched base
OpenVLA weights with fresh LoRA adapters unless an adapter is explicitly supplied.
It uses `training_bowl_v1`: 200 train trajectories / 22,311 examples, 50 validation
trajectories / 5,454 examples, and 50 test trajectories / 5,517 examples.
The profile checks the bowl camera/action contract, split counts and prior source
input verification before training. Older datasets are rejected for this profile.
The internal `panda_pick_place` schema and `panda_grasp_v1` action key remain valid.

Settings: 5,000 parameter updates, learning rate 0.0001, LoRA rank 32,
accumulation 1, shuffle buffer 512. A compatible A40 benchmark selects the actual
microbatch; the profile's nominal batch 5 is not used as an unverified fallback.
If GPU/VRAM/input/model/software compatibility cannot be established, training is
refused. The previous A5000's fitting batch does not configure an A40.

Full validation runs before the first update, every 1,000 updates, and at the
end when needed. Checkpoints are saved every 1,000 updates and at a normal early
exit. The test split is excluded from training and validation. Inner training
limit is 4 hours; the detached supervisor's outer limit is 5 hours, including
setup after launch. These are not deployment-to-shutdown billing limits.
The network volume needs at least 3 GiB free for checkpoint storage.

## Start from the Mac

Use the usual training-image A40 pod with the network volume at `/workspace`,
the original model at `/workspace/openvla-7b`, the verified bowl bundle and the
RunPod stop secret. No Docker rebuild is needed. In the Mac repo terminal:

```bash
caffeinate -i .venv/bin/python -m cloud.training_cloud_session \
  --ssh 'ssh root@POD_IP -p POD_PORT -i ~/.ssh/id_ed25519' \
  --config training/pick_bowl_5000.json
```

The profile selects `/workspace/panda-training/training_bowl_v1` automatically.
Wait for the cloud session to start before disconnecting. Training then continues
in tmux independently of the Mac. Checkpoints, plots, metrics and logs remain on
the network volume before the wrapper requests stopping. If the Mac stays
connected, reports download automatically; weights remain on the volume. If
disconnected, resume a pod on the volume and use the launcher's `--fetch-only`
with the saved session name. Confirm stopped status in RunPod.

For a manual cloud launch after checkout and installing tmux:

```bash
bash scripts/train_detached.sh training/pick_bowl_5000.json
```

This also selects the bowl bundle automatically. The legacy smoke and older
training profiles retain their previous dataset defaults. Read
[detached training](TRAINING_OVERNIGHT.md) for timeout and retrieval details.
