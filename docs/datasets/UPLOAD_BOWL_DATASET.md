# Upload the bowl training dataset

Run these commands from the repository root on the Mac. No RunPod is needed.
The portable bundle contains all 300 trajectories: 200 training, 50 validation,
and 50 test. Images, text instructions and seven-dimensional actions are
embedded in TFRecords. The original replay folders remain local.

## Prepare and check the bundle

```bash
caffeinate -i .venv-training/bin/python -m data_tools.prepare_training_dataset \
  --manifest datasets/panda_pick_bowl/manifest.json \
  --output-dir datasets/panda_pick_bowl/training_bowl_v1

caffeinate -i .venv-training/bin/python -m data_tools.check_training_inputs \
  --dataset datasets/panda_pick_bowl/training_bowl_v1 \
  --source-manifest datasets/panda_pick_bowl/manifest.json \
  --processor .cache-openvla/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f
```

Run the second command only after the first succeeds. Continue to uploading
only after both succeed. The processor check loads no model weights.
The internal `panda_pick_place` schema name is retained for compatibility;
this bundle contains the new cube-to-bowl task and its bowl camera contract.
Packaging refuses an existing destination. If packaging already completed,
run the input check rather than rebuilding it.

## Authenticate and upload

```bash
.venv-training/bin/hf auth whoami
```

Use the existing Hugging Face login with write access. If needed, run
`.venv-training/bin/hf auth login` and enter a write token at its prompt.
A read-only token cannot upload. Do not share tokens in chat.

```bash
caffeinate -i .venv-training/bin/hf upload-large-folder \
  Maxalderone/robot-openvla-panda-bowl \
  datasets/panda_pick_bowl/training_bowl_v1 \
  --repo-type dataset --private --num-workers 4
```

This creates a private repository if absent and uploads the bundle contents
at the repository root. The installed CLI displays progress and periodic
status reports. Keep this terminal open and the Mac connected to the internet;
other terminals and applications can be used in parallel. `caffeinate -i`
prevents automatic idle sleep while the command runs. Closing the lid or
manually sleeping the Mac interrupts uploading. Rerun the same upload command
to resume; its progress cache is stored inside the bundle's `.cache` folder.

The bundle is prepared and uploaded by these user-run commands. This guide
does not start either operation automatically.
