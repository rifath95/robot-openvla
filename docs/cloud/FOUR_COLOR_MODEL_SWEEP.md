# Four-color model sweep with a pretrained baseline

Profile: `inference/sweep_profiles/four_color_all_models.json`. Source training session: `panda-train-20261013-110217-1084`.

The selected sweep compares six models: pretrained base, then four-color checkpoints 50, 100, 300, 500, and 1,000. The base entry has internal index 0, but **is not an adapter checkpoint 0**. It loads the unchanged pretrained neural weights without any LoRA adapter, and uses the same Panda action-normalization statistics copied from checkpoint 50. This robot-specific calibration keeps action decoding comparable; it does not fine-tune the neural weights.

| Pair | Role |
| --- | --- |
| Red / blue | Seen training colors |
| Green / yellow | Seen training colors |
| Black / purple | Unseen target colors |
| Orange / white | Unseen target colors |

Each pair uses 10 unseen position seeds (1000–1009), two target instructions per scene, and all six models: **480 total trials, 80 per model**. Both target instructions use the same initial state/image, and visible colors swap physical slots across scenes. Greedy generation, 50-action maximum, independent scene initialization per trial. Evaluation positions are checked against all 300 collected demonstrations before the local plan is created.

Seen and unseen aggregate results are reported separately. Do not interpret the all-pairs aggregate as an unseen-color generalization score. Correct target contact, wrong contact, no contact, paired-instruction consistency, and clean completion remain distinct. Comparisons with the calibrated base are evidence about behavior before versus after fine-tuning, rather than a guarantee that any success originates specifically from pretraining.

## Runtime planning

A permission-authorized read-only check of the active older red/blue cloud sweep on October 13 measured 26 finished cloud trials, averaging **12.67 seconds between trial completions**, including result publication. Individual trial execution averaged about 10.75 seconds in the preceding 22-trial sample. Its first adapter model load/merge took 17.58 seconds. These measurements involve the older model and are an estimate source, not results for this new sweep.

Expected A40 runtime for 480 trials: **about 2–3 hours**, allowing for a base model and new checkpoints taking longer rollouts. At the user's $0.59/hour rate that corresponds to roughly $1.18–$1.77 for that duration, excluding deployment or unrelated running time. The selected profile uses a conservative **3-hour process estimate**, **4-hour inner limit**, and **5-hour outer limit**, following the requested one-hour buffers. Completion stops earlier. The rejected 960-trial option is not scheduled.

## Launch

Use a separate A40 training-template pod, the same network volume and stop secret, with `NVIDIA_DRIVER_CAPABILITIES=all`. Do not use the pod running the other sweep or a training job.

A local 480-trial plan is already prepared at:

`outputs/sweeps/scripted_four_color_models_20261013_132943_e69718c3`

After receiving the pod's current SSH address, launch locally:

```bash
.venv/bin/python -m cloud.cloud_sweep_session \
  --ssh 'ssh root@HOST -p PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --resume outputs/sweeps/scripted_four_color_models_20261013_132943_e69718c3
```

To create a fresh sweep in the future from the single editable profile:

```bash
.venv/bin/python -m cloud.cloud_sweep_session \
  --ssh 'ssh root@HOST -p PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --profile inference/sweep_profiles/four_color_all_models.json
```

A new profile launch creates a new results folder; use its printed path with `--resume` to continue that same sweep, or `--resume ... --fetch-only` to download results later. Do not create a fresh profile launch when you intend to resume an existing sweep.

After `Detached cloud sweep started`, the Mac may disconnect. MuJoCo and inference run entirely on the pod. Completed trial images/actions/states, server logs, reports, startup metadata and status are saved on the network volume before the supervisor requests stopping. A connected watcher fetches them automatically. Code is frozen once, weights are staged once, and adapters merge in VRAM with a separate OS-assigned loopback port for each server. See [cloud sweep lifecycle and fetching](CLOUD_SIMULATOR_SWEEPS.md).

The current older red/blue sweep continues unchanged; its frozen code and independent session are unaffected by preparing or pushing this profile.
