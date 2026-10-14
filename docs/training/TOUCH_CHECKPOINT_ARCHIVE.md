# Archived single-cube touch checkpoint 300

Archived on October 13, 2026 to a **private model repository**:

[Maxalderone/robot-openvla-panda-touch-checkpoint-300](https://huggingface.co/Maxalderone/robot-openvla-panda-touch-checkpoint-300)

Pinned archive revision: `9d9e2b22098ab794f2a6161a4ad7d4b223a110ac`.

The archive contains the complete update-300 LoRA checkpoint, processor/tokenizer files, action-normalization statistics, dataset provenance, training reports, plots, logs, TensorBoard events and a SHA-256 manifest. All 34 files were downloaded from the committed HF revision and verified before cleanup. The adapter SHA-256 is `c4cf739322644f7d06708d2faff3c2df3b2b79ffe4cee745c372ba4d0542c073`.

## Which training run and dataset?

- Training session: `panda-train-20261013-034928-1786`.
- Archived adapter: update **300**. The included reports describe the full **500-update** run.
- Instruction: `touch the red cube.`
- Dataset: [Maxalderone/robot-openvla-panda-touch](https://huggingface.co/datasets/Maxalderone/robot-openvla-panda-touch), revision `4db62eb5225e9d02df450565f1987eff65dd71c4`.
- Base model: `openvla/openvla-7b`. Base export provenance and configuration are included; the base weights are not duplicated.
- This is the **single-cube touch model**, not a model trained on the newer red/blue dataset.

`provenance.json` distinguishes checkpoint timing from total run time. The full run's metrics and plots are under `reports/`, and the dataset's bundle checksums and verification records are under `dataset_metadata/`.

## Download later

From the Mac repository terminal, using your existing authenticated HF account:

```bash
.venv-training/bin/hf download \
  Maxalderone/robot-openvla-panda-touch-checkpoint-300 \
  --revision 9d9e2b22098ab794f2a6161a4ad7d4b223a110ac \
  --local-dir outputs/restored_checkpoints/touch_checkpoint_000300
```

On a new cloud pod, download this same model repository and pinned revision into a new local/volume directory using `huggingface_hub.snapshot_download` with your HF read credential. Validate the downloaded `SHA256SUMS` and checkpoint completeness before inference. Keep the entire directory, not only the safetensors file.

The project inference workflow can merge it with the matching OpenVLA base weights using `training.merge_adapter`. The adapter can initialize further fine-tuning, but no optimizer state is included, so it is not an exact optimizer-state resume.

## October 14 follow-up

The network-volume copy of checkpoint 300's adapter weight was removed during
[the follow-up cleanup](../reviews/VOLUME_CLEANUP_20261014.md). Its metadata and
reports remain, and the verified HF archive above is the restore source.
The newer four-color checkpoints are retained separately.

## Approved volume cleanup (October 13 snapshot)

After upload verification, removed the other 14 checkpoints from this session and `/workspace/panda-training/training_touch_v1`. Retained:

- `/workspace/openvla-7b`.
- `/workspace/panda-training/sessions/panda-train-20261013-034928-1786/runs/training/checkpoint_000300`.
- All reports, logs, plots, metrics, benchmark results and diagnostics.
- Small configuration/metadata copies of the deleted checkpoints.

Logical space removed: **6.455 GiB**. The volume subsequently reported **14 GiB free**.

Archive receipt: `/workspace/panda-training/reports/touch_checkpoint_300_hf_archive.json`.
Cleanup receipt: `/workspace/panda-training/reports/touch_cleanup_after_hf_archive.json`.
Both receipts are also saved locally under `outputs/archives/hf-models/hf-archives/`.

The newer red/blue dataset was verified separately at `Maxalderone/robot-openvla-panda-color-touch`, revision `48d4da5e692b3f13c774e8cf547a8668115fae09`. It has not yet been downloaded to this volume as part of this archive/cleanup operation.
