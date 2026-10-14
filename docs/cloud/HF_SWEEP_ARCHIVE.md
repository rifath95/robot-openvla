# Archive a simulator sweep to Hugging Face, then stop the pod

This avoids paying for GPU runtime while a slow Mac download completes. Upload directly from the network volume to a private HF dataset repository, verify the committed bytes on the pod, then request pod stop. Download from HF later without a running GPU.

The helper is `cloud.archive_sweep`. It supports completed multi-cube sweeps, checks their recorded completion and complete trial inventory, packages each model/cube-count/palette group separately (36 archive parts for the 1,080-trial profile), and includes the original reports, plans, logs and model provenance. No base weights are duplicated. Original volume files are retained; this command performs no cleanup.

## Run in the pod terminal

```bash
cd /root/robot-openvla
git pull --ff-only
/opt/openvla/bin/python -u -m cloud.archive_sweep \
  --sweep scripted_multi_color_models_20261013_164548_eeda379b \
  --repo Maxalderone/robot-openvla-simulator-results
```

Paste an HF **write** token at the hidden prompt. It is passed to detached processes in memory, not written into a receipt, log, command argument or shell history. The helper uses the pod's existing RunPod secret and checks API/network-volume access before launching. It refuses an active training pod and refuses simultaneous uploads of the same sweep.

The private dataset repository is created automatically. An existing public repository is rejected. The uploaded files are under a folder named after the sweep, so future sweeps can use the same repository.

Wait for **`Detached HF archive upload started. Mac may disconnect.`** After that, SSH/Mac disconnects do not cancel the job. In the cloud/web terminal:

```bash
tail -F /workspace/panda-training/reports/hf-archives/scripted_multi_color_models_20261013_164548_eeda379b/upload.log
```

Each archive part prints its group and trial count while packing; the HF client provides transfer progress; verification prints each remote file. Ctrl+C stops watching, not uploading.

## Verification, deadlines and receipts

The helper packs onto container disk, avoiding another archive copy on the network volume. It requires approximately 16 GiB free on container disk for this profile, allowing packing and verification caches. It avoids an extra whole-volume size traversal before packing. Preparation time includes NFS file reads and compression, not only internet transfer.

After upload it pins the returned HF commit, downloads every archive part, report and manifest onto container disk, and compares their byte counts and SHA-256 checksums. Only after all match does it write `receipt.json` with `verified: true`, exact revision, uploaded path, file count and byte count.

Receipts and logs persist under `/workspace/panda-training/reports/hf-archives/SWEEP_NAME/`:

- `status.json`: packing, uploading/verifying, verified or failed.
- `receipt.json`: successfully verified HF revision and files.
- `supervisor_status.json`: completion outcome and runtime budget.
- `pod_stop_status.json`: attempted/accepted stop request.
- `upload.log`: packing, transfer and verification progress.

Default estimate is one hour, with one hour added for the inner deadline (two hours) and another for the outer budget (three hours). The supervisor kills a worker reaching its deadline and still requests pod stop. Failure also requests stop and retains original data for retry. API failure can prevent stopping; check the portal. Read-only preflight failure occurs before a detached job is armed and does not stop the pod automatically.

## Download later on the Mac

Using your authenticated local HF installation:

```bash
caffeinate -i .venv-training/bin/hf download \
  Maxalderone/robot-openvla-simulator-results \
  --repo-type dataset \
  --include 'scripted_multi_color_models_20261013_164548_eeda379b/*' \
  --local-dir outputs/hf-sweep-archives
```

Use `--revision VERIFIED_COMMIT` from the receipt to pin the exact verified archive. No pod needs to run during this download. For the small reports only, change the include pattern to `SWEEP_NAME/reports/*`; for a particular model/count/palette, select the corresponding `archives/*.tar.gz` file. The manifest lists checksums and trial counts for every part. Extract chosen parts into one sweep folder and copy the `reports/` contents into its root to restore the ordinary results/replay layout.

This path uses the HF [upload](https://huggingface.co/docs/huggingface_hub/guides/upload) and [download](https://huggingface.co/docs/huggingface_hub/guides/download) APIs. Code and local mocked upload/verification tests are prepared before cloud execution; the actual remote upload is performed by the user's command.
