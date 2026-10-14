# Default sweep workflow: simulate first, transfer on a cheaper pod

Agreed October 14, 2026. Keep simulation and transfer separate so packing/network work cannot slow the robot trials. The existing HF archive currently downloading to the Mac is unchanged.

## Stage 1: simulator pod

Use the chosen simulation GPU, attach the network volume containing the base/adapter checkpoints, and run the cloud sweep. The launcher now defaults to `--delivery transfer-pod`. It watches progress without starting a large Mac download when finished, acknowledges completion, and lets the cloud supervisor request pod stop. A disconnected Mac also does not prevent stopping.

```bash
caffeinate -i .venv/bin/python -m cloud.cloud_sweep_session \
  --ssh 'ssh root@SIMULATOR_IP -p SIMULATOR_PORT -i ~/.ssh/id_ed25519_maxalderone' \
  --profile inference/sweep_profiles/multi_color_all_models.json
```

Save the printed `scripted_...` folder name. All completed trial artifacts stay on the volume. No packing or HF uploading runs in this stage. Do not launch this example to repeat an already completed sweep unless a new experiment is intended.

## Stage 2: cheaper transfer pod

After confirming stage 1 stopped, deploy a cheaper pod that can attach the **same network volume**. A CPU pod is sufficient if available and compatible; a cheap GPU pod also works. No model loading, CUDA inference or MuJoCo simulation occurs. The existing training template provides `/opt/openvla/bin/python`, HF dependencies, SSH and the RunPod stop secret, so it can be reused without a Docker rebuild. Select by total rate × duration, considering CPU/NFS/network performance, rather than assuming the cheapest hourly option is always cheapest overall.

Prepare the transfer command and HF write credential **before deployment**. In the transfer pod terminal:

```bash
if [ -d /root/robot-openvla/.git ]; then
  git -C /root/robot-openvla pull --ff-only
else
  git clone https://github.com/rifath95/robot-openvla.git /root/robot-openvla
fi
cd /root/robot-openvla
/opt/openvla/bin/python -u -m cloud.archive_sweep \
  --sweep YOUR_COMPLETED_SWEEP_NAME \
  --repo Maxalderone/robot-openvla-simulator-results
```

Paste the HF write token at the hidden prompt. It is not stored in files. Once `Detached HF archive upload started` appears, the Mac can disconnect. Use the printed `upload.log` path with `tail -F` in the web terminal to watch progress.

The helper:

1. Packs one model/count/palette group at a time on container disk using light compression.
2. Saves each completed archive part and its source identity on the volume for reuse after a restart.
3. Uploads that ready part through one background uploader while packing the next. Reports and the final manifest upload after all parts are ready/uploaded.
4. Reuses unchanged packed parts and skips archive files whose remote size/SHA match. Incomplete parts are rebuilt; no original trial data is removed.
5. Downloads the pinned final archive files to container disk and verifies sizes/SHA-256, saves a verified receipt, then requests pod stop.

Failure/timeouts also request stopping while preserving source results and completed volume-cache parts. Existing runtime buffer policy remains: estimate + one hour for the inner limit; another hour for the outer budget. Default transfer estimate is one hour. Adjust `--estimated-seconds` before a transfer if available measurements indicate it is insufficient on the chosen pod. Confirm stopped status in RunPod because management API/network errors can prevent shutdown.

The reusable cache consumes additional network-volume space. For the 1,080-trial profile, allow approximately 6 GiB free on the volume and 16 GiB on container disk before the initial transfer. Cached parts reduce the retry requirement. Remove caches/source copies only through a separately authorized cleanup after the archive has been verified and retained.

## Stage 3: Mac download, no pod

Use the ordinary HF download command with the completed sweep prefix and, preferably, the pinned revision from `receipt.json`. No GPU/CPU pod needs to remain running. Download the small reports first if desired, or all archive parts. Interrupted HF downloads can be retried using the same destination and cache.

See [HF archive instructions](HF_SWEEP_ARCHIVE.md) for exact download patterns, receipts and extraction layout.

## What has been tested

Local tests cover the stop-without-download handoff, packing/upload overlap, cross-container archive reuse, metadata-based remote skip, upload interruption, checksum mismatch and supervisor stop paths. Actual transfer duration and cost on a cheaper pod have not been measured yet; no claim of a guaranteed speedup is made. Packing during simulation remains intentionally disabled.

The old explicit direct-Mac fetch is still a legacy alternative: it streams a temporary archive, has a 30-minute timeout and discards partial data on failure. This workflow uses HF delivery instead; fixing that alternative is separate work.

## Upload two completed sweeps on one transfer pod

Use `--sweeps` instead of launching two independent single-sweep supervisors.
A single detached supervisor holds both upload locks, packs/uploads/verifies
sequentially, and requests stop only after all selected archives verify, or on
failure/timeout. The first sweep's verified receipt survives failure of the
second; source results remain. Each sweep retains its own receipt and cache.
A combined log and receipt are written under a printed `hf-archives/batch_...`
folder. No token is written to these records.

For the completed lift and bowl sweeps, after cloning/pulling the repository
on the cheaper transfer pod, run:

```bash
/opt/openvla/bin/python -u -m cloud.archive_sweep \
  --sweeps \
    scripted_lift_models_20261014_105736_941b29f8 \
    scripted_place_bowl_models_20261014_111236_f42bb9bf \
  --repo Maxalderone/robot-openvla-simulator-results
```

Enter the HF write token once at the hidden prompt. Follow the printed combined
log with `tail -F`. After detached launch the Mac may disconnect. The default
one-hour estimate yields a two-hour inner and three-hour outer deadline for
the combined job. Verification checks downloaded bytes at each pinned HF commit.
Cleanup remains separate and requires the per-sweep verified receipt.

### Authorized cleanup after both verified uploads

For the current lift and bowl transfers, add `--delete-after-verification` to
the combined command. The job first uploads and verifies both complete sweeps.
Only then does it retain compact reports alongside each verification receipt
and delete each original sweep folder and its `packed-parts` cache from the
network volume. Logs, receipts and training artifacts remain. Failed upload or
verification skips this cleanup; interrupted deletion may leave residual files
but the already verified HF archives remain available. A cleanup receipt records
the deleted paths. The same supervisor requests pod stop after cleanup or on
failure/timeout. Without this explicit flag, the default still retains originals.
