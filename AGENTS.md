# Simulator sweep result delivery

The agreed default is a two-stage workflow: run simulation on its GPU pod, save
results to the network volume and auto-stop; then launch a cheaper transfer pod
on the same volume to pack/upload to HF, verify and auto-stop. A CPU pod is
sufficient when compatible with the volume. Keep packing/upload work off the
simulator pod. Remind the user of this plan before deployment, and prepare the
transfer command and credential in advance. Honor explicit alternatives without
re-asking an already established choice.

Use `cloud.cloud_sweep_session` with its default `--delivery transfer-pod`, then
`cloud.archive_sweep` on the transfer pod. See `docs/cloud/TWO_STAGE_SWEEP_WORKFLOW.md`.

For the 1,080-trial sweep `scripted_multi_color_models_20261013_164548_eeda379b`,
the user switched to HF upload followed by pod stop after the direct download
hit its 30-minute timeout on October 14. The old temporary partial archive was
deleted by the failed downloader. Use the prepared HF archive helper for this
sweep; do not retry the old direct-fetch command.

For user-authorized routine volume cleanup, rely on the established record of
completed downloads and reviews. Do not repeat expensive file-by-file inventory
or checksum checks without a specific unresolved concern. Retain current work
whose download is still incomplete.
