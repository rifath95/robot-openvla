# Simulator sweep result delivery

Before the user deploys a paid pod for a future simulator sweep, remind them to
choose the result-delivery method during preparation: direct download to the Mac,
or cloud-to-Hugging-Face archiving so the GPU can stop before a slow Mac download.
Honor an already stated choice without asking again. Prepare the selected flow
and credentials before deployment; do not assume the sweep launcher automatically
uploads to HF. See `docs/cloud/HF_SWEEP_ARCHIVE.md` for the separate archive helper.

For the 1,080-trial sweep `scripted_multi_color_models_20261013_164548_eeda379b`,
the user switched to HF upload followed by pod stop after the direct download
hit its 30-minute timeout on October 14. The old temporary partial archive was
deleted by the failed downloader. Use the prepared HF archive helper for this
sweep; do not retry the old direct-fetch command.

For user-authorized routine volume cleanup, rely on the established record of
completed downloads and reviews. Do not repeat expensive file-by-file inventory
or checksum checks without a specific unresolved concern. Retain current work
whose download is still incomplete.
