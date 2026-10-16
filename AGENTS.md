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

All replay and action-replay buttons in generated WATCH_TRAJECTORIES.md and run
reports must directly launch local playback via common.replay_links and the
analysis.replay_launcher service. Do not link those buttons to Python source
files. Both playback modes start automatically; preserve manual CLI defaults.

# Training report delivery

Whenever the user chooses HF upload before a training pod auto-stops, include
and verify the training reports as well as the selected checkpoints. Preserve
plots, metrics, logs, configuration, dataset/source provenance, completion
status and available timings. Reports must be independently downloadable from
HF without downloading adapter weights or restarting a pod. Provide the
report-only download command and destination when the run completes.

# Resumable long training

For long training experiments, keep intermediate checkpoints as inference-ready
LoRA adapters. At the final checkpoint (also on a graceful runtime-limit exit),
save adapter weights plus optimizer state, cumulative update, RNG states and
input-iterator state so future continuation can restore training state. Back up
this final state to HF when HF delivery is selected, and keep report downloads
independent of optimizer/adapter payloads. Do not silently treat an adapter-only
continuation as a full resume.

For the uninterrupted 12,000-update colored bowl experiment (checkpoints 6,000,
8,000, 10,000, 12,000), the user explicitly chose **same-pod HF delivery** for
the separate sweep pod: run 800 trials, pack/upload results to HF, verify, then
auto-stop. Use cloud.color_bowl_sweep with the color_place_bowl_600_uninterrupted
profile. Do not substitute the default transfer-pod delivery for this experiment.

# Deadline policy

This is the standing default for **all future timed jobs**, including training,
benchmarks, simulator sweeps, transfers, HF uploads and chained workflows, unless
the user explicitly overrides it. For a complete planning budget B, use
inner = ceil(1.25 * B) and outer = ceil(1.50 * B). Both percentages refer to B.
Before deploying any job, verify its launcher follows this rule; update older
fixed-hour buffer implementations before using them. Do not change deadlines
of already-running jobs or rewrite historical run metadata.

Planning budgets must comfortably cover the complete job through saving, HF
delivery/verification, recovery allowance and pod shutdown. Set the inner
computation deadline to 125% of that budget and the outer deadline to 150% of
the original budget (not 150% of inner). At inner, stop starting computation
gracefully; allow checkpoint saving, packing, HF upload and verification to
continue until the outer guard. Preserve and label unfinished work as incomplete.
Reserve a short interval before outer for recovery cleanup and the pod stop API.
For the current run: training budget/inner/outer 8/10/12 h; sweep 12/15/18 h.

For the new uninterrupted 12,000-update colored bowl run, initialize LoRA from
the original saved checkpoint 0 of panda-train-color-bowl-20261015-094416-4344d31f,
using the pinned HF source and hashes in its training profile. Do not create a
new random LoRA initialization. Start a fresh optimizer at update zero and keep
it continuous through 12,000; save full training state at the final checkpoint.
