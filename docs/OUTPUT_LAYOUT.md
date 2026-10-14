# Local outputs layout

`outputs/` is organized by purpose. Start with the generated local
[outputs index](../outputs/README.md), which links to sweep and training reports.

| Location | Contents |
|---|---|
| `outputs/sweeps/` | Scripted simulator sweeps, including all per-trial recordings and REPORT.md |
| `outputs/training/` | Downloaded fine-tuning reports, loss curves and metrics |
| `outputs/simulator_runs/cloud/` | Individual trials simulated on the Mac using a cloud model |
| `outputs/simulator_runs/local/` | Individual trials using the local model |
| `outputs/simulator_runs/sessions/` | Cloud inference session startup records |
| `outputs/simulator_runs/action_replays/` | Recorded expert actions executed through physics |
| `outputs/simulator_runs/demonstrations/` | Older standalone scripted demonstrations |
| `outputs/benchmarks/training/gpu-benchmarks/` | Downloaded GPU benchmark measurements and inference benchmarks |
| `outputs/archives/hf-sweeps/hf-sweep-archives/` | Downloaded HF sweep archive parts and manifests |
| `outputs/archives/hf-models/hf-archives/` | Model archive receipts/downloads |
| `outputs/collection_checks/` | Dataset collection checks and rejected attempts |
| `outputs/previews/` | Camera captures and scene previews |
| `outputs/diagnostics/` | Reviews, audits, pipeline checks and diagnostic logs |
| `outputs/volume_cleanup/` | Network-volume cleanup receipts |

The existing run names remain intact inside these categories. For example, the
latest sweep report is now
`outputs/sweeps/scripted_multi_color_models_20261013_164548_eeda379b/REPORT.md`.
Training downloads continue inside `outputs/training/phase3_training/`; the older
`phase3_training_1000/` report set is also retained under `training/`.

## New runs and old commands

Default local save locations use `common.output_layout.output_path`. Individual
Mac runs, cloud session metadata, sweeps, training downloads, benchmark downloads,
dataset collection checks and archive restoration follow this layout. Explicit
`--output-dir` arguments still select the caller's chosen location.

Old flat paths are accepted by sweep resume/fetch, the replay launcher, offline
replay, action replay, archive restoration, and the GPU training comparison
reader when the equivalent categorized location exists. This compatibility uses
path resolution, not hundreds of aliases cluttering the output root. Use the new
paths in fresh commands and links.

Compatibility applies to folder arguments passed to these modules. A command
that directly names a moved file, such as `python outputs/OLD_RUN/replay.py`,
needs its updated script path; use `python -m simulation.replay_run OLD_FOLDER`
to resolve an old folder argument automatically.

For example:

```bash
.venv/bin/python -m analysis.replay_launcher \
  --sweep outputs/sweeps/scripted_multi_color_models_20261013_164548_eeda379b
```

Restarting the replay launcher refreshes its local links. Reload the report's
Markdown Preview after migration. Relative per-trial links remain valid because
each complete sweep directory moves together.

## Migration and preserved data

The October 14 migration uses same-filesystem directory renames. It does not
copy or delete recordings, images, states, weights or datasets. Every moved entry
retains its filesystem identity. Collision checks run before moving anything;
the receipt records old and new locations under
`outputs/diagnostics/organization/output_layout_migration.json`.

References in maintained documentation and local Markdown reports are updated.
Copies of changed original local reports are retained under
`outputs/diagnostics/organization/original_reports/`. Raw provenance JSON, logs,
HF archive manifests and downloaded archive contents retain their original bytes;
historical paths inside those records describe the original run location.
Cloud network-volume paths and HF repository folder names are unchanged.

The helper `python -m data_tools.organize_outputs` previews any remaining flat
entries; `--apply` organizes them without overwriting existing destinations.
Outputs remain excluded from Git, while this guide and path-handling code are
tracked.
