# Repository layout

| Folder | Purpose |
|---|---|
| `simulation/` | Panda controller, action contract, environment, camera, manual viewer and replay. |
| `simulation/scenes/` | Bowl and plate scene XML. The original `scene.xml` remains at the repository root. |
| `simulation/demos/` | Early joint, scripted pick/place and GUI smoke demonstrations. |
| `inference/` | OpenVLA predictor, prompts, diagnostics and local/remote policy loops. |
| `cloud/` | SSH sessions, HTTP serving/client, model preparation, dataset transfers and startup measurements. |
| `data_tools/` | Scripted demonstration collection, dataset verification and RLDS packaging. |
| `training/` | LoRA training integration/configs, continuation, batch policy and cloud supervisors/benchmark workers. |
| `training/upstream/` | Pinned official OpenVLA sources and their license; preserved during reorganization. |
| `analysis/` | Saved-run analysis, input audits, expert evaluation and timing/GPU report generators. |
| `common/` | Stable project paths, independent of the location of an implementation module. |
| `tests/` | Automated tests, with fixture states in `tests/fixtures/`. |
| `docs/` | Guides and findings grouped by topic; start with the [index](README.md). |
| `requirements/` | Dependency files; install from the repository root using `-r requirements/<file>.txt`. |
| `docker/` | Build recipes. Docker build context stays at the repository root. |
| `scripts/` | Shell cloud startup/stop and detached-training entry points. |
| `datasets/` | Existing generated demonstrations and portable training datasets. |
| `outputs/` | Existing run results, plots, benchmark measurements and saved replays. |
| `mujoco_menagerie/` | The unchanged upstream Git submodule and Panda mesh assets. |

Paths in this table are relative to the repository root. Generated data,
outputs, virtual environments and caches have not been relocated.

## Familiar commands

Nine small root launchers delegate to the implementation packages:

| Root command file | Implementation |
|---|---|
| `end_effector_demo.py` | `simulation.end_effector_demo` |
| `launch_sim.py` | `simulation.launch_sim` |
| `replay_run.py` | `simulation.replay_run` |
| `cloud_session.py` | `cloud.cloud_session` |
| `benchmark_cloud_session.py` | `cloud.benchmark_cloud_session` |
| `upload_training_dataset.py` | `cloud.upload_training_dataset` |
| `download_training_dataset.py` | `cloud.download_training_dataset` |
| `openvla_mac.py` | `inference.openvla_mac` |
| `train_panda.py` | `training.train_panda` |

Use them from the repository root as before. Secondary tools use module commands:

```bash
.venv/bin/python -m simulation.capture_camera
.venv/bin/python -m inference.remote_loop --help
.venv/bin/python -m cloud.training_cloud_session --help
.venv/bin/python -m data_tools.collect_panda_broad --help
.venv-training/bin/python -m analysis.summarize_gpu_benchmarks
```

The GPU comparison is now [docs/benchmarks/GPU_BATCH_COMPARISON.md](benchmarks/GPU_BATCH_COMPARISON.md).
The cloud benchmark synchronizer and report generator update this new location
automatically. Recorded per-run reports remain under `outputs/gpu-benchmarks/`.

## Compatibility and verification

The original root `scene.xml` keeps its bytes and path, and the root replay
launcher remains available. Existing per-run/per-trajectory `replay.py` scripts
can therefore find the repository as before. Scene fingerprints and the shared
action contract are unchanged. Bowl/plate XML references were updated for their
new folder; the runtime bowl grip settings remain in `simulation/bowl_scene.py`.

Shared IK retries are in `simulation/ik_retries.py`; macOS GUI interpreter
handling is in `simulation/viewer.py`. Simulation no longer needs to import a
policy loop just to obtain its retry helper.

Run automated tests with package names, for example:

```bash
.venv/bin/python -m unittest tests.test_bowl_scene tests.test_replay_run
.venv-training/bin/python -m unittest tests.test_training_preparation tests.test_cloud_session
```

Some tests require the simulation environment; others require the training
environment. The old `test_mujoco.py`, which opened a window immediately, is now
the manual demo `simulation/demos/mujoco_smoke.py` and is outside test discovery.

Dockerfile/workflow paths and build allowlists have been updated. Reorganization
does not change dependency versions, pretrained weights, adapters, dataset splits,
training configurations or stored cloud-volume paths. Local checks do not replace
an actual image build and GPU cloud smoke run.
