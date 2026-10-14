"""Local output categories and compatibility with the former flat layout.

Cloud volume paths and original provenance records are not rewritten.
"""
from pathlib import Path
from common.paths import OUTPUTS_ROOT

OUTPUT_CATEGORIES=('sweeps','training','simulator_runs','benchmarks','archives',
                   'diagnostics','collection_checks','previews','volume_cleanup')


def category(name):
    if name.startswith('scripted_'):return 'sweeps'
    if name.startswith('phase3_training') or name=='phase3_smoke_loss.png':return 'training'
    if name=='gpu-benchmarks':return 'benchmarks/training'
    if name=='hf-sweep-archives':return 'archives/hf-sweeps'
    if name=='hf-archives':return 'archives/hf-models'
    if name=='cloud-sweeps':return 'sweeps/cloud-work'
    if name.startswith('cloud_session_'):return 'simulator_runs/sessions'
    if name.startswith(('remote_loop_','remote_single_action_','remote_inference_')):return 'simulator_runs/cloud'
    if name.startswith(('openvla_loop_','single_action_')):return 'simulator_runs/local'
    if name.startswith('action_replay_'):return 'simulator_runs/action_replays'
    if name.startswith(('panda_demo_','pick_place_')):return 'simulator_runs/demonstrations'
    if 'collection_checks' in name or name=='panda_dataset_v1':return 'collection_checks'
    if name.endswith('_scene_preview') or name in ('camera_review','camera_image.png'):return 'previews'
    if name=='volume-cleanups':return 'volume_cleanup'
    return 'diagnostics'


def output_path(name,*,root=None):
    """Canonical destination for a logical output name, without creating it."""
    relative=Path(name)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('Use a relative output name without parent traversal')
    return (Path(root) if root is not None else OUTPUTS_ROOT)/category(relative.parts[0])/relative


def resolve_output_path(path):
    """Resolve old local outputs/foo paths after migration; leave others alone."""
    path=Path(path)
    if path.exists():return path
    absolute=path.absolute()
    try:relative=absolute.relative_to(OUTPUTS_ROOT)
    except ValueError:return path
    candidate=output_path(relative)
    return candidate if candidate.exists() else path
