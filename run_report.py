"""Generate a readable timing/setup report from saved remote-loop artifacts."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean

MISSING = 'Not recorded'


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def seconds(value):
    return f'{value:.4f} s' if number(value) else MISSING


def memory(value):
    return f'{value / 1024**3:.2f} GiB' if number(value) else MISSING


def table(rows):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| Measurement | Value |', '| --- | ---: |'] +
                     [f'| {cell(key)} | {cell(value)} |' for key, value in rows])


def write_run_report(folder):
    folder = Path(folder)
    summary = read_json(folder / 'summary.json')
    status = read_json(folder / 'status.json')
    startup = read_json(folder / 'startup_info.json')
    records = [read_json(path) for path in sorted(folder.glob('step_*/execution.json'))]
    if not records:
        records = summary.get('steps', []) + summary.get('rejections', [])
    def value(row, group, key):
        return (row.get(group) or {}).get(key)
    def average(group, key):
        values = [value(row, group, key) for row in records]
        return mean(v for v in values if number(v)) if any(number(v) for v in values) else None
    def combined(parts):
        values = []
        for row in records:
            components = [value(row, group, key) for group, key in parts]
            if all(number(v) for v in components):
                values.append(sum(components))
        return mean(values) if values else None
    local_parts = [('timings_seconds', k) for k in
                   ('capture_and_state_save', 'ik', 'robot_execution_wall', 'next_capture_and_state_save')]
    local = combined(local_parts)
    request = average('request_timings_seconds', 'request_round_trip')
    cycle = average('timings_seconds', 'cycle')
    overhead = []
    network = []
    for row in records:
        parts = [value(row, g, k) for g, k in local_parts]
        rt = value(row, 'request_timings_seconds', 'request_round_trip')
        total = value(row, 'timings_seconds', 'cycle')
        server = value(row, 'server_timings_seconds', 'server_processing')
        if all(number(v) for v in [*parts, rt, total]):
            overhead.append(total - rt - sum(parts))
        if number(rt) and number(server):
            network.append(rt - server)
    gpu = startup.get('gpu', {})
    mounts = (startup.get('volume_mount') or {}).get('filesystems', [])
    mount = mounts[0] if mounts else {}
    fs = mount.get('fstype')
    source = mount.get('source', '')
    if fs in ('fuse.gee', 'fuse.geese'):
        storage = f'Global volume (inferred from {fs})'
    elif fs in ('nfs', 'nfs4') and '/networkvolumes/' in source:
        storage = f'Network volume (inferred from {fs} and mount source)'
    else:
        storage = f'Type unverified; filesystem {fs}' if fs else MISSING
    scales = Counter(row.get('action_scale') for row in records if row.get('executed'))
    tracking = [row.get('position_tracking_error_metres') for row in records]
    tracking = [v for v in tracking if number(v)]
    exported = startup.get('completed_export_present_before_setup')
    lines = [f'# Run report\n\nRun: `{folder.name}`',
             table([
                 ('Result', status.get('phase', summary.get('phase', MISSING))),
                 ('Completed movements', summary.get('completed_steps', status.get('completed_steps', MISSING))),
                 ('Requested cycles', summary.get('requested_steps', status.get('requested_steps', MISSING))),
                 ('Prediction attempts', summary.get('prediction_attempts', status.get('prediction_attempts', MISSING))),
                 ('Rejected/held cycles', summary.get('rejected_steps', status.get('rejected_steps', len(summary.get('rejections', []))))),
                 ('Stop reason', status.get('stop_reason') or summary.get('stop_reason') or 'None recorded'),
                 ('Active trial duration', seconds(summary.get('active_seconds', status.get('active_seconds')))),
                 ('Elapsed trial duration, including pauses', seconds(summary.get('elapsed_seconds', status.get('elapsed_seconds')))),
             ]),
             'Startup/model loading is separate from trial duration. These durations are not pod billing/runtime measurements.',
             '## Cloud setup and model loading',
             table([
                 ('GPU', gpu.get('name', MISSING)),
                 ('Total VRAM', memory(gpu.get('total_vram_bytes'))),
                 ('Used VRAM after loading', memory(gpu.get('used_vram_bytes_after_loading'))),
                 ('Free VRAM after loading', memory(gpu.get('free_vram_bytes_after_loading'))),
                 ('PyTorch allocated after loading', memory(gpu.get('pytorch_allocated_bytes_after_loading'))),
                 ('PyTorch reserved after loading', memory(gpu.get('pytorch_reserved_bytes_after_loading'))),
                 ('PyTorch peak allocated during loading', memory(gpu.get('pytorch_peak_allocated_bytes_during_loading'))),
                 ('Storage', storage), ('Mount source', source or MISSING),
                 ('Model location', startup.get('model_dir', MISSING)),
                 ('Completed export present before setup', 'Yes' if exported is True else 'No' if exported is False else MISSING),
                 ('Checkpoint validation/download/export preparation', seconds(startup.get('model_preparation_seconds'))),
                 ('Model loading', seconds(startup.get('model_load_seconds'))),
                 ('Device / precision / attention', ' / '.join(str(startup.get(k, MISSING)) for k in ('device', 'dtype', 'attention'))),
                 ('PyTorch / Transformers', ' / '.join(str(startup.get(k, MISSING)) for k in ('pytorch_version', 'transformers_version'))),
                 ('Visible host CPU count', startup.get('visible_cpu_count', MISSING)),
                 ('Visible host RAM', memory(startup.get('visible_host_ram_bytes'))),
                 ('Session ready time (UTC)', startup.get('ready_at_utc', MISSING)),
             ]),
             'Loading includes processor setup, checkpoint reading, model construction and synchronized GPU transfer; it does not isolate storage-to-VRAM transfer. A present export is validated before reuse. CPU/RAM visibility may exceed the pod allocation. Memory is a startup snapshot, not peak inference usage. Multiple trials can share one session loading measurement. Storage type is inferred only when the recorded mount identifies it; confirm volume name/type in RunPod.',
             '## Average full cycle — overview',
             table([
                 ('Remote request: start sending image → action received', seconds(request)),
                 ('Local simulation and recording', seconds(local)),
                 ('Remaining cycle overhead', seconds(mean(overhead) if overhead else None)),
                 ('Complete cycle', seconds(cycle)),
             ]),
             f'Averages use {len(records)} finalized cycle records, including rejected/held or interrupted movements when their timing records exist. Each measurement uses only records containing that field; missing data is never treated as zero. Cycles with no execution/timing report are excluded. Small sum differences can arise from rounding or differing coverage.',
             'The cycle starts before input capture and ends after resulting image/state capture. Remaining overhead includes encoding, connection setup, response validation, scheduling and recording work outside the named stages. Prediction waits and pauses during a cycle can affect its wall time.',
             '## Remote request breakdown',
             table([
                 ('Complete request round trip', seconds(request)),
                 ('Server processing included in round trip', seconds(average('server_timings_seconds', 'server_processing'))),
                 ('Remaining network/tunnel/request overhead', seconds(mean(network) if network else None)),
             ]),
             'Server processing is included in the round trip, not added to it. The remainder is their measured difference, not isolated network latency. Upload versus download network delays cannot reliably be separated.',
             table([(label, seconds(average('server_timings_seconds', key))) for label, key in [
                 ('Receive request body', 'receive_body'), ('Decode/prepare image', 'preprocess'),
                 ('Image/text processor and GPU transfer', 'processor_and_device_transfer'),
                 ('OpenVLA prediction', 'model_inference')]]),
             'Receive body: server reads incoming request bytes, possibly waiting for arrival. Decode/prepare: turns transmitted PNG data into an RGB image. Processor/transfer: resizes and normalizes pixels, tokenizes the instruction, creates tensors and copies them to the GPU; these operations are measured together. Prediction: vision encoding and action-token generation through the model, followed by decoding to seven numbers; original OpenVLA uses multiple internal forward passes per prediction. Server processing may include additional small overhead beyond these stages.',
             '## Local simulation and recording breakdown',
             table([(label, seconds(average('timings_seconds', key))) for label, key in [
                 ('Capture/save input image and state', 'capture_and_state_save'), ('Solve IK', 'ik'),
                 ('Execute robot movement', 'robot_execution_wall'),
                 ('Capture/save resulting image and state', 'next_capture_and_state_save')]] +
                   [('Local stages subtotal', seconds(local)),
                    ('After action receipt: IK + movement + next capture', seconds(combined(local_parts[1:])))]),
             'Movement is intentionally paced to simulated duration when the live viewer is enabled. Input and resulting images are captured separately to keep step records self-contained.',
             '## Controller and replay',
             table([
                 ('Full bounded actions', scales.get(1.0, 0)), ('Half-size actions', scales.get(.5, 0)),
                 ('Quarter-size actions', scales.get(.25, 0)), ('One-eighth-size actions', scales.get(.125, 0)),
                 ('Maximum position tracking error', f'{max(tracking)*1000:.3f} mm' if tracking else MISSING),
                 ('Recorded motion files', len(list(folder.glob('step_*/motion_frames.npz')))),
                 ('Playback', 'Run replay.py with the repository simulation environment' if (folder/'replay.py').exists() else 'Use replay_run.py with this run folder'),
             ]),
             'Completed movements demonstrate pipeline execution, not successful pickup/placement. Recorded frames support offline playback; older endpoint-only runs use approximate interpolated motion.',
             'Source files: `startup_info.json`, `status.json`, `summary.json`, and `step_*/execution.json`. Missing startup measurements are reported as Not recorded.']
    path = folder / 'RUN_REPORT.md'
    path.write_text('\n\n'.join(lines) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', nargs='+', type=Path)
    args = parser.parse_args()
    for folder in args.run_dirs:
        if not folder.is_dir():
            parser.error(f'Run folder not found: {folder}')
        print(write_run_report(folder))


if __name__ == '__main__':
    main()
