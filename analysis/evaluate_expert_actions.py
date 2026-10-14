"""Compare saved expert actions with real HTTP predictions; never move a robot.

Runs on the Mac using PNGs and original instructions from verified demonstrations.
No model, simulator, TensorFlow, or training dependencies are loaded locally.
"""

from common.output_layout import output_path

from common.paths import PROJECT_ROOT

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen
from urllib.parse import urlsplit
import uuid

import numpy as np

from cloud.connection_client import request_prediction
from data_tools.prepare_training_dataset import load_episodes, normalize

ROOT = PROJECT_ROOT
DEFAULT_SOURCE = ROOT / 'datasets/panda_pick_place/panda_expansion_20261010_104720/combined_manifest.json'
DEFAULT_DATASET = ROOT / 'datasets/panda_pick_place/training_v1'
CHECKPOINT_500 = '/workspace/panda-training/sessions/panda-train-20261010-233038-932/runs/training/checkpoint_000500'
NAMES = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']


def prepare_samples(source, dataset, split='train', selection='diagnostic', per_group=8):
    """Bind source images/labels to the prepared training bundle before sampling."""
    manifest = json.loads((dataset / 'manifest.json').read_text())
    statistics = json.loads((dataset / 'dataset_statistics.json').read_text())
    # The dataset audit already decoded images. Fingerprints recheck exact PNG
    # bytes here, avoiding a Pillow/TensorFlow dependency on the simulator Mac.
    episodes = load_episodes(source, verify_images=False)
    actual = {e['id']: (e['split'], len(e['rows']), e['fingerprint']) for e in episodes}
    expected = {e['id']: (e['split'], e['transitions'], e['sha256']) for e in manifest['episodes']}
    if actual != expected:
        raise ValueError('Source demonstrations do not match the prepared dataset')
    samples = []
    for episode in episodes:
        if episode['split'] != split:
            continue
        previous_stage = None
        for row in episode['rows']:
            stage = row['stage']
            # Isolate the transition of interest without adding a stage input to OpenVLA.
            group = 'first_lift' if stage == 'lift' and previous_stage != 'lift' else stage
            previous_stage = stage
            action = np.asarray(row['action'], dtype=np.float32)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError('Invalid expert action')
            samples.append(dict(episode_id=episode['id'], split=split, index=row['index'],
                                stage=stage, group=group, instruction=row['instruction'],
                                image=str(episode['path'] / row['observation']),
                                expert_action=action.tolist()))
    if not samples:
        raise ValueError('No examples in requested split')
    population = defaultdict(list)
    for sample in samples:
        population[sample['group']].append(sample)
    if selection == 'diagnostic':
        samples = []
        for group in sorted(population):
            rows = population[group]
            indices = np.linspace(0, len(rows)-1, min(per_group, len(rows)), dtype=int)
            samples.extend(rows[i] for i in indices)
    return samples, statistics, {k: len(v) for k, v in population.items()}


def check_server(health, statistics, expected_adapter):
    if health.get('mode') != 'openvla' or health.get('model_loaded') is not True:
        raise ValueError('A loaded OpenVLA server is required; fixed test actions are not accepted')
    info = health.get('startup_info', {})
    if info.get('unnorm_key') != 'panda_grasp_v1':
        raise ValueError('Server must use panda_grasp_v1 action statistics')
    if info.get('evaluation_provenance', {}).get('adapter_checkpoint') != expected_adapter:
        raise ValueError('Server adapter does not match --expected-adapter; restart with the updated launcher/code')
    server_stats = info.get('action_statistics', {})
    local_stats = statistics['panda_grasp_v1']['action']
    for key in ('q01', 'q99', 'mask'):
        if key not in server_stats or not np.array_equal(server_stats[key], local_stats[key]):
            raise ValueError(f'Server/local normalization mismatch: {key}')


def compare_actions(expert, predicted, statistics):
    expert, predicted = np.asarray(expert), np.asarray(predicted)
    if predicted.shape != (7,) or not np.isfinite(predicted).all():
        raise ValueError('Expected seven finite prediction values')
    # Labels were float32 in RLDS. Prediction values stay physical units as in deployment.
    errors = np.abs(predicted - expert)
    normalized_error = np.abs(normalize(predicted, statistics) - normalize(expert, statistics))
    return dict(absolute_error=errors.tolist(), normalized_absolute_error=normalized_error.tolist(),
                translation_l2_error_mm=float(np.linalg.norm(errors[:3])*1000),
                gripper_state_match=bool((predicted[6] >= .5) == (expert[6] >= .5)),
                expert_upward=bool(expert[2] > .001), predicted_upward=bool(predicted[2] > .001))


def aggregate(rows):
    if not rows:
        return dict(examples=0)
    errors = np.array([r['comparison']['absolute_error'] for r in rows])
    normalized = np.array([r['comparison']['normalized_absolute_error'] for r in rows])
    upward = [r for r in rows if r['comparison']['expert_upward']]
    return dict(examples=len(rows), mean_absolute_error=errors.mean(axis=0).tolist(),
                normalized_l1=float(normalized.mean()),
                mean_translation_l2_error_mm=float(np.mean([r['comparison']['translation_l2_error_mm'] for r in rows])),
                gripper_state_accuracy=float(np.mean([r['comparison']['gripper_state_match'] for r in rows])),
                expert_upward_examples=len(upward),
                predicted_upward_when_expert_upward=(float(np.mean([r['comparison']['predicted_upward'] for r in upward])) if upward else None))


def save_report(output, plan, rows, phase, error=None):
    groups = {group: aggregate([r for r in rows if r['group'] == group]) for group in plan['population_counts']}
    summary = dict(phase=phase, error=error, completed=len(rows), planned=len(plan['samples']),
                   overall=aggregate(rows), groups=groups, robot_moved=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    lines = ['# Expert-action evaluation', '', f'Status: **{phase}**. Completed {len(rows)}/{len(plan["samples"])} predictions.', '',
             f'Checkpoint: `{plan["expected_adapter"]}`',
             f'Split: `{plan["split"]}`. Selection: `{plan["selection"]}`. Original demonstration instructions used.', '',
             'Saved images are independent observations; no simulation or robot movement occurs.', '',
             '**These are autoregressively generated actions, not teacher-forced training metrics.**',
             'Normalized L1 uses training q01/q99 bounds (clipped), with gripper in [0,1].',
             'Gripper state uses a 0.5 threshold. Upward means dz > 1 mm per action.',
             'Translation error below is distance between predicted and expert commands, not execution tracking error.', '',
             'The diagnostic subset gives each group similar coverage; its overall average is not a full-dataset estimate.',
             'Evaluating all images gives full-split generated-action errors, still not task success or teacher-forced loss.', '',
             '| Group | Available | Evaluated | Translation error (mm) | Normalized L1 | Gripper agreement | Upward agreement* |',
             '|---|---:|---:|---:|---:|---:|---:|']
    health_path = output / 'server_health.json'
    if health_path.is_file():
        info = json.loads(health_path.read_text()).get('startup_info', {})
        gpu = info.get('gpu', {})
        gpu_text = gpu.get('name', 'not recorded')
        if 'total_vram_bytes' in gpu:
            gpu_text += f' ({gpu["total_vram_bytes"]/2**30:.1f} GiB VRAM)'
        lines[6:6] = ['', f'GPU: {gpu_text}. Model loading: {info.get("model_load_seconds", "not recorded")} seconds.',
                      f'Model directory: `{info.get("model_dir", "not recorded")}`. '
                      f'Action statistics: `{info.get("unnorm_key", "not recorded")}`.',
                      f'PyTorch: {info.get("pytorch_version", "not recorded")}; '
                      f'Transformers: {info.get("transformers_version", "not recorded")}.',
                      'Volume mount metadata: `'+json.dumps(info.get('volume_mount', {}), sort_keys=True)+'`.']
    for group, result in groups.items():
        if not result['examples']:
            lines.append(f'| {group} | {plan["population_counts"][group]} | 0 | — | — | — | — |')
            continue
        upward = result['predicted_upward_when_expert_upward']
        up = '—' if upward is None else f'{upward:.1%} ({result["expert_upward_examples"]} examples)'
        lines.append(f'| {group} | {plan["population_counts"][group]} | {result["examples"]} | '
                     f'{result["mean_translation_l2_error_mm"]:.3f} | {result["normalized_l1"]:.4f} | '
                     f'{result["gripper_state_accuracy"]:.1%} | {up} |')
    lines.extend(['', '*Among examples where the expert requested upward movement.', '',
                  'Per-example predictions, labels, instructions, image hashes, and timings: `predictions.jsonl`.',
                  'Per-component physical errors: `summary.json` (XYZ in metres, rotation in radians, gripper unitless).',
                  'Model loading, GPU, volume mount, and checkpoint provenance: `server_health.json`.', '',
                  'Interpretation: good expert-scene predictions plus poor rollouts suggest a state-coverage or execution issue. '
                  'Poor expert-scene predictions require investigating learning and the inference pipeline first. '
                  'Neither outcome alone proves a cause.'])
    if error:
        lines.extend(['', f'Error: {error}'])
    (output / 'EVALUATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    if plan.get('matched_diagnostic'):
        save_matched_report(output, plan, rows, phase, error)


def save_matched_report(output, plan, rows, phase, error=None):
    """Both action errors use the same physical labels and normalization bounds."""
    def metrics(selected):
        if not selected:
            return {'examples': 0}
        teacher_rows = [dict(comparison=r['teacher_forced_comparison']) for r in selected]
        teacher = [r['prediction']['teacher_forced'] for r in selected]
        return dict(examples=len(selected), generated=aggregate(selected), teacher_forced=aggregate(teacher_rows),
                    teacher_action_token_accuracy=float(np.mean([t['action_token_accuracy'] for t in teacher])),
                    teacher_loss_including_stop=float(np.mean([t['loss_including_stop'] for t in teacher])),
                    teacher_normalized_l1_token_targets=float(np.mean([t['normalized_l1_token_targets'] for t in teacher])),
                    teacher_invalid_action_token_count=sum(sum(t['invalid_action_tokens']) for t in teacher),
                    teacher_token_accuracy_per_component=np.mean([t['token_matches'] for t in teacher],axis=0).tolist())
    groups = {g: metrics([r for r in rows if r['group'] == g]) for g in plan['population_counts']}
    result = dict(phase=phase, error=error, completed=len(rows), planned=len(plan['samples']),
                  overall=metrics(rows), groups=groups, robot_moved=False)
    (output/'matched_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines = ['# Matched teacher-forced and generated-action diagnostic', '',
             f'Status: **{phase}**. Completed {len(rows)}/{len(plan["samples"])} examples.', '',
             f'Checkpoint: `{plan["expected_adapter"]}`. Split: `{plan["split"]}`.', '',
             'Each image/instruction is evaluated both ways with the same resident model. '
             'Normal generation runs first and never receives the expert action. '
             'Teacher forcing supplies preceding expert action tokens within this one action; '
             'it does not supply history of robot steps. No training or robot movement occurs.', '',
             'Both translation errors and normalized L1 below compare with the same original expert command. '
             'The separate teacher normalized_l1_token_targets field follows the training metric, comparing '
             'decoded action-token bin centers. Cross entropy includes the stop token; action accuracy excludes it.', '',
             '| Group | N | Generated XYZ error (mm) | Teacher XYZ error (mm) | Generated normalized L1 | Teacher normalized L1 | Teacher action-token accuracy |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, m in [('Overall',result['overall']), *groups.items()]:
        if not m['examples']:
            continue
        g,t = m['generated'],m['teacher_forced']
        lines.append(f'| {name} | {m["examples"]} | {g["mean_translation_l2_error_mm"]:.3f} | '
                     f'{t["mean_translation_l2_error_mm"]:.3f} | {g["normalized_l1"]:.4f} | '
                     f'{t["normalized_l1"]:.4f} | {m["teacher_action_token_accuracy"]:.1%} |')
    lines += ['', 'Per-component errors, gripper/upward agreement, teacher loss and invalid-token counts: '
              '`matched_summary.json`. Per-example token IDs, physical actions, image hashes and timings: '
              '`predictions.jsonl`. GPU, model loading, adapter provenance and storage metadata: '
              '`server_health.json`. Total elapsed time: `timing.json`.', '',
              'Teacher argmax uses the full vocabulary, matching training. Non-action predictions are flagged; '
              'the official decoder clips them to action bins, which can make their physical errors misleading.', '',
              'The balanced training-image subset is not a held-out success rate. '
              'This uses the merged resident inference checkpoint, not an exact replay of unmerged training '
              'weights, batches or autocast numerics. Good teacher forcing with poor generation suggests '
              'difficulty generating components without expert preceding tokens; errors in both indicate '
              'that these expert inputs are not reproduced reliably. Neither alone identifies the root cause.']
    if error:
        lines += ['',f'Error: {error}']
    (output/'MATCHED_REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='train')
    parser.add_argument('--selection', choices=('diagnostic', 'all'), default='diagnostic')
    parser.add_argument('--per-group', type=int, default=8)
    parser.add_argument('--server-url', default='http://127.0.0.1:8000')
    parser.add_argument('--expected-adapter', default=CHECKPOINT_500)
    parser.add_argument('--timeout-seconds', type=float, default=180)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Verify data and save a plan without contacting a server')
    parser.add_argument('--matched', action='store_true', help='Compare teacher forcing and normal generation on each example')
    args = parser.parse_args()
    if args.per_group <= 0 or not np.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error('Sample count and timeout must be positive')
    url = urlsplit(args.server_url)
    if (url.scheme != 'http' or url.hostname != '127.0.0.1' or url.path or url.query or url.fragment
            or url.username or url.password):
        parser.error('Use a loopback HTTP origin such as http://127.0.0.1:8000 through SSH')
    try:
        if url.port is None:
            raise ValueError('Missing port')
    except ValueError:
        parser.error('Specify a valid localhost port')
    samples, stats, counts = prepare_samples(args.source_manifest, args.dataset, args.split, args.selection, args.per_group)
    prefix = 'matched_evaluation' if args.matched else 'expert_evaluation'
    output = args.output_dir or output_path(f'{prefix}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}')
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(split=args.split, selection=args.selection, expected_adapter=args.expected_adapter,
                source_manifest=str(args.source_manifest.resolve()), dataset=str(args.dataset.resolve()),
                population_counts=counts, samples=samples, robot_moved=False,
                matched_diagnostic=args.matched)
    (output / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (output / 'dataset_statistics.json').write_text(json.dumps(stats, indent=2)+'\n')
    print(f'Evaluation outputs: {output}', flush=True)
    rows = []
    if args.dry_run:
        save_report(output, plan, rows, 'planned_only')
        print(f'Checked {sum(counts.values())} source examples; selected {len(samples)}. No server contacted.')
        return 0
    started = time.monotonic()
    try:
        with urlopen(args.server_url+'/health', timeout=10) as response:
            health = json.load(response)
        (output / 'server_health.json').write_text(json.dumps(health, indent=2)+'\n')
        check_server(health, stats, args.expected_adapter)
        if args.matched and (health.get('matched_diagnostic_available') is not True
                             or health.get('startup_info',{}).get('instruction_processing') !=
                             'lowercase to match official RLDSBatchTransform'):
            raise ValueError('Restart the server with the latest matched-diagnostic code')
        save_report(output, plan, rows, 'running')
        for i, sample in enumerate(samples, 1):
            options = {'expert_action': sample['expert_action']} if args.matched else {}
            prediction = request_prediction(sample['image'], sample['instruction'], server_url=args.server_url,
                                            expected_mode='openvla', timeout_seconds=args.timeout_seconds, **options)
            if prediction.get('unnorm_key') != 'panda_grasp_v1':
                raise ValueError('Prediction used unexpected normalization')
            row = dict(**sample, image_sha256=hashlib.sha256(Path(sample['image']).read_bytes()).hexdigest(),
                       prediction=prediction, comparison=compare_actions(sample['expert_action'], prediction['action'], stats))
            if args.matched:
                row['teacher_forced_comparison'] = compare_actions(
                    sample['expert_action'], prediction['teacher_forced']['action'], stats)
            with (output / 'predictions.jsonl').open('a') as file:
                file.write(json.dumps(row, allow_nan=False)+'\n')
            rows.append(row)
            save_report(output, plan, rows, 'running')
            print(f'{i}/{len(samples)}: {sample["episode_id"]} step {sample["index"]} ({sample["group"]}); '
                  f'expert dz={sample["expert_action"][2]:+.5f}, predicted dz={prediction["action"][2]:+.5f}', flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        save_report(output, plan, rows, 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', str(exc))
        print(f'Evaluation stopped; completed predictions preserved. {exc}', flush=True)
        return 1
    save_report(output, plan, rows, 'completed')
    (output / 'timing.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-started), indent=2)+'\n')
    report = 'MATCHED_REPORT.md' if args.matched else 'EVALUATION_REPORT.md'
    print(f'Completed. Read {output / report}. No robot moved.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
