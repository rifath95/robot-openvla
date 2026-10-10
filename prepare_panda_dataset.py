"""Validate successful replayed episodes and build portable Panda JSONL/stats.

This is a staging dataset, not an RLDS training integration. It never loads a
model or mixes Bridge normalization into the Panda action labels.
"""

import argparse
from datetime import datetime, timezone
from dataset_paths import PANDA_DATASET_ROOT
import hashlib
import json
from pathlib import Path

import numpy as np

from panda_actions import CONVENTION_ID, PHYSICS_STEPS, action_contract, to_controller_action, from_controller_action


def fingerprint(folder):
    """Bind replay eligibility to commands, initial physics state and contract."""
    folder = Path(folder)
    metadata = json.loads((folder / 'episode.json').read_text())
    fixed = {key: metadata[key] for key in ('action_contract', 'instruction', 'scene_xml_sha256', 'mujoco_version')}
    if metadata.get('scenario') is not None:
        fixed['scenario'] = metadata['scenario']
    digest = hashlib.sha256(json.dumps(fixed, sort_keys=True).encode())
    for name in ('transitions.jsonl', 'initial_state.npz'):
        digest.update((folder / name).read_bytes())
    rows = [json.loads(line) for line in (folder / 'transitions.jsonl').read_text().splitlines()]
    for row in rows:
        for name in (row['observation'], row['next_observation'],
                     f"step_{row['index']:04}/after_state.npz"):
            path = (folder / name).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError('Episode artifact must remain inside its directory')
            digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_dataset(episodes, output):
    output = Path(output)
    records = []
    for episode in episodes:
        episode = Path(episode).resolve()
        metadata = json.loads((episode / 'episode.json').read_text())
        replay = json.loads((episode / 'replay.json').read_text())
        if metadata.get('action_contract') != action_contract():
            raise ValueError(f'Incompatible convention: {episode}')
        if not metadata.get('success') or not metadata.get('training_eligible') or not replay.get('passed'):
            raise ValueError(f'Only successful replay-verified expert episodes are eligible: {episode}')
        if replay.get('episode_sha256') != fingerprint(episode):
            raise ValueError(f'Episode changed since replay: {episode}')
        transitions = [json.loads(line) for line in (episode / 'transitions.jsonl').read_text().splitlines()]
        if not transitions or len(transitions) != metadata['transition_count'] or len(transitions) != replay['replayed_transitions']:
            raise ValueError('Episode transition count mismatch')
        for expected, row in enumerate(transitions, 1):
            if row['index'] != expected or not row['executed'] or row['physics_steps'] != PHYSICS_STEPS:
                raise ValueError('Rejected, missing, or partial transition')
            if abs(row['physics_seconds'] - metadata['physics_seconds_per_action']) > 1e-12:
                raise ValueError('Action duration mismatch')
            action = np.asarray(row['action'], dtype=float)
            bounded = from_controller_action(to_controller_action(action))
            if not np.allclose(action, bounded, atol=1e-12, rtol=0):
                raise ValueError('Executed label is outside the Panda contract bounds')
            if row['instruction'] != metadata['instruction']:
                raise ValueError('Instruction mismatch')
            # Resolve within the episode, never silently accept arbitrary paths.
            for key in ('observation', 'next_observation'):
                path = (episode / row[key]).resolve()
                if not path.is_relative_to(episode) or not path.is_file():
                    raise ValueError(f'Missing or invalid {key}')
            records.append(dict(episode=str(episode), index=row['index'],
                                image=str(episode / row['observation']),
                                next_image=str(episode / row['next_observation']),
                                instruction=row['instruction'], action=row['action'],
                                convention=CONVENTION_ID, physics_seconds=row['physics_seconds']))
    if not records:
        raise ValueError('At least one eligible expert episode is required')
    actions = np.array([row['action'] for row in records])
    statistics = dict(mean=actions.mean(axis=0).tolist(), std=actions.std(axis=0).tolist(),
                      min=actions.min(axis=0).tolist(), max=actions.max(axis=0).tolist(),
                      q01=np.quantile(actions, .01, axis=0).tolist(),
                      q99=np.quantile(actions, .99, axis=0).tolist(),
                      mask=[True] * 6 + [False])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'dataset.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
    (output / 'normalization_stats.json').write_text(json.dumps({CONVENTION_ID: {'action': statistics}}, indent=2) + '\n')
    report = dict(convention=action_contract(), episodes=len(episodes), transitions=len(records),
                  ready_for_finetuning=False,
                  limitations=['A scripted pilot is a pipeline check, not proof of adequate task coverage.',
                               'JSONL references local files; bundle episodes and regenerate paths after copying.',
                               'RLDS conversion, varied demonstrations, and training integration remain to be implemented.',
                               'Zero-variance dimensions must be handled explicitly when normalizing a training dataset.'])
    (output / 'dataset_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes', nargs='+', type=Path)
    parser.add_argument('--output-dir', type=Path, default=PANDA_DATASET_ROOT / 'staging' /
                        datetime.now(timezone.utc).strftime('dataset_%Y%m%d_%H%M%S'),
                        help='Defaults to a new staging directory under datasets/panda_pick_place')
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.episodes, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
