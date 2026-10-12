"""Bind a training profile to its verified dataset without loading model weights."""
import argparse
import json
from pathlib import Path

PROFILE_KEYS=('dataset_bundle','expected_camera','expected_split_counts')


def select_dataset(config, explicit=None):
    expected=config.get('dataset_bundle')
    dataset=Path(explicit) if explicit else Path('/workspace/panda-training')/(expected or 'training_v1')
    if expected and dataset.name!=expected:
        raise ValueError(f'This profile requires {expected}; selected {dataset.name}')
    return dataset


def validate_dataset(config,dataset):
    dataset=select_dataset(config,dataset)
    if not any(key in config for key in PROFILE_KEYS):return
    manifest=json.loads((dataset/'manifest.json').read_text())
    verification=json.loads((dataset/'verification.json').read_text())
    inputs=json.loads((dataset/'input_check.json').read_text())
    if not verification.get('passed') or not inputs.get('passed') or not inputs.get('all_source_images_actions_instructions_match'):
        raise ValueError('Bowl training requires a verified export and source-input check')
    counts=config.get('expected_split_counts')
    if counts and any(record.get('split_counts')!=counts for record in (manifest,verification,inputs)):
        raise ValueError('Dataset split counts do not match the training profile')
    if config.get('expected_camera'):
        from simulation.panda_actions import action_contract
        if manifest.get('action_contract')!=action_contract(camera=config['expected_camera']):
            raise ValueError('Dataset camera/action contract does not match the training profile')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--dataset',type=Path)
    args=parser.parse_args()
    try:print(select_dataset(json.loads(args.config.read_text()),args.dataset))
    except ValueError as exc:parser.error(str(exc))


if __name__=='__main__':main()
