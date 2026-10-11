"""Compare cached processor inputs and saved rollout alignment, without loading weights."""

from common.paths import PROJECT_ROOT
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = PROJECT_ROOT


def audit(rollout, processor_dir, output):
    os.environ.setdefault('HF_MODULES_CACHE', str(ROOT / '.cache-openvla/modules'))
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from data_tools.check_training_inputs import official_batch_transform
    from analysis.evaluate_expert_actions import prepare_samples, DEFAULT_SOURCE, DEFAULT_DATASET
    from inference.openvla_prompt import action_prompt
    from data_tools.prepare_training_dataset import load_episodes, normalize
    scene_hash = hashlib.sha256(b''.join((ROOT / name).read_bytes() for name in
        ('scene.xml', 'mujoco_menagerie/franka_emika_panda/panda.xml'))).hexdigest()

    samples, statistics, population = prepare_samples(DEFAULT_SOURCE, DEFAULT_DATASET)
    episodes = [e for e in load_episodes(DEFAULT_SOURCE, verify_images=False) if e['split'] == 'train']
    processor = AutoProcessor.from_pretrained(str(processor_dir), local_files_only=True, trust_remote_code=True)
    transform = official_batch_transform(processor)

    def inputs(sample, prompt):
        with Image.open(sample['image']) as image:
            image = image.convert('RGB')
            batch = transform({'dataset_name': b'panda_grasp_v1',
                               'action': normalize(sample['expert_action'], statistics)[None, :],
                               'observation': {'image_primary': np.asarray(image)[None, ...]},
                               'task': {'language_instruction': sample['instruction'].encode()}})
            inference = processor(prompt, image)
        ids = inference['input_ids'][0]
        if ids[-1] != 29871:
            ids = torch.cat([ids, ids.new_tensor([29871])])
        prefix = batch['input_ids'][batch['labels'] == -100]
        return torch.equal(prefix, ids), torch.equal(batch['pixel_values'], inference['pixel_values'][0])

    checked = []
    for sample in samples:
        fixed, pixels = inputs(sample, action_prompt(sample['instruction']))
        original, _ = inputs(sample, f"In: What action should the robot take to {sample['instruction']}?\nOut:")
        checked.append({'episode_id': sample['episode_id'], 'index': sample['index'],
                        'original_prompt_matches_training': original,
                        'fixed_prompt_matches_training': fixed, 'pixels_identical': pixels})
    if not all(r['fixed_prompt_matches_training'] and r['pixels_identical'] for r in checked):
        raise ValueError('Training/inference processor mismatch remains')

    # Verify every distinct training instruction, including ones outside the 80-image sample.
    instructions = sorted({r['instruction'] for e in episodes for r in e['rows']})
    representative = samples[0]
    for instruction in instructions:
        sample = dict(representative, instruction=instruction)
        if not inputs(sample, action_prompt(instruction))[0]:
            raise ValueError(f'Instruction token mismatch: {instruction}')

    summary = json.loads((rollout / 'summary.json').read_text())
    startup = json.loads((rollout / 'startup_info.json').read_text())
    if startup.get('unnorm_key') != 'panda_grasp_v1' or startup.get('action_statistics') != statistics['panda_grasp_v1']['action']:
        raise ValueError('Saved cloud action statistics differ from training')
    steps = summary['steps']
    first_close = next(s for s in steps if s['raw_action'][6] < .5)
    # This project's Panda state has 7 arm + 2 finger coordinates, then cube free joint.
    state = np.load(rollout / f"step_{first_close['step']:03d}" / 'scene_state.npz')
    if state['qpos'].shape != (16,):
        raise ValueError('Unexpected Panda/cube state layout')
    cube = state['qpos'][9:12]
    live_offset = float(np.linalg.norm(np.asarray(first_close['before_position'])[:2] - cube[:2]) * 1000)
    expert_close = []
    for episode in episodes:
        meta = json.loads((episode['path'] / 'episode.json').read_text())
        close = next(r for r in episode['rows'] if r['stage'] == 'close')
        before = np.load(episode['path'] / Path(close['observation']).parent / 'before_state.npz')
        offset = float(np.linalg.norm(np.asarray(close['control_position_before'])[:2] - before['qpos'][9:11]) * 1000)
        expert_close.append({'episode_id': episode['id'], 'offset_xy_mm': offset,
                             'scene_matches_current': meta['scene_xml_sha256'] == scene_hash})
    result = {'weights_loaded': False, 'cloud_requests': 0, 'processor_dir': str(processor_dir),
              'training_examples': sum(population.values()), 'distinct_training_instructions': len(instructions),
              'checked_images': checked, 'first_live_close_step': first_close['step'],
              'live_close_offset_xy_mm': live_offset, 'expert_close': expert_close,
              'saved_cloud_action_statistics_match': True,
              'original_prompt_mismatches': sum(not r['original_prompt_matches_training'] for r in checked),
              'fixed_prompt_mismatches': 0, 'pixel_mismatches': 0}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    offsets = [r['offset_xy_mm'] for r in expert_close]
    (output / 'AUDIT_REPORT.md').write_text(f'''# OpenVLA pipeline and grasp alignment audit

Local CPU audit. No model weights loaded; no cloud requests or robot movement.
Rollout: `{rollout.name}`. Processor: `{processor_dir}`.

| Check | Result |
|---|---|
| Original inference prompt | {result['original_prompt_mismatches']}/{len(checked)} sampled prompts differ from training tokens |
| Fixed inference prompt | 0/{len(checked)} sampled prompts differ; all {len(instructions)} distinct training instructions also pass |
| Image tensors | Bit-for-bit equal for all {len(checked)} sampled images, across both vision backbones |
| Saved cloud action normalization | Same panda_grasp_v1 q01/q99/mask and other statistics as the training bundle |
| Demonstration scene | {sum(r['scene_matches_current'] for r in expert_close)}/{len(expert_close)} training episodes match current scene fingerprint |
| Expert horizontal offset at closing | Median {np.median(offsets):.3f} mm; maximum {max(offsets):.3f} mm across {len(offsets)} training episodes |
| Live horizontal offset at closing | {live_offset:.3f} mm at action {first_close['step']} |

Training's official RLDSBatchTransform lowercases instructions. Previous inference retained uppercase world X/Y, yielding different tokens. Inference now uses the same lowercase convention. Original instruction text remains in saved requests and results.

The image path is consistent: the project RLDS adapter decodes the full 640×480 RGB PNG and calls the same image processor as inference; its resize_resolution argument does not resize the image beforehand. The cached processor uses `{processor.image_processor.image_resize_strategy}` and outputs two stacked vision backbones. The official prompt prefix, including token 29871, matches after the fix.

The live hand closed while misaligned; expert close images depict an aligned hand. Thus the successful expert-image close predictions do not imply successful alignment or recovery on the rollout's own states. This observation and the text mismatch are separate findings. The audit does not prove that capitalization caused the failed grasp, nor validate the merged cloud checkpoint numerically.

Next: repeat the same checkpoint-1000 expert-image evaluation and live trial after updating server code. Reuse existing base weights and adapters; no new dataset, fine-tuning or Docker rebuild is required. Compare against the saved pre-fix results before deciding on further training.
''')
    print(output / 'AUDIT_REPORT.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--processor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.rollout.resolve(), args.processor.resolve(), args.output.resolve())
