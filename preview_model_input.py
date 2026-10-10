"""Preview the cached OpenVLA image processor, without loading model weights."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache-openvla'))
os.environ.setdefault('HF_MODULES_CACHE', str(ROOT / '.cache-openvla/modules'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--model-dir', type=Path,
                        help='Local checkpoint folder. Defaults to the Mac download manifest.')
    args = parser.parse_args()
    folder = args.model_dir
    if folder is None:
        folder = Path(json.loads((ROOT / '.cache-openvla/model_revision.json').read_text())['path'])
    from PIL import Image
    import numpy as np
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(folder), local_files_only=True, trust_remote_code=True)
    image = Image.open(args.image).convert('RGB')
    inputs = processor('In: What action should the robot take to pick up the red cube?\nOut:', image)
    pixels = inputs['pixel_values'][0]
    metadata = dict(source_image_size=list(image.size), pixel_values_shape=list(inputs['pixel_values'].shape),
                    resize_strategy=processor.image_processor.image_resize_strategy,
                    model_dir=str(folder), weights_loaded=False, previews=[])
    for index, params in enumerate(processor.image_processor.tvf_normalize_params):
        channels = pixels[index * 3:(index + 1) * 3].detach().cpu().numpy()
        restored = channels * np.array(params['std'])[:, None, None] + np.array(params['mean'])[:, None, None]
        rgb = (np.clip(restored.transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)
        target = args.image.parent / f'model_input_backbone_{index}.png'
        Image.fromarray(rgb).save(target)
        red = (rgb[:, :, 0].astype(float) > 1.5 * rgb[:, :, 1]) & (rgb[:, :, 0] > 80)
        metadata['previews'].append(dict(path=str(target), red_pixel_count=int(red.sum())))
    (args.image.parent / 'model_input_preview.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
