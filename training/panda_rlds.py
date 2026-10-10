"""Project adapter: portable TFDS episodes -> official OpenVLA batch transform.

Unlike the OXE mixture loader, this reads our explicit validation/test splits and
uses the saved training-only statistics without recomputing them on held-out data.
"""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset

from prepare_training_dataset import normalize


class PandaRLDSDataset(IterableDataset):
    def __init__(self, data_root_dir, data_mix, batch_transform, resize_resolution=None,
                 shuffle_buffer_size=128, train=True, image_aug=False, split=None):
        import tensorflow as tf
        import tensorflow_datasets as tfds
        tf.config.set_visible_devices([], 'GPU')  # TensorFlow must not reserve training VRAM.
        if image_aug:
            raise ValueError('Image augmentation is disabled for the first controlled Panda experiment')
        if data_mix != 'panda_grasp_v1':
            raise ValueError('This adapter only accepts panda_grasp_v1')
        self.root=Path(data_root_dir)
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.dataset_statistics=json.loads((self.root/'dataset_statistics.json').read_text())
        self.split=split or ('train' if train else 'validation')
        self.batch_transform=batch_transform
        self.builder=tfds.builder_from_directory(str(self.root/self.manifest['builder_directory']))
        # Shuffle encoded PNGs, not full 640x480 arrays, to keep CPU RAM modest.
        decoders={'steps':{'observation':{'image':tfds.decode.SkipDecoding()}}}
        episodes=self.builder.as_dataset(split=self.split,shuffle_files=train,decoders=decoders)
        self.dataset=episodes.flat_map(lambda ep:ep['steps']).filter(lambda step:step['is_action_valid'])
        options=tf.data.Options();options.threading.private_threadpool_size=2
        self.dataset=self.dataset.with_options(options)
        if train:
            self.dataset=self.dataset.shuffle(shuffle_buffer_size,seed=20261010,
                                             reshuffle_each_iteration=True).repeat()
        self.dataset_length=self.manifest['split_counts'][self.split]['transitions']

    def __iter__(self):
        for row in self.dataset.as_numpy_iterator():
            with Image.open(io.BytesIO(row['observation']['image'])) as im:
                image=np.asarray(im.convert('RGB'))
            batch={'dataset_name':b'panda_grasp_v1',
                   'action':normalize(row['action'],self.dataset_statistics)[None,:],
                   'observation':{'image_primary':image[None,...]},
                   'task':{'language_instruction':row['language_instruction']}}
            yield self.batch_transform(batch)

    def __len__(self):
        return self.dataset_length
