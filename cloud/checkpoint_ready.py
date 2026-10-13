"""Reject checkpoint evaluation until the trainer's final metadata is published."""
import argparse
import json
from pathlib import Path
import struct


def check_ready(folder):
    folder=Path(folder)
    # save_adapter writes training_config only after the adapter and processor.
    for name in ('adapter_config.json','dataset_statistics.json','training_config.json',
                 'processor_config.json','preprocessor_config.json'):
        path=folder/name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Checkpoint is not fully saved: missing {name}')
        json.loads(path.read_text())
    weights=folder/'adapter_model.safetensors'
    with weights.open('rb') as stream:
        prefix=stream.read(8)
        if len(prefix)!=8:raise ValueError('Adapter weights are incomplete')
        length=struct.unpack('<Q',prefix)[0]
        if not 1<=length<=16*1024**2:raise ValueError('Invalid adapter weight header length')
        header=stream.read(length)
        if len(header)!=length:raise ValueError('Adapter weight header is incomplete')
    tensors=json.loads(header)
    if not isinstance(tensors,dict):raise ValueError('Invalid adapter weight header')
    offsets=[value['data_offsets'] for key,value in tensors.items() if key!='__metadata__']
    data_bytes=weights.stat().st_size-8-length
    if (not offsets or any(len(pair)!=2 or any(type(x) is not int for x in pair) or
                           not 0<=pair[0]<=pair[1]<=data_bytes for pair in offsets) or
        max(pair[1] for pair in offsets)!=data_bytes):
        raise ValueError('Adapter weights are incomplete or have inconsistent offsets')
    return folder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    args=parser.parse_args()
    try:check_ready(args.checkpoint)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(f'Checkpoint not ready: {exc}. Wait for the trainer to print Saved adapter checkpoint, then retry.',flush=True)
        return 1
    print(f'Checkpoint save verified: {args.checkpoint}',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
