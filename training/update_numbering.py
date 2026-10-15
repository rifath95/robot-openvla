"""Cumulative reporting for adapter-weight continuation; loop limits stay local."""
import json
import os
from pathlib import Path


def validate_initial_update(adapter, update):
    if type(update) is not int or update < 0:
        raise ValueError('Initial update must be a nonnegative integer')
    if update:
        if adapter is None:
            raise ValueError('Cumulative continuation requires an initial adapter')
        saved = json.loads((Path(adapter)/'training_config.json').read_text())
        if saved.get('checkpoint_update') != update:
            raise ValueError('Initial update differs from the saved adapter checkpoint')
    return update


def initial_update():
    update = int(os.environ.get('PANDA_INITIAL_UPDATE', '0'))
    if update < 0:
        raise ValueError('Negative initial update')
    return update


def cumulative_update(local_update):
    return initial_update()+local_update
