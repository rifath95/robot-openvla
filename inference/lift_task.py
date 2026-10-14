"""Predeclared unseen lift evaluation positions, independent of dataset collection."""
import numpy as np


def cube_position_for_seed(seed):
    if type(seed) is not int or seed<0:raise ValueError('Scene seed must be nonnegative')
    rng=np.random.default_rng([20261014,seed,4])
    return [float(rng.uniform(.40,.68)),float(rng.uniform(-.16,.16))]
