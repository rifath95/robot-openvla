"""Predeclared unseen lift evaluation positions, independent of dataset collection."""
import numpy as np


def cube_position_for_seed(seed):
    if type(seed) is not int or seed<0:raise ValueError('Scene seed must be nonnegative')
    rng=np.random.default_rng([20261014,seed,4])
    return [float(rng.uniform(.40,.68)),float(rng.uniform(-.16,.16))]


def validate_positions(actual,expected,tolerance=1e-12):
    """Accept cross-platform rounding while rejecting changed seeds or scenes."""
    import math
    if not isinstance(actual,list) or len(actual)!=len(expected):
        raise ValueError('Saved lift positions differ from seeds')
    for saved,generated in zip(actual,expected):
        if not isinstance(saved,dict) or set(saved)!={'red'}:
            raise ValueError('Saved lift positions differ from seeds')
        xy=saved['red']
        if (not isinstance(xy,list) or len(xy)!=2 or
            any(type(a) not in (int,float) or not math.isfinite(a) or abs(a-b)>tolerance
                for a,b in zip(xy,generated['red']))):
            raise ValueError('Saved lift positions differ from seeds')
