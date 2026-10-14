"""Predeclared placement layouts and cross-platform consistency checks."""
import math
import numpy as np
INSTRUCTION='pick up the red cube and place it in the blue bowl.'


def positions(slot):
    rng=np.random.default_rng([20261014,slot,5])
    for _ in range(10000):
        cube=rng.uniform([.40,-.16],[.68,.16]);bowl=rng.uniform([.40,-.16],[.68,.16])
        if .18<=np.linalg.norm(cube-bowl)<=.32:
            return dict(cube_xy=cube.tolist(),bowl_xy=bowl.tolist())
    raise RuntimeError('Unable to predeclare separated positions')


def layout(seed):
    p=positions(seed);return dict(red=p['cube_xy'],bowl=p['bowl_xy'])


def validate_positions(actual,expected,tolerance=1e-12):
    if not isinstance(actual,list) or len(actual)!=len(expected):raise ValueError('Saved placement layouts differ')
    for a,b in zip(actual,expected):
        if not isinstance(a,dict) or set(a)!={'red','bowl'}:raise ValueError('Saved placement layouts differ')
        for key in ('red','bowl'):
            if (not isinstance(a[key],list) or len(a[key])!=2 or
                any(type(x) not in (int,float) or not math.isfinite(x) or abs(x-y)>tolerance for x,y in zip(a[key],b[key]))):
                raise ValueError('Saved placement layouts differ')
