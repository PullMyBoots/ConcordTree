"""Group-invariant Gaussian conditional-mean model; finite statistical learner.

No unconditional MSC consistency claim. Model assets must be independently
simulated/trained, never fitted to benchmark labels. No neural computation.
"""
from __future__ import annotations
import itertools
import numpy as np
import torch
from .pattern_layout import pattern_layout,canonical_pattern


def orbit_permutation(order):
    reps,_,_=pattern_layout();lookup={tuple(x):i for i,x in enumerate(reps)}
    dest=np.array([lookup[canonical_pattern(row[list(order)])] for row in reps])
    return np.argsort(dest)


def group_permutations():
    perms=[]
    for swap,left,right in itertools.product((False,True),repeat=3):
        a=[0,1];b=[2,3]
        if left:a.reverse()
        if right:b.reverse()
        perms.append(orbit_permutation(b+a if swap else a+b))
    return np.stack(perms)


def candidate_permutations():
    return np.stack([orbit_permutation(o) for o in [(0,1,2,3),(0,2,1,3),(0,3,1,2)]])


class MSCKernel:
    def __init__(self,path,*,device='cuda:0'):
        z=np.load(path);self.device=torch.device(device)
        self.landmarks=torch.as_tensor(z['landmarks'],device=self.device)
        self.coefficients=torch.as_tensor(z['coefficients'],device=self.device)
        self.bandwidth=float(z['bandwidth'])
        self.conditional=bool(z['conditional']) if 'conditional' in z else False
        self.output_kind=str(z['output_kind']) if 'output_kind' in z else 'conditional_mean'
        self.perms=torch.as_tensor(candidate_permutations(),device=self.device)


