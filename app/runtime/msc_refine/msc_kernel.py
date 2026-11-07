"""Group-invariant Gaussian conditional-mean model; finite statistical learner.

No unconditional MSC consistency claim. Model assets must be independently
simulated/trained, never fitted to benchmark labels. No neural computation.
"""
from __future__ import annotations
import itertools
import numpy as np
import torch
from .quartet_bank import pattern_layout,canonical_pattern


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


def collapse_numpy(counts):
    mapping=pattern_layout()[1][0]
    c=np.asarray(counts)
    return np.stack([c[...,mapping==j].sum(-1) for j in range(15)],-1)


def hellinger(counts,conditional=False):
    if conditional:
        counts=counts.clone();counts[...,0]=0
    return (counts/counts.sum(-1,keepdim=True).clamp_min(1)).sqrt()


@torch.inference_mode()
def group_kernel(x,landmarks,bandwidth,*,batch_rows=4096):
    """Exact8-element group average; no low-order approximation to Gaussian."""
    perms=torch.as_tensor(group_permutations(),device=x.device)
    z=landmarks[:,perms].reshape(-1,15).contiguous()
    zn=z.square().sum(-1);result=[]
    for first in range(0,len(x),batch_rows):
        a=x[first:first+batch_rows]
        distance=(a.square().sum(-1,keepdim=True)+zn[None,:]-2*(a@z.T)).clamp_min(0)
        k=torch.exp(-distance/(2*bandwidth**2))
        result.append(k.reshape(len(a),len(landmarks),8).mean(-1))
    return torch.cat(result)


class MSCKernel:
    def __init__(self,path,*,device='cuda:0'):
        z=np.load(path);self.device=torch.device(device)
        self.landmarks=torch.as_tensor(z['landmarks'],device=self.device)
        self.coefficients=torch.as_tensor(z['coefficients'],device=self.device)
        self.bandwidth=float(z['bandwidth'])
        self.conditional=bool(z['conditional']) if 'conditional' in z else False
        self.output_kind=str(z['output_kind']) if 'output_kind' in z else 'conditional_mean'
        self.perms=torch.as_tensor(candidate_permutations(),device=self.device)

    @torch.inference_mode()
    def predict_orbits(self,counts):
        c=torch.as_tensor(counts,device=self.device,dtype=self.landmarks.dtype)
        x=hellinger(c,self.conditional)[:,self.perms].reshape(-1,15)
        k=group_kernel(x,self.landmarks,self.bandwidth)
        # Finite KRR values are scores, not calibrated bounded probabilities.
        return (k@self.coefficients).reshape(-1,3)

    def predict_counts(self,counts):
        return self.predict_orbits(collapse_numpy(counts))

    def probabilities(self,counts):
        score=self.predict_orbits(counts)
        if self.output_kind=='logits':return score.softmax(-1)
        return simplex_projection(score)


def simplex_projection(scores):
    """Euclidean projection onto3-class probability simplex (no clipping knob).

    Projection cannot increase squared distance to a one-hot response. It
    constrains a finite conditional-mean estimate; it is not calibration.
    """
    u=scores.sort(-1,descending=True).values
    index=torch.arange(1,4,device=scores.device,dtype=scores.dtype)
    shift=(u.cumsum(-1)-1)/index
    rho=(u>shift).sum(-1).clamp_min(1)-1
    theta=shift.gather(-1,rho[:,None])
    return (scores-theta).clamp_min(0)
