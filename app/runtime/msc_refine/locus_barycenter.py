"""Conditional Hellinger barycenter of observed-locus pattern frequencies.

This finite summary is not a characteristic distribution embedding or an
MSC consistency theorem. No gene tree/biological branch inference occurs.
"""
from __future__ import annotations
import numpy as np
import torch


def numpy_barycenter(counts):
    c=np.asarray(counts,dtype=np.float64).copy()
    if c.ndim!=3 or c.shape[-1]!=15 or not np.isfinite(c).all() or (c<0).any():
        raise ValueError('finite nonnegative quartet x observed-locus x15 counts required')
    c[...,0]=0;mass=c.sum(-1,keepdims=True)
    x=np.sqrt(np.divide(c,mass,out=np.zeros_like(c),where=mass>0)).sum(1)
    norm=np.linalg.norm(x,axis=-1,keepdims=True)
    return np.divide(x,norm,out=np.zeros_like(x),where=norm>0)


@torch.inference_mode()
def barycenter(counts):
    c=counts.to(torch.float64).clone()
    if c.ndim!=3 or c.shape[-1]!=15 or not bool(torch.isfinite(c).all()) or bool((c<0).any()):
        raise ValueError('finite nonnegative quartet x observed-locus x15 counts required')
    c[...,0]=0;mass=c.sum(-1,keepdim=True)
    x=torch.where(mass>0,c/mass.clamp_min(1),0.).sqrt().sum(1)
    norm=x.norm(dim=-1,keepdim=True)
    return torch.where(norm>0,x/norm.clamp_min(torch.finfo(x.dtype).tiny),0.)
