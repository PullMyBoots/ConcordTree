"""Same FP64 transition/Frechet mathematics, one packed host-to-device copy.

The small model/Jacobian already lives on the host. Compute its symmetric 4x4
eigensystem there as well; keep the O(N*K) exponentials and all pruning/count
work on the GPU. No autograd host tape or precision change is introduced.
"""
import numpy as np
import torch
import triton
from numpy_model import decode_numpy
from spectral_cuda import _forward

@torch.no_grad()
def packed_transition(parameters,vector,raw,device,decoded=None):
    if decoded is None:decoded=decode_numpy(parameters,vector)
    k=parameters.spec.gamma_categories+int(parameters.spec.invariant)
    m=20+2*k;d=len(parameters.initial);e=len(raw)
    pi=decoded.values[:4];q=decoded.values[4:20].reshape(4,4)
    root=np.sqrt(pi);sym=root[:,None]*q/root[None,:]
    lam,u=np.linalg.eigh((sym+sym.T)*.5)
    v=u/root[:,None];inv=u.T*root[None,:]
    host=np.r_[decoded.values,decoded.jacobian.reshape(-1),lam,v.reshape(-1),inv.reshape(-1),raw,(raw/2)**2]
    packed=torch.tensor(host,device=device,dtype=torch.float64)
    pi=packed[:4];q=packed[4:20].reshape(4,4);rates=packed[20:20+k];weights=packed[20+k:m]
    jac=packed[m:m+m*d].reshape(m,d);offset=m+m*d
    lam=packed[offset:offset+4];v=packed[offset+4:offset+20].reshape(4,4);inv=packed[offset+20:offset+36].reshape(4,4)
    raw=packed[offset+36:offset+36+e];lengths=packed[offset+36+e:]
    p=torch.empty((e,k,4,4),device=device,dtype=torch.float64)
    _forward[(triton.cdiv(e*k,32),)](lengths,rates,lam,v,inv,p,e,k,32,enable_fp_fusion=False)
    return pi,q,rates,weights,jac,p,(q,lengths,rates,pi,lam,v,inv),raw
