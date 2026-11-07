"""Explicit GPU endpoint/model loss score, no heterogeneous autograd tape.

Reuses unchanged spectral_cuda forward/Frechet-adjoint kernels AND GPU basis.
Only a small NumPy decoder Jacobian is new; pruning/count kernels unchanged.
"""
import numpy as np
import torch
import triton
from spectral_cuda import _forward, _backward
from numpy_model import decode_numpy


@torch.no_grad()
def prepare_model(parameters, vector, device):
    model=decode_numpy(parameters,vector);k=parameters.spec.gamma_categories+int(parameters.spec.invariant)
    m=20+2*k;d=len(parameters.initial)
    packed=torch.tensor(np.r_[model.values,model.jacobian.reshape(-1)],device=device,dtype=torch.float64)
    return packed[:4],packed[4:20].reshape(4,4),packed[20:20+k],packed[20+k:m],packed[m:].reshape(m,d)


@torch.no_grad()
def transitions(q,lengths,rates,pi):
    root=pi.sqrt();symmetric=root[:,None]*q/root[None,:]
    lam,u=torch.linalg.eigh((symmetric+symmetric.T)*.5)
    v=(u/root[:,None]).contiguous();inv=(u.T*root[None,:]).contiguous()
    e=lengths.numel();k=rates.numel()
    p=torch.empty((e,k,4,4),device=q.device,dtype=torch.float64)
    _forward[(triton.cdiv(e*k,32),)](lengths,rates,lam,v,inv,p,e,k,32,enable_fp_fusion=False)
    return p,(q,lengths,rates,pi,lam,v,inv)


@torch.no_grad()
def endpoint_score(p,tape,counts,root_counts,category_counts,weights,jacobian,sites,*,coordinate,raw_branches):
    q,t,r,pi,lam,v,inv=tape;e=t.numel();k=r.numel();blocks=triton.cdiv(e*k,32)
    # Same clamp/log derivative as the surrogate: zeros BELOW floor; safe even
    # for structural invariant-class zeros. Never differentiate counts.
    g=torch.where(p>=1e-300,(-counts/sites)/p.clamp_min(1e-300),0.).contiguous()
    part=torch.empty((blocks,4,4),device=q.device,dtype=torch.float64)
    gt=torch.empty((e,k),device=q.device,dtype=torch.float64)
    _backward[(blocks,)](g,p,q.contiguous(),t,r,lam,v,inv,part,gt,e,k,32,enable_fp_fusion=False)
    gq=inv.T@part.sum(0)@v.T
    gr=(gt*t[:,None]).sum(0)
    branch=(gt*r[None,:]).sum(1)
    if coordinate=='log':branch*=t
    elif coordinate=='sqrt':branch*=2*t/raw_branches  # exact existing exp(2log(s/2)) chain
    else:raise ValueError('coordinate must be log or sqrt')
    gp=(-root_counts/sites)/pi;gw=(-category_counts/sites)/weights
    primitive=torch.cat((gp,gq.reshape(-1),gr,gw))
    model=primitive@jacobian if jacobian.shape[1] else primitive.new_empty(0)
    return torch.cat((branch,model))


@torch.no_grad()
def evaluate(plan,parameters,values,*,coordinate,backend):
    e=len(plan.edges);sites=plan.sequence.shape[1]
    raw=torch.tensor(values[:e],device=plan.device,dtype=torch.float64)
    logt=2*(raw/2).log() if coordinate=='sqrt' else raw
    pi,q,r,w,jac=prepare_model(parameters,values[e:],plan.device)
    p,tape=transitions(q,logt.exp(),r,pi)
    ll,c,rc,cc=plan.expectation(p,pi,w,backend=backend)
    gradient=endpoint_score(p,tape,c,rc,cc,w,jac,sites,coordinate=coordinate,raw_branches=raw)
    # A single result copy transports the scalar loss and existing O(N+d) score.
    result=torch.cat((-ll.reshape(1)/sites,gradient)).cpu().numpy()
    return float(result[0]),result[1:]


@torch.no_grad()
def likelihood_only(plan,parameters,values,*,coordinate):
    e=len(plan.edges)
    raw=torch.tensor(values[:e],device=plan.device,dtype=torch.float64)
    t=(raw/2)**2 if coordinate=='sqrt' else raw.exp()
    pi,q,r,w,_=prepare_model(parameters,values[e:],plan.device)
    p,_=transitions(q,t,r,pi)
    return -float(plan.likelihood(p,pi,w).cpu())/plan.sequence.shape[1]
