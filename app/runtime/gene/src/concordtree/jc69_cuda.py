"""Fused Triton float64 JC69 pruning kernels (CPU Torch remains oracle)."""
import torch
import triton
import triton.language as tl

@triton.jit
def _read4(P, n, s, S:tl.constexpr,valid):
    offsets=(n*S+s)*4
    a=tl.load(P+offsets,valid,other=1.)
    b=tl.load(P+offsets+1,valid,other=1.)
    c=tl.load(P+offsets+2,valid,other=1.)
    d=tl.load(P+offsets+3,valid,other=1.)
    return a,b,c,d

@triton.jit
def _write4(P,n,s,S:tl.constexpr,a,b,c,d,valid):
    offsets=(n*S+s)*4
    tl.store(P+offsets,a,valid); tl.store(P+offsets+1,b,valid)
    tl.store(P+offsets+2,c,valid); tl.store(P+offsets+3,d,valid)

@triton.jit
def _up(U,X,C,V,LOG,N:tl.constexpr,S:tl.constexpr,H:tl.constexpr,B:tl.constexpr):
    nodepos=tl.program_id(0); node=tl.load(V+nodepos)
    s=tl.program_id(1)*B+tl.arange(0,B); valid=s<S
    p0=tl.full((B,),1.,tl.float64); p1=p0; p2=p0; p3=p0
    for h in tl.static_range(H):
        child=tl.load(C+node*H+h); x=tl.load(X+child)
        a,b,c,d=_read4(U,child,s,S,valid)
        common=(1-x)*(a+b+c+d)*.25
        p0*=x*a+common; p1*=x*b+common; p2*=x*c+common; p3*=x*d+common
    norm=tl.maximum(p0+p1+p2+p3,1e-300)
    _write4(U,node,s,S,p0/norm,p1/norm,p2/norm,p3/norm,valid)
    tl.store(LOG+nodepos*triton.cdiv(S,B)+tl.program_id(1),tl.sum(tl.where(valid,tl.log(norm),0.)))

@triton.jit
def _down(U,D,X,P,V,COUNT,S:tl.constexpr,B:tl.constexpr):
    node=tl.load(V+tl.program_id(0)); parent=tl.load(P+node); x=tl.load(X+node)
    s=tl.program_id(1)*B+tl.arange(0,B); valid=s<S
    u0,u1,u2,u3=_read4(U,node,s,S,valid)
    p0,p1,p2,p3=_read4(U,parent,s,S,valid)
    d0,d1,d2,d3=_read4(D,parent,s,S,valid)
    sumu=u0+u1+u2+u3; common=(1-x)*sumu*.25
    a0=d0*p0/tl.maximum(x*u0+common,1e-300)
    a1=d1*p1/tl.maximum(x*u1+common,1e-300)
    a2=d2*p2/tl.maximum(x*u2+common,1e-300)
    a3=d3*p3/tl.maximum(x*u3+common,1e-300)
    norma=tl.maximum(a0+a1+a2+a3,1e-300)
    a0/=norma; a1/=norma; a2/=norma; a3/=norma
    dot=a0*u0+a1*u1+a2*u2+a3*u3
    off=(1-x)*tl.maximum(sumu-dot,0.)*.25
    probability=off/tl.maximum(x*dot+(1-x)*sumu*.25,1e-300)
    tl.atomic_add(COUNT+node,tl.sum(tl.where(valid,probability,0.)))
    common=(1-x)*.25
    _write4(D,node,s,S,x*a0+common,x*a1+common,x*a2+common,x*a3+common,valid)

def expectation(sequence,lengths,*,leaf_ids,internals,nonroot,children,parents,
                block_sites,need_counts):
    import math
    import numpy as np
    device=lengths.device; m=len(lengths)-1
    x=torch.exp(-4*lengths/3)
    counts=torch.zeros_like(lengths); ll=torch.zeros((),device=device,dtype=torch.float64)
    for start in range(0,sequence.shape[1],block_sites):
        block=torch.as_tensor(sequence[:,start:start+block_sites].astype(np.int64),device=device)
        s=block.shape[1]
        up=torch.ones((m+1,s,4),device=device,dtype=torch.float64)
        obs=torch.nn.functional.one_hot(block.clamp_max(3),4).to(torch.float64)
        up[leaf_ids]=torch.where((block<4)[...,None],obs,torch.ones_like(obs))
        for level in reversed(internals):
            if not len(level): continue
            logs=torch.empty((len(level),triton.cdiv(s,256)),device=device,dtype=torch.float64)
            _up[(len(level),triton.cdiv(s,256))](up,x,children,level,logs,m,s,children.shape[1],256)
            ll+=logs.sum()
        ll-=s*math.log(4.)
        if need_counts:
            down=torch.ones_like(up)
            for level in nonroot:
                _down[(len(level),triton.cdiv(s,256))](up,down,x,parents,level,counts,s,256)
    return float(ll.cpu()),counts
