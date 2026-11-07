"""Implicit endpoint scratch: J_ij=(a_i P_ij) U_j, store only four a_i.

Matrix CSR propagation/tiling is unchanged. FP64 count partials are reduced
in a fixed two-stage order, without atomic floating additions.
Reconstruct each joint in the SAME message dtype before summing in FP64.
This is a scratch representation change, not a rank approximation or estimator.
"""
import torch
import triton
import triton.language as tl
from repeats_generic_cuda import upward


@triton.jit
def _down(U, THETA, FACTOR, P, OFF, LINK, REC, R, K:tl.constexpr, NI,
          W:tl.constexpr, T:tl.constexpr, FLOOR:tl.constexpr):
    rr=tl.program_id(0)*T+tl.arange(0,T);k=tl.program_id(1);valid=rr<R
    node=tl.load(REC+rr*5,valid,other=NI)
    row=tl.load(REC+rr*5+1,valid,other=0)
    local=tl.load(REC+rr*5+2,valid,other=0)
    start=tl.load(REC+rr*5+3,valid,other=0)
    degree=tl.load(REC+rr*5+4,valid,other=0)
    w=tl.arange(0,W);state=tl.arange(0,4)
    mask=valid[:,None]&(w[None,:]<degree[:,None])
    parent=tl.load(LINK+start[:,None]+w[None,:],mask,other=0)
    posterior=tl.load(THETA+(parent[:,:,None]*K+k)*4+state[None,None,:],mask[:,:,None],other=0.)
    a=tl.sum(posterior,1)
    inner=node<NI;partial_row=tl.load(OFF+node)+local
    u=tl.load(U+(partial_row[:,None]*K+k)*4+state[None,:],valid[:,None]&inner[:,None],other=0.).to(THETA.dtype.element_ty)
    leaf=((local[:,None]==state[None,:])|(local[:,None]==4)).to(THETA.dtype.element_ty)
    u=tl.where(inner[:,None],u,leaf)
    p=tl.load(P+(node[:,None,None]*K+k)*16+state[None,:,None]*4+state[None,None,:],valid[:,None,None],other=0.).to(THETA.dtype.element_ty)
    den=tl.sum(p*u[:,None,:],2);ratio=a/tl.maximum(den,FLOOR)
    # Four ratios suffice because P and U are already resident until next pass.
    tl.store(FACTOR+(row[:,None]*K+k)*4+state[None,:],ratio,valid[:,None])
    joint=ratio[:,:,None]*p*u[:,None,:]
    down=tl.sum(joint,1)
    tl.store(THETA+(row[:,None]*K+k)*4+state[None,:],down,valid[:,None]&inner[:,None])


@triton.jit
def _reduce(U, FACTOR, P, OFF, CHUNKS, COUNT, K:tl.constexpr, NI, B:tl.constexpr):
    chunk=tl.program_id(0);k=tl.program_id(1)
    node=tl.load(CHUNKS+chunk*3);start=tl.load(CHUNKS+chunk*3+1);n=tl.load(CHUNKS+chunk*3+2)
    s=tl.arange(0,B);state=tl.arange(0,4);valid=s<n
    row=start+s;local=row-tl.load(OFF+node);inner=node<NI
    ratio=tl.load(FACTOR+(row[:,None]*K+k)*4+state[None,:],valid[:,None],other=0.)
    u=tl.load(U+(row[:,None]*K+k)*4+state[None,:],valid[:,None]&inner,other=0.).to(FACTOR.dtype.element_ty)
    leaf=((local[:,None]==state[None,:])|(local[:,None]==4)).to(FACTOR.dtype.element_ty)
    u=tl.where(inner,u,leaf)
    p=tl.load(P+(node*K+k)*16+state[:,None]*4+state[None,:]).to(FACTOR.dtype.element_ty)
    # Preserve multiplication order AND message-precision rounding. Reducing
    # ratio*U first, or using FP64 joints for mixed mode, would change semantics.
    joint=(ratio[:,:,None]*p[None,:,:])*u[:,None,:]
    count=tl.sum(joint.to(tl.float64),0)
    tl.store(COUNT+(chunk*K+k)*16+state[:,None]*4+state[None,:],count)


def downward(plan,width,records):
    tile=max(1,min(128,128//width))
    floor=1e-300 if plan.ddtype==torch.float64 else 1e-30
    _down[(triton.cdiv(len(records),tile),plan.categories)](
        plan.up,plan.theta,plan.joint,plan.trans,plan.offsets,plan.links,records,
        len(records),plan.categories,plan.ni,width,tile,floor)


@triton.jit
def _sum_partials(PART,RANGE,COUNT,K:tl.constexpr,B:tl.constexpr):
    node=tl.program_id(0).to(tl.int64);k=tl.program_id(1)
    first=tl.load(RANGE+node*2).to(tl.int64);size=tl.load(RANGE+node*2+1)
    q=tl.arange(0,16)
    if B<=4096:
        j=tl.arange(0,B)
        values=tl.load(PART+((first+j[:,None])*K+k)*16+q[None,:],j[:,None]<size,other=0.)
        count=tl.sum(values,0)
    else:
        # Bound the compiled tensor even for millions of endpoint partials.
        # FP64 sums include every original partial; no floating atomic update.
        lane=tl.arange(0,1024)
        values=tl.full((1024,16),0,tl.float64)
        for begin in range(0,size,1024):
            j=begin.to(tl.int64)+lane
            values+=tl.load(PART+((first+j[:,None])*K+k)*16+q[None,:],j[:,None]<size,other=0.).to(tl.float64)
        count=tl.sum(values,0)
    tl.store(COUNT+(node*K+k)*16+q,count)

def reduce_counts(plan):
    _reduce[(len(plan.chunks),plan.categories)](
        plan.up,plan.joint,plan.trans,plan.offsets,plan.chunks,plan.count_partials,
        plan.categories,plan.ni,128)
    _sum_partials[(plan.m+1,plan.categories)](
        plan.count_partials,plan.count_ranges,plan.counts,plan.categories,plan.count_capacity)
