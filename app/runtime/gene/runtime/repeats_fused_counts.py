"""Deterministic endpoint counts fused into the matrix posterior traversal.

Each same-node tile emits a small FP64 count partial and the usual FP32 child
posterior. There is no per-pattern factor tape, second endpoint reconstruction,
or floating-point atomic update. The final fixed nodewise sum is deterministic.
"""
import numpy as np
import torch
import triton
import triton.language as tl
from repeats_generic_cuda import upward


def count_tile_metadata(buckets, node_count):
    """Preserve posterior/CSR order; assign disjoint partial slots by node."""
    descriptors = []
    nodes = []
    for width, records in buckets:
        tile = max(1, min(128, 128//width))
        rec = np.asarray(records)
        cuts = np.r_[0, np.flatnonzero(rec[1:, 0] != rec[:-1, 0])+1, len(rec)]
        sizes = np.diff(cuts)
        blocks = (sizes+tile-1)//tile
        first = np.repeat(cuts[:-1], blocks)
        local = np.arange(int(blocks.sum()))-np.repeat(np.cumsum(blocks)-blocks, blocks)
        first = first+local*tile
        count = np.minimum(tile, np.repeat(sizes, blocks)-local*tile)
        target = rec[first, 0]
        descriptors.append(np.stack((first, count, target, np.zeros_like(first)), axis=1))
        nodes.append(target)
    target = np.concatenate(nodes)
    # Stable sorting preserves the existing bucket/tile order within each node.
    order = np.argsort(target, kind='stable')
    destination = np.empty_like(order)
    destination[order] = np.arange(len(order))
    offset = 0
    for descriptor in descriptors:
        n = len(descriptor)
        descriptor[:, 3] = destination[offset:offset+n]
        offset += n
    sizes = np.bincount(target, minlength=node_count).astype(np.int64)
    ranges = np.stack((np.cumsum(sizes)-sizes, sizes), axis=1)
    capacity = 1 << int(max(1, int(sizes.max()))-1).bit_length()
    return descriptors, ranges, capacity, len(target)


@triton.jit
def _down(U, THETA, PARTIAL, P, OFF, LINK, REC, BLOCK, K:tl.constexpr, NI,
          W:tl.constexpr, T:tl.constexpr, FLOOR:tl.constexpr):
    block=tl.program_id(0);k=tl.program_id(1)
    first=tl.load(BLOCK+block*4).to(tl.int64);n=tl.load(BLOCK+block*4+1)
    destination=tl.load(BLOCK+block*4+3).to(tl.int64)
    rr=first+tl.arange(0,T);valid=tl.arange(0,T)<n
    node=tl.load(REC+rr*5,valid,other=NI)
    row=tl.load(REC+rr*5+1,valid,other=0).to(tl.int64)
    local=tl.load(REC+rr*5+2,valid,other=0)
    start=tl.load(REC+rr*5+3,valid,other=0).to(tl.int64)
    degree=tl.load(REC+rr*5+4,valid,other=0)
    state=tl.arange(0,4)
    if W <= 128:
        # Preserve the original arithmetic and dispatch for the bounded tiles.
        w=tl.arange(0,W)
        mask=valid[:,None]&(w[None,:]<degree[:,None])
        parent=tl.load(LINK+start[:,None]+w[None,:],mask,other=0).to(tl.int64)
        posterior=tl.load(THETA+(parent[:,:,None]*K+k)*4+state[None,None,:],mask[:,:,None],other=0.)
        a=tl.sum(posterior,1)
    else:
        # A common subtree pattern can have hundreds of thousands of parents.
        # Bound the live vector by the existing 128-element tile budget, and
        # visit every CSR link with a runtime loop instead of expanding W.
        # FP64 accumulation limits additional roundoff across many chunks;
        # endpoint arithmetic below remains in the original posterior dtype.
        tl.static_assert(T == 1)
        a0=tl.full((),0,tl.float64);a1=tl.full((),0,tl.float64)
        a2=tl.full((),0,tl.float64);a3=tl.full((),0,tl.float64)
        degree0=tl.sum(degree,0);start0=tl.sum(start,0)
        for begin in range(0,degree0,128):
            w=begin+tl.arange(0,128)
            mask=w<degree0
            parent=tl.load(LINK+start0+w,mask,other=0).to(tl.int64)
            location=(parent*K+k)*4
            a0+=tl.sum(tl.load(THETA+location,mask,other=0.).to(tl.float64),0)
            a1+=tl.sum(tl.load(THETA+location+1,mask,other=0.).to(tl.float64),0)
            a2+=tl.sum(tl.load(THETA+location+2,mask,other=0.).to(tl.float64),0)
            a3+=tl.sum(tl.load(THETA+location+3,mask,other=0.).to(tl.float64),0)
        a=tl.where(state[None,:]==0,a0,tl.where(state[None,:]==1,a1,tl.where(state[None,:]==2,a2,a3))).to(THETA.dtype.element_ty)
    inner=node<NI;partial_row=tl.load(OFF+node).to(tl.int64)+local
    u=tl.load(U+(partial_row[:,None]*K+k)*4+state[None,:],valid[:,None]&inner[:,None],other=0.).to(THETA.dtype.element_ty)
    leaf=((local[:,None]==state[None,:])|(local[:,None]==4)).to(THETA.dtype.element_ty)
    u=tl.where(inner[:,None],u,leaf)
    p=tl.load(P+(node[:,None,None]*K+k)*16+state[None,:,None]*4+state[None,None,:],valid[:,None,None],other=0.).to(THETA.dtype.element_ty)
    den=tl.sum(p*u[:,None,:],2);ratio=a/tl.maximum(den,FLOOR)
    joint=(ratio[:,:,None]*p)*u[:,None,:]
    down=tl.sum(joint,1)
    tl.store(THETA+(row[:,None]*K+k)*4+state[None,:],down,valid[:,None]&inner[:,None])
    # Matrix products remain in the posterior dtype before FP64 tile sums.
    # Kernel fusion can change FP32 count roundoff relative to a stored tape;
    # this is algebraic equality, not a bitwise floating-point guarantee.
    counts=tl.sum(tl.where(valid[:,None,None],joint,0.).to(tl.float64),0)
    tl.store(PARTIAL+(destination*K+k)*16+state[:,None]*4+state[None,:],counts)


def downward(plan,width,data):
    records,blocks=data
    tile=max(1,min(128,128//width))
    floor=1e-300 if plan.ddtype==torch.float64 else 1e-30
    # All large degrees share one compiled shape. Degree is read from REC;
    # W only selects the bounded implementation, never truncates the input.
    compiled_width=width if width<=128 else 256
    _down[(len(blocks),plan.categories)](plan.up,plan.theta,plan.count_partials,plan.trans,
        plan.offsets,plan.links,records,blocks,plan.categories,plan.ni,compiled_width,tile,floor)


def reduce_counts(plan):
    from repeats_factorized_cuda import _sum_partials
    _sum_partials[(plan.m+1,plan.categories)](plan.count_partials,plan.count_ranges,
        plan.counts,plan.categories,plan.count_capacity)
