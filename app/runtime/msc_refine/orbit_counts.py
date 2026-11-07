"""Exact streamed15-orbit site counts, no256-bin quartet state retained."""
from __future__ import annotations
import numpy as np
import torch
import triton
import triton.language as tl
from .quartet_bank import pattern_layout,canonical_pattern


@triton.jit
def _count(P,Q,MAP,OUT,STRIDE,BEGIN,END,BLOCKS:tl.constexpr,WIDTH:tl.constexpr):
    query=tl.program_id(0);block=tl.program_id(1)
    site=BEGIN+block*WIDTH+tl.arange(0,WIDTH)
    shift=((site%2)*4).to(tl.int32)
    a=tl.load(Q+query*4);b=tl.load(Q+query*4+1);c=tl.load(Q+query*4+2);d=tl.load(Q+query*4+3)
    aa=(tl.load(P+a*STRIDE+site//2,site<END,other=0x44).to(tl.int32)>>shift)&15
    bb=(tl.load(P+b*STRIDE+site//2,site<END,other=0x44).to(tl.int32)>>shift)&15
    cc=(tl.load(P+c*STRIDE+site//2,site<END,other=0x44).to(tl.int32)>>shift)&15
    dd=(tl.load(P+d*STRIDE+site//2,site<END,other=0x44).to(tl.int32)>>shift)&15
    valid=(site<END)&(aa<4)&(bb<4)&(cc<4)&(dd<4)
    code=aa*64+bb*16+cc*4+dd
    orbit=tl.load(MAP+code,valid,other=15)
    value=tl.histogram(orbit,32);j=tl.arange(0,32)
    tl.store(OUT+(query*BLOCKS+block)*15+j,value,j<15)


@torch.inference_mode()
def orbit_counts(packed,queries,length,*,slab_sites=1<<20,width=8192):
    """Resident native nibble-packed MSA; bounded site-partial scratch.

    Caller batches queries. Complete-case unknown handling is identical to the
    256-pattern backend. Counts are exact int32 for length<2**31. Slab/width
    are execution layout only and cannot change the mathematical result.
    """
    if (not packed.is_cuda or packed.dtype!=torch.uint8 or packed.ndim!=2 or
        not queries.is_cuda or queries.ndim!=2 or queries.shape[1]!=4 or
        not 0<length<2**31 or packed.shape[1]!=(length+1)//2):
        raise ValueError('resident packed CUDA MSA, Qx4 queries and valid length required')
    if width<128 or width&(width-1) or slab_sites%width:raise ValueError('power-of-two width divides slab')
    q=queries.to(torch.int64).contiguous()
    if bool(((q<0)|(q>=len(packed))).any()) or bool((q.sort(1).values.diff(dim=1)==0).any()):
        raise ValueError('distinct in-range query taxa required')
    mapping=torch.as_tensor(pattern_layout()[1][0],device=packed.device,dtype=torch.int32)
    result=torch.zeros((len(q),15),device=packed.device,dtype=torch.int32)
    for begin in range(0,length,slab_sites):
        end=min(length,begin+slab_sites);blocks=triton.cdiv(end-begin,width)
        partial=torch.empty((len(q),blocks,15),device=packed.device,dtype=torch.int32)
        _count[(len(q),blocks)](packed,q,mapping,partial,packed.shape[1],begin,end,blocks,width,num_warps=4)
        result+=partial.sum(1,dtype=torch.int32)
    return result


def resolution_maps():
    reps,_,_=pattern_layout();lookup={tuple(p):i for i,p in enumerate(reps)}
    maps=[]
    for order in ((0,1,2,3),(0,2,1,3),(0,3,1,2)):
        maps.append([lookup[canonical_pattern(p[list(order)])] for p in reps])
    return np.asarray(maps,dtype=np.int64)


def resolved_counts(counts):
    """Taxon permutation is an exact permutation of the15 integer bins."""
    result=torch.empty((*counts.shape[:-1],3,15),device=counts.device,dtype=counts.dtype)
    for s,mapping in enumerate(resolution_maps()):
        # source bin j becomes mapping[j] in the re-ordered quartet.
        result[...,s,:].zero_()
        result[...,s,:].scatter_add_(-1,torch.as_tensor(mapping,device=counts.device).expand_as(counts),counts)
    return result
