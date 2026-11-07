"""Bounded scratch fallback for unusually long or numerous observed loci.

All integer counts are retained. FP64 gene sums use the same 64-locus blocks
and order as tiled_barycenter, including zero padding of the final block.
"""
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .bit_orbits import _events


@triton.jit
def _add_genes(C,TOTAL,G:tl.constexpr):
    q=tl.program_id(0).to(tl.int64);j=tl.arange(0,16);lane=tl.arange(0,64)
    total=tl.load(TOTAL+q*16+j)
    for block in range(tl.cdiv(G,64)):
        g=block*64+lane
        c=tl.load(C+(q*G+g[:,None])*15+j[None,:],(g[:,None]<G)&(j[None,:]<15),other=0).to(tl.float64)
        c=tl.where(j[None,:]==0,0.,c);mass=tl.sum(c,1)
        x=libdevice.sqrt_rn(libdevice.div_rn(c,tl.maximum(mass[:,None],1.)))
        total+=tl.sum(x,0)
    tl.store(TOTAL+q*16+j,total)


@triton.jit
def _finish(TOTAL,O):
    q=tl.program_id(0);j=tl.arange(0,16)
    total=tl.load(TOTAL+q*16+j)
    norm=libdevice.sqrt_rn(tl.sum(total*total,0))
    tl.store(O+q*15+j,libdevice.div_rn(total,tl.where(norm>0,norm,1.)),j<15)


@torch.inference_mode()
def streamed_barycenter(planes,query,regions,device_regions,*,budget=16<<20,check=None):
    """Trusted scorer inputs; loci chunks end on the original 64-gene boundary."""
    q=len(query)
    result=torch.empty((q,15),device=planes.device,dtype=torch.float64)
    if not q:return result
    # The caller already batches queries; fallback can always count at least
    # one tile of a 64-gene block without exceeding the declared budget.
    if budget<q*(64*15*20+16*8+15*8):
        raise ValueError('query batch exceeds bounded counter scratch budget')
    total=torch.zeros((q,16),device=planes.device,dtype=torch.float64)
    first=0
    while first<len(regions):
        if check is not None:check()
        last=min(len(regions),first+4096)
        lengths=(regions[first:last,1]+31)//32-regions[first:last,0]//32
        # Avoid a single long locus multiplying thousands of tiny loci's work.
        if int(lengths.max())>4*max(1,float(np.median(lengths))):
            last=min(len(regions),first+64);lengths=lengths[:last-first]
        g=last-first
        fixed=q*(g*15*16+16*8+15*8)
        # Shrink on 64-gene boundaries to reserve int64 totals and tile scratch.
        if fixed+q*g*15*4>budget:
            g=max(64,((budget//q-16*8-15*8)//(15*20)//64)*64)
            last=min(last,first+g);g=last-first;lengths=lengths[:g]
            fixed=q*(g*15*16+16*8+15*8)
        tiles=triton.cdiv(int(lengths.max()),128)
        tile_batch=max(1,min(65535,(budget-fixed)//(q*g*15*4)))
        integer=torch.zeros((q,g,15),device=planes.device,dtype=torch.int64)
        for begin in range(0,tiles,tile_batch):
            if check is not None:check()
            t=min(tile_batch,tiles-begin)
            out=torch.empty((q,g,t,15),device=planes.device,dtype=torch.int32)
            _events[(q,g,t)](planes,query,device_regions[first:last],out,planes.shape[2],g,t,128,
                             begin,num_warps=1)
            integer.add_(out.sum(2,dtype=torch.int64))
            del out
        _add_genes[(q,)](integer,total,g,num_warps=4,enable_fp_fusion=False)
        del integer
        first=last
    _finish[(q,)](total,result,num_warps=4,enable_fp_fusion=False)
    return result
