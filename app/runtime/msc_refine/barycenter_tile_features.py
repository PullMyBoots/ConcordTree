"""FP64 barycenter directly from exact int32 word-tile counts.

Integer tile sums stay in registers. Division, square root, gene-block sums
and normalization follow the original fused_barycenter operation order.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _single_gene(C,O,T:tl.constexpr,BT:tl.constexpr):
    q=tl.program_id(0).to(tl.int64);j=tl.arange(0,16)
    if BT<=256:
        t=tl.arange(0,BT)
        c=tl.load(C+(q*T+t[:,None])*15+j[None,:],(t[:,None]<T)&(j[None,:]<15),other=0).to(tl.int64)
        count=tl.sum(c,0).to(tl.float64)
    else:
        integers=tl.full((256,16),0,tl.int64)
        for begin in range(0,T,256):
            t=begin+tl.arange(0,256)
            integers+=tl.load(C+(q*T+t[:,None])*15+j[None,:],(t[:,None]<T)&(j[None,:]<15),other=0).to(tl.int64)
        count=tl.sum(integers,0).to(tl.float64)
    count=tl.where(j==0,0.,count)
    mass=tl.sum(count,0)
    value=libdevice.sqrt_rn(libdevice.div_rn(count,tl.maximum(mass,1.)))
    norm=libdevice.sqrt_rn(tl.sum(value*value,0))
    value=libdevice.div_rn(value,tl.where(norm>0,norm,1.))
    tl.store(O+q*15+j,value,j<15)

@triton.jit
def _tile_barycenter(C,O,G:tl.constexpr,T:tl.constexpr,BG:tl.constexpr):
    q=tl.program_id(0).to(tl.int64);j=tl.arange(0,16);gene=tl.arange(0,BG)
    total=tl.full((16,),0,tl.float64)
    for block in range(tl.cdiv(G,BG)):
        g=block*BG+gene
        integers=tl.full((BG,16),0,tl.int64)
        for t in range(T):
            c=tl.load(C+((q*G+g[:,None])*T+t)*15+j[None,:],(g[:,None]<G)&(j[None,:]<15),other=0).to(tl.int64)
            integers+=c
        c=integers.to(tl.float64)
        c=tl.where(j[None,:]==0,0.,c);mass=tl.sum(c,1)
        ratio=libdevice.div_rn(c,tl.maximum(mass[:,None],1.))
        x=libdevice.sqrt_rn(ratio);total+=tl.sum(x,0)
    norm=libdevice.sqrt_rn(tl.sum(total*total,0))
    value=libdevice.div_rn(total,tl.where(norm>0,norm,1.))
    tl.store(O+q*15+j,value,j<15)

@torch.inference_mode()
def tiled_barycenter(counts):
    if not counts.is_cuda or counts.dtype!=torch.int32 or counts.ndim!=4 or counts.shape[3]!=15 or not counts.is_contiguous():
        raise ValueError('resident contiguous QxGxTx15 int32 counts required')
    g,t=counts.shape[1:3]
    if g<1 or t<1:raise ValueError('nonempty loci and word tiles required')
    result=torch.empty((len(counts),15),device=counts.device,dtype=torch.float64)
    if not len(counts):return result
    if g==1:
        _single_gene[(len(counts),)](counts,result,t,triton.next_power_of_2(t),num_warps=4,enable_fp_fusion=False)
    else:
        _tile_barycenter[(len(counts),)](counts,result,g,t,64,num_warps=4,enable_fp_fusion=False)
    return result
