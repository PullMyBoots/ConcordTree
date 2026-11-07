"""Single-kernel FP64 implementation of the unchanged Hellinger barycenter."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _barycenter(C,O,G:tl.constexpr,B:tl.constexpr):
    q=tl.program_id(0);j=tl.arange(0,16);gene=tl.arange(0,B)
    total=tl.full((16,),0,tl.float64)
    for block in range(tl.cdiv(G,B)):
        g=block*B+gene
        c=tl.load(C+(q*G+g[:,None])*15+j[None,:],(g[:,None]<G)&(j[None,:]<15),other=0).to(tl.float64)
        c=tl.where(j[None,:]==0,0.,c);mass=tl.sum(c,1)
        ratio=libdevice.div_rn(c,tl.maximum(mass[:,None],1.))
        x=libdevice.sqrt_rn(ratio);total+=tl.sum(x,0)
    norm=libdevice.sqrt_rn(tl.sum(total*total,0))
    value=libdevice.div_rn(total,tl.where(norm>0,norm,1.))
    tl.store(O+q*15+j,value,j<15)

@torch.inference_mode()
def fused_barycenter(counts):
    """Private trusted nonnegative integer counter output,QxGx15 contiguous."""
    if not counts.is_cuda or counts.ndim!=3 or counts.shape[2]!=15 or not counts.is_contiguous():
        raise ValueError('resident contiguous QxGx15 counter output required')
    result=torch.empty((len(counts),15),device=counts.device,dtype=torch.float64)
    _barycenter[(len(counts),)](counts,result,counts.shape[1],64,num_warps=4,enable_fp_fusion=False)
    return result
