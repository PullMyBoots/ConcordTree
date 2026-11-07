"""Direct Gaussian arithmetic with coalesced transposed landmark reads.

The 15 ordered FP32 FMA operations, original coefficient dtype, 128-reference
partial sums and final FP64 tile reduction are unchanged.
"""
import torch
import triton
import triton.language as tl
from .barycenter_kernel import BarycenterKernel

@triton.jit
def _transposed_runtime(X,P,Z,C,OUT,N,M:tl.constexpr,T:tl.constexpr,
                        SCALE:tl.constexpr,BQ:tl.constexpr,BZ:tl.constexpr,RG:tl.constexpr):
    q=tl.program_id(0)*BQ+tl.arange(0,BQ)
    cls=tl.program_id(1);tile=tl.program_id(2)
    z=tile*BZ+tl.arange(0,BZ)
    distance=tl.full((BQ,BZ),0,tl.float32)
    for d in tl.static_range(15):
        perm=tl.load(P+cls*15+d)
        a=tl.load(X+q*15+perm,q<N,other=0)
        b=tl.load(Z+d*M+z,z<M,other=0)
        difference=a[:,None]-b[None,:]
        distance=tl.fma(difference,difference,distance)
    coefficient=tl.load(C+z,z<M,other=0)
    values=tl.exp(-distance*SCALE)*coefficient[None,:]
    if RG:
        value=tl.sum(tl.sum(tl.reshape(values,(BQ,BZ//RG,RG)),2),1)
    else:
        value=tl.sum(values,1)
    tl.store(OUT+(q*3+cls)*T+tile,value,q<N)

class TransposedBarycenterKernel(BarycenterKernel):
    def __init__(self,*args,query_tile=16,warps=4,reduction_group=0,**kwargs):
        super().__init__(*args,**kwargs)
        self.transposed=self.expanded.T.contiguous()
        self.query_tile=query_tile
        self.warps=warps
        self.reduction_group=reduction_group

    @torch.inference_mode()
    def predict_features(self,features):
        x=torch.as_tensor(features,device=self.device,dtype=torch.float32).contiguous()
        if x.ndim!=2 or x.shape[1]!=15:
            raise ValueError('quartet x15 barycenter features required')
        if len(x)==0:
            return torch.empty((0,3),device=self.device,dtype=torch.float32)
        tiles=triton.cdiv(len(self.expanded),128)
        partial=torch.empty((len(x),3,tiles),device=self.device,dtype=torch.float32)
        _transposed_runtime[(triton.cdiv(len(x),self.query_tile),3,tiles)](
            x,self.perms,self.transposed,self.expanded_coefficients,partial,
            len(x),len(self.expanded),tiles,1/(2*self.bandwidth**2),
            self.query_tile,128,self.reduction_group,num_warps=self.warps)
        return partial.sum(-1,dtype=torch.float64).float()
