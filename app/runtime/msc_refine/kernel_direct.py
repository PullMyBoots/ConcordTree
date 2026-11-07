"""Direct coordinate-difference Gaussian expansion; same mathematical score."""
import torch
import triton
import triton.language as tl
from .msc_kernel import MSCKernel,hellinger,group_permutations

@triton.jit
def _direct(X,P,Z,C,OUT,N:tl.constexpr,M:tl.constexpr,T:tl.constexpr,
            SCALE:tl.constexpr,BQ:tl.constexpr,BZ:tl.constexpr):
    q=tl.program_id(0)*BQ+tl.arange(0,BQ);cls=tl.program_id(1);tile=tl.program_id(2)
    z=tile*BZ+tl.arange(0,BZ);distance=tl.full((BQ,BZ),0,tl.float32)
    for d in tl.static_range(15):
        perm=tl.load(P+cls*15+d)
        a=tl.load(X+q*15+perm,q<N,other=0)
        b=tl.load(Z+z*15+d,z<M,other=0)
        difference=a[:,None]-b[None,:]
        distance=tl.fma(difference,difference,distance)
    coefficient=tl.load(C+z,z<M,other=0)
    value=tl.sum(tl.exp(-distance*SCALE)*coefficient[None,:],1)
    tl.store(OUT+(q*3+cls)*T+tile,value,q<N)

class DirectMSCKernel(MSCKernel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        p=torch.as_tensor(group_permutations(),device=self.device)
        self.expanded=self.landmarks[:,p].reshape(-1,15).contiguous()
        self.expanded_coefficients=(self.coefficients/8).repeat_interleave(8).contiguous()

    @torch.inference_mode()
    def predict_orbits(self,counts):
        c=torch.as_tensor(counts,device=self.device,dtype=torch.float32)
        x=hellinger(c,self.conditional).contiguous();tiles=triton.cdiv(len(self.expanded),128)
        partial=torch.empty((len(x),3,tiles),device=self.device,dtype=torch.float32)
        _direct[(triton.cdiv(len(x),16),3,tiles)](x,self.perms,self.expanded,self.expanded_coefficients,partial,
            len(x),len(self.expanded),tiles,1/(2*self.bandwidth**2),16,128,num_warps=4)
        return partial.sum(-1,dtype=torch.float64).float()
