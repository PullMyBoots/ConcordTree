"""Same direct Gaussian arithmetic, with runtime row count to avoid JIT churn."""
import torch
import triton
import triton.language as tl
from .kernel_direct import DirectMSCKernel
from .msc_kernel import hellinger

@triton.jit
def _direct_runtime(X,P,Z,C,OUT,N,M:tl.constexpr,T:tl.constexpr,
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

class DynamicDirectMSCKernel(DirectMSCKernel):
    @torch.inference_mode()
    def predict_orbits(self,counts):
        c=torch.as_tensor(counts,device=self.device,dtype=torch.float32)
        x=hellinger(c,self.conditional).contiguous();tiles=triton.cdiv(len(self.expanded),128)
        partial=torch.empty((len(x),3,tiles),device=self.device,dtype=torch.float32)
        _direct_runtime[(triton.cdiv(len(x),16),3,tiles)](x,self.perms,self.expanded,self.expanded_coefficients,partial,
            len(x),len(self.expanded),tiles,1/(2*self.bandwidth**2),16,128,num_warps=4)
        return partial.sum(-1,dtype=torch.float64).float()
