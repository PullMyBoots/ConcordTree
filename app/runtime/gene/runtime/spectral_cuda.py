"""Optional fused FP64 evaluation of the SAME reversible exponential/Frechet VJP.

No eigenvector differentiation. spectral.py and torch.matrix_exp remain oracles.
Only first derivatives are supported, as required by the existing L-BFGS-B fit.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _mm(a,b):
    # FP64 4x4 multiplication, batched first axis. No TF32/tensor-core cast.
    return tl.sum(a[:,:,:,None]*b[:,None,:,:],2)


@triton.jit
def _forward(T,R,LAM,V,INV,P,E,K:tl.constexpr,B:tl.constexpr):
    x=tl.program_id(0)*B+tl.arange(0,B);valid=x<E*K
    a=tl.arange(0,4)
    t=tl.load(T+x//K,valid,other=0.)*tl.load(R+x%K,valid,other=0.)
    lam=tl.load(LAM+a)
    vm=tl.load(V+a[:,None]*4+a[None,:])
    im=tl.load(INV+a[:,None]*4+a[None,:])
    em1=libdevice.expm1(t[:,None]*lam[None,:])
    p=_mm(vm[None,:,:]*em1[:,None,:],im[None,:,:])
    p+=tl.cast(a[:,None]==a[None,:],tl.float64)[None,:,:]
    tl.store(P+x[:,None,None]*16+a[None,:,None]*4+a[None,None,:],tl.maximum(p,0.),valid[:,None,None])


@triton.jit
def _backward(G,P,Q,T,R,LAM,V,INV,PART,GT,E,K:tl.constexpr,B:tl.constexpr):
    block=tl.program_id(0)
    x=block*B+tl.arange(0,B);valid=x<E*K;a=tl.arange(0,4)
    ptr=x[:,None,None]*16+a[None,:,None]*4+a[None,None,:]
    g=tl.load(G+ptr,valid[:,None,None],other=0.)
    p=tl.load(P+ptr,valid[:,None,None],other=0.)
    q=tl.load(Q+a[:,None]*4+a[None,:])
    vt=tl.load(V+a[None,:]*4+a[:,None])
    it=tl.load(INV+a[None,:]*4+a[:,None])
    inner=_mm(_mm(vt[None,:,:],g),it[None,:,:])
    lam=tl.load(LAM+a)
    t=tl.load(T+x//K,valid,other=0.)*tl.load(R+x%K,valid,other=0.)
    delta=tl.abs(lam[:,None]-lam[None,:])
    z=t[:,None,None]*delta[None,:,:]
    phi=tl.where(z<1e-5,1-z*.5+z*z/6-z*z*z/24,
                 -libdevice.expm1(-z)/tl.maximum(z,1e-300))
    high=tl.maximum(lam[:,None],lam[None,:])
    divided=t[:,None,None]*libdevice.exp(t[:,None,None]*high[None,:,:])*phi
    part=tl.sum(inner*divided,0)
    tl.store(PART+block*16+a[:,None]*4+a[None,:],part)
    gt=tl.sum(tl.sum(g*_mm(q[None,:,:],p),2),1)
    tl.store(GT+x,gt,valid)


class _CudaReversibleExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,lengths,rates,pi):
        if q.device.type!='cuda' or q.dtype!=torch.float64:
            raise ValueError('spectral-cuda requires CUDA FP64 inputs')
        root=pi.sqrt()
        symmetric=root[:,None]*q/root[None,:]
        symmetric=(symmetric+symmetric.T)*.5
        lam,u=torch.linalg.eigh(symmetric)
        v=(u/root[:,None]).contiguous();inv=(u.T*root[None,:]).contiguous()
        e=lengths.numel();k=rates.numel()
        p=torch.empty((e,k,4,4),device=q.device,dtype=q.dtype)
        _forward[(triton.cdiv(e*k,32),)](lengths,rates,lam,v,inv,p,e,k,32,
                                      enable_fp_fusion=False)
        ctx.save_for_backward(q,lengths,rates,p,lam,v,inv)
        return p

    @staticmethod
    def backward(ctx,g):
        q,t,r,p,lam,v,inv=ctx.saved_tensors
        e=t.numel();k=r.numel();blocks=triton.cdiv(e*k,32)
        part=torch.empty((blocks,4,4),device=q.device,dtype=q.dtype)
        gt=torch.empty((e,k),device=q.device,dtype=q.dtype)
        _backward[(blocks,)](g.contiguous(),p,q.contiguous(),t,r,lam,v,inv,part,gt,e,k,32,
                            enable_fp_fusion=False)
        gq=inv.T@part.sum(0)@v.T
        return gq,(gt*r[None,:]).sum(1),(gt*t[:,None]).sum(0),None


def spectral_cuda_transitions(q,lengths,rates,pi):
    return _CudaReversibleExp.apply(q,lengths,rates,pi)
