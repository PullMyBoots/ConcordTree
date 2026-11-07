"""Block-floating full-site likelihood messages: CLV * 2**integer_exponent.

Multiplying/dividing by a representable power of two is an exact binary scale
operation (outside underflow). Exponents add exactly; natural-log conversion
is needed only for category mixing, not at every internal message. The model,
sites, rate-category mixing and fixed-branch NNI alternatives are unchanged.
This is a numerical representation, not an approximate likelihood model.
"""
import torch
import triton
import triton.language as tl
from concordtree.jc69_cuda import _read4,_write4
from streamed_likelihood import _p4
from weighted_likelihood import _weighted_root,_weighted_nni
from analytic_invariant_likelihood import _invariant_values,_invariant_root,_invariant_nni
from flat_uniformized_likelihood import FlatInvariantLikelihood
from pruned_weighted_likelihood_eager import PrunedWeightedLikelihood
from replayed_likelihood import Replay


@triton.jit
def _normalize(a,b,c,d):
    norm=tl.maximum(tl.maximum(tl.maximum(a,b),tl.maximum(c,d)),1.e-30)
    # The floor is normal FP32, so IEEE exponent extraction is valid even if
    # individual zero/rare-state components are subnormal. No log/exp/divide.
    exponent=((norm.to(tl.int32,bitcast=True)>>23)&255)-126
    factor=((127-exponent)<<23).to(tl.float32,bitcast=True)
    return a*factor,b*factor,c*factor,d*factor,exponent


@triton.jit
def _binary_leaf(U,E,P,OBS,ROWS,TAXA,EDGE,M,K:tl.constexpr,W:tl.constexpr,T:tl.constexpr):
    pos=tl.program_id(0)*T+tl.arange(0,T);j=pos//W;s=pos%W;ok=j<M;k=tl.program_id(1)
    row=tl.load(ROWS+j,ok,other=0);tax=tl.load(TAXA+j,ok,other=0)
    code=tl.load(OBS+tax*W+s,ok,other=4);e=tl.load(EDGE+row,ok,other=0)
    a,b,c,d=_p4(P,e,k,((code==0)|(code==4)).to(tl.float32),
        ((code==1)|(code==4)).to(tl.float32),((code==2)|(code==4)).to(tl.float32),
        ((code==3)|(code==4)).to(tl.float32),K)
    a,b,c,d,exponent=_normalize(a,b,c,d)
    _write4(U,row,k*W+s,K*W,a,b,c,d,ok)
    tl.store(E+(row*K+k)*W+s,exponent,ok)


@triton.jit
def _binary_send(U,E,P,ROWS,DEPEND,EDGE,M,K:tl.constexpr,W:tl.constexpr,T:tl.constexpr):
    pos=tl.program_id(0)*T+tl.arange(0,T);j=pos//W;s=pos%W;ok=j<M;k=tl.program_id(1)
    row=tl.load(ROWS+j,ok,other=0);ok=ok&(row>=0);row=tl.maximum(row,0)
    r0=tl.load(DEPEND+row*2,ok,other=0);r1=tl.load(DEPEND+row*2+1,ok,other=0)
    edge=tl.load(EDGE+row,ok,other=0)
    a,b,c,d=_read4(U,r0,k*W+s,K*W,ok);x,y,z,w=_read4(U,r1,k*W+s,K*W,ok)
    a,b,c,d,e0=_normalize(a*x,b*y,c*z,d*w)
    a,b,c,d=_p4(P,edge,k,a,b,c,d,K)
    a,b,c,d,e1=_normalize(a,b,c,d)
    _write4(U,row,k*W+s,K*W,a,b,c,d,ok)
    exponent=tl.load(E+(r0*K+k)*W+s,ok,other=0)+tl.load(E+(r1*K+k)*W+s,ok,other=0)
    tl.store(E+(row*K+k)*W+s,exponent+e0+e1,ok)


class BinaryMessages:
    def __init__(self,tree,names,metrics,**kwargs):
        # Two <=100-bit normalizations per node bound accumulated exponents
        # by400*N. This assertion is storage capacity, not a scientific route.
        if len(names)*400>=2**31:raise OverflowError('int32 exponent capacity exceeded')
        super().__init__(tree,names,metrics,scale_dtype=torch.int32,**kwargs)
        self.binary_invariant=hasattr(self,'invariant_weight')

    def _kernels(self,slab,weights):
        k,w,parts=self.k,self.width,self.parts
        if self.binary_invariant:
            _invariant_values[(triton.cdiv(w,256),)](self.codes,self.pi,self.iv,self.invariant_weight,w,256)
        _binary_leaf[(triton.cdiv(len(self.leafrows)*w,128),k)](
            self.u,self.logs,self.p,slab,self.leafrows,self.leafindex,self.rowedge,len(self.leafrows),k,w,128,enable_fp_fusion=False)
        for level in self.levels:
            _binary_send[(triton.cdiv(len(level)*w,128),k)](
                self.u,self.logs,self.p,level,self.deps,self.rowedge,len(level),k,w,128,enable_fp_fusion=False)
        if self.binary_invariant:
            _invariant_root[(parts,)](self.u,self.logs,self.pi,self.logweights,self.rootrows,self.rootpartial,weights,self.iv,k,w,w,128,LOG2=True,enable_fp_fusion=False)
            if self.nni:
                _invariant_nni[(len(self.internal_edges),parts)](self.u,self.logs,self.p,self.pi,self.logweights,self.score_edges,self.incoming,self.partial,weights,self.iv,k,w,w,128,parts,LOG2=True,enable_fp_fusion=False)
        else:
            _weighted_root[(parts,)](self.u,self.logs,self.pi,self.logweights,self.rootrows,self.rootpartial,weights,k,w,w,128,LOG2=True,enable_fp_fusion=False)
            if self.nni:
                _weighted_nni[(len(self.internal_edges),parts)](self.u,self.logs,self.p,self.pi,self.logweights,self.score_edges,self.incoming,self.partial,weights,k,w,w,128,parts,LOG2=True,enable_fp_fusion=False)


class BinaryInvariantReplay(Replay,BinaryMessages,FlatInvariantLikelihood):pass
class BinaryVariableReplay(Replay,BinaryMessages,PrunedWeightedLikelihood):pass
