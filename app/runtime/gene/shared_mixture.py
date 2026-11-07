"""One common category scale for all three NNI alternatives.

With binary incoming exponents e_k and conditional join probabilities p_jk,
L_j=2**M * sum_k w_k p_jk 2**(e_k-M). M is shared across j=0,1,2.
The likelihood ratio cancels 2**M analytically. This avoids per-alternative
log-space mixtures, without deleting sites/categories or approximating a rate.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as ld
from concordtree.jc69_cuda import _read4
from streamed_likelihood import _join
from binary_likelihood import _binary_leaf,_binary_send,BinaryMessages
from replayed_likelihood import Replay
from flat_uniformized_likelihood import FlatInvariantLikelihood
from pruned_weighted_likelihood_eager import PrunedWeightedLikelihood

@triton.jit
def _linear_invariant(CODES,PI,OUT,WEIGHT,W:tl.constexpr,T:tl.constexpr):
    s=tl.program_id(0)*T+tl.arange(0,T);ok=s<W;code=tl.load(CODES+s,ok,other=5)
    p=tl.where(code<4,tl.load(PI+tl.minimum(code,3)),tl.where(code==4,1.,0.)).to(tl.float64)
    tl.store(OUT+s,p*tl.load(WEIGHT),ok)

@triton.jit
def _common_root(U,E,PI,CW,ROWS,OUT,WGT,IV,K:tl.constexpr,W:tl.constexpr,T:tl.constexpr,INVARIANT:tl.constexpr):
    s=tl.program_id(0)*T+tl.arange(0,T);ok=s<W
    r0=tl.load(ROWS);r1=tl.load(ROWS+1);r2=tl.load(ROWS+2)
    maximum=tl.full((T,),-2147483647,tl.int32)
    if INVARIANT:
        iv=tl.load(IV+s,ok,other=0.);maximum=tl.where(iv>0,0,maximum)
    for k in tl.static_range(K):
        e=tl.load(E+(r0*K+k)*W+s,ok,other=0)+tl.load(E+(r1*K+k)*W+s,ok,other=0)+tl.load(E+(r2*K+k)*W+s,ok,other=0)
        maximum=tl.maximum(maximum,e)
    total=tl.full((T,),0.,tl.float64)
    if INVARIANT:total=iv*ld.exp2(tl.where(iv>0,-maximum,0).to(tl.float64))
    for k in tl.static_range(K):
        a,b,c,d=_read4(U,r0,k*W+s,K*W,ok);x,y,z,w=_read4(U,r1,k*W+s,K*W,ok)
        f,g,h,i=_read4(U,r2,k*W+s,K*W,ok)
        p=(a*x*f*tl.load(PI)+b*y*g*tl.load(PI+1)+c*z*h*tl.load(PI+2)+d*w*i*tl.load(PI+3)).to(tl.float64)
        e=tl.load(E+(r0*K+k)*W+s,ok,other=0)+tl.load(E+(r1*K+k)*W+s,ok,other=0)+tl.load(E+(r2*K+k)*W+s,ok,other=0)
        total+=p*tl.load(CW+k)*ld.exp2((e-maximum).to(tl.float64))
    value=ld.log(tl.maximum(total,1.e-300))+maximum.to(tl.float64)*tl.full((),0.6931471805599453,tl.float64)
    tl.store(OUT+tl.program_id(0),tl.sum(tl.where(ok,value*tl.load(WGT+s,ok,other=0).to(tl.float64),0.),0))

@triton.jit
def _common_nni(U,E,P,PI,CW,EDGES,INCOMING,OUT,WGT,IV,K:tl.constexpr,W:tl.constexpr,T:tl.constexpr,PARTS:tl.constexpr,INVARIANT:tl.constexpr):
    pos=tl.program_id(0);tile=tl.program_id(1);edge=tl.load(EDGES+pos)
    ra=tl.load(INCOMING+pos*4);rb=tl.load(INCOMING+pos*4+1);rc=tl.load(INCOMING+pos*4+2);rd=tl.load(INCOMING+pos*4+3)
    s=tile*T+tl.arange(0,T);ok=s<W;maximum=tl.full((T,),-2147483647,tl.int32)
    if INVARIANT:
        iv=tl.load(IV+s,ok,other=0.);maximum=tl.where(iv>0,0,maximum)
    for k in tl.static_range(K):
        e=tl.load(E+(ra*K+k)*W+s,ok,other=0)+tl.load(E+(rb*K+k)*W+s,ok,other=0)
        e+=tl.load(E+(rc*K+k)*W+s,ok,other=0)+tl.load(E+(rd*K+k)*W+s,ok,other=0)
        maximum=tl.maximum(maximum,e)
    t0=tl.full((T,),0.,tl.float64);t1=t0;t2=t0
    if INVARIANT:
        initial=iv*ld.exp2(tl.where(iv>0,-maximum,0).to(tl.float64));t0=initial;t1=initial;t2=initial
    for k in tl.static_range(K):
        a0,a1,a2,a3=_read4(U,ra,k*W+s,K*W,ok);b0,b1,b2,b3=_read4(U,rb,k*W+s,K*W,ok)
        c0,c1,c2,c3=_read4(U,rc,k*W+s,K*W,ok);d0,d1,d2,d3=_read4(U,rd,k*W+s,K*W,ok)
        p0=_join(P,PI,edge,k,a0*b0,a1*b1,a2*b2,a3*b3,c0*d0,c1*d1,c2*d2,c3*d3,K)
        p1=_join(P,PI,edge,k,a0*c0,a1*c1,a2*c2,a3*c3,b0*d0,b1*d1,b2*d2,b3*d3,K)
        p2=_join(P,PI,edge,k,a0*d0,a1*d1,a2*d2,a3*d3,b0*c0,b1*c1,b2*c2,b3*c3,K)
        e=tl.load(E+(ra*K+k)*W+s,ok,other=0)+tl.load(E+(rb*K+k)*W+s,ok,other=0)
        e+=tl.load(E+(rc*K+k)*W+s,ok,other=0)+tl.load(E+(rd*K+k)*W+s,ok,other=0)
        weight=tl.load(CW+k)*ld.exp2((e-maximum).to(tl.float64))
        t0+=p0*weight;t1+=p1*weight;t2+=p2*weight
    baseline=tl.maximum(t0,1.e-300)
    d1=ld.log(tl.maximum(t1,1.e-300)/baseline);d2=ld.log(tl.maximum(t2,1.e-300)/baseline)
    l0=ld.log(baseline)+maximum.to(tl.float64)*tl.full((),0.6931471805599453,tl.float64)
    multiplicity=tl.load(WGT+s,ok,other=0).to(tl.float64)
    tl.store(OUT+(pos*3)*PARTS+tile,tl.sum(tl.where(ok,d1*multiplicity,0.),0))
    tl.store(OUT+(pos*3+1)*PARTS+tile,tl.sum(tl.where(ok,d2*multiplicity,0.),0))
    tl.store(OUT+(pos*3+2)*PARTS+tile,tl.sum(tl.where(ok,l0*multiplicity,0.),0))

class SharedMixture:
    def __init__(self,tree,names,metrics,**kwargs):
        super().__init__(tree,names,metrics,**kwargs)
        weights=[w for r,w in zip(metrics['category_rates'],metrics['category_weights']) if r>0]
        self.category_weights=torch.tensor(weights+[getattr(self,'invariant_weight',0.)],device=self.device,dtype=torch.float64)
        self.invariant_weight_tensor=self.category_weights[-1:]

    def _kernels(self,slab,weights):
        k,w=self.k,self.width
        if self.binary_invariant:
            _linear_invariant[(triton.cdiv(w,256),)](self.codes,self.pi,self.iv,self.invariant_weight_tensor,w,256)
        _binary_leaf[(triton.cdiv(len(self.leafrows)*w,128),k)](self.u,self.logs,self.p,slab,self.leafrows,self.leafindex,self.rowedge,len(self.leafrows),k,w,128,enable_fp_fusion=False)
        for level in self.levels:
            _binary_send[(triton.cdiv(len(level)*w,128),k)](self.u,self.logs,self.p,level,self.deps,self.rowedge,len(level),k,w,128,enable_fp_fusion=False)
        self._mixture(weights)

    def _mixture(self,weights):
        k,w,parts=self.k,self.width,self.parts;invariant=self.binary_invariant
        iv=self.iv if invariant else self.rootpartial
        _common_root[(parts,)](self.u,self.logs,self.pi,self.category_weights,self.rootrows,self.rootpartial,weights,iv,k,w,128,invariant,enable_fp_fusion=False)
        if self.nni:
            _common_nni[(len(self.internal_edges),parts)](self.u,self.logs,self.p,self.pi,self.category_weights,self.score_edges,self.incoming,self.partial,weights,iv,k,w,128,parts,invariant,enable_fp_fusion=False)

class SharedInvariantReplay(Replay,SharedMixture,BinaryMessages,FlatInvariantLikelihood):pass
class SharedVariableReplay(Replay,SharedMixture,BinaryMessages,PrunedWeightedLikelihood):pass
