"""Research candidate: one observed-curvature step for each NNI central edge.

All alternatives use the same four cut messages. The endpoint score and its
observed likelihood curvature supply an affine-length step. A second
streamed pass evaluates the actual observed likelihood and retains the best of
the inherited and proposed length independently for all three topologies.
There is no reference access, candidate grid, damping parameter or memory N*L.
This is an approximate local profile, not exact scalar maximization.

The first implementation is eager for numerical validation; do not compare its
timing with the released CUDA graph scanner or promote without a full run.
"""
import math,time
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as ld
from concordtree.jc69_cuda import _read4
from streamed_likelihood import _join
from shared_mixture import SharedMixture,_common_root,_common_nni
from binary_likelihood import BinaryMessages
from flat_uniformized_likelihood import FlatInvariantLikelihood
from pruned_weighted_likelihood_eager import PrunedWeightedLikelihood


@triton.jit
def _central_stats(U,E,P,DP,FI,PI,CW,EDGES,INCOMING,OUT,LIK,WGT,IV,
                   K:tl.constexpr,W:tl.constexpr,T:tl.constexpr,PARTS:tl.constexpr,INVARIANT:tl.constexpr):
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
    g0=t0;g1=t0;g2=t0;h0=t0;h1=t0;h2=t0
    if INVARIANT:
        initial=iv*ld.exp2(tl.where(iv>0,-maximum,0).to(tl.float64));t0=initial;t1=initial;t2=initial
    for k in tl.static_range(K):
        a0,a1,a2,a3=_read4(U,ra,k*W+s,K*W,ok);b0,b1,b2,b3=_read4(U,rb,k*W+s,K*W,ok)
        c0,c1,c2,c3=_read4(U,rc,k*W+s,K*W,ok);d0,d1,d2,d3=_read4(U,rd,k*W+s,K*W,ok)
        p0=_join(P,PI,edge,k,a0*b0,a1*b1,a2*b2,a3*b3,c0*d0,c1*d1,c2*d2,c3*d3,K)
        dp0=_join(DP,PI,edge,k,a0*b0,a1*b1,a2*b2,a3*b3,c0*d0,c1*d1,c2*d2,c3*d3,K)
        fi0=_join(FI,PI,edge,k,a0*b0,a1*b1,a2*b2,a3*b3,c0*d0,c1*d1,c2*d2,c3*d3,K)
        p1=_join(P,PI,edge,k,a0*c0,a1*c1,a2*c2,a3*c3,b0*d0,b1*d1,b2*d2,b3*d3,K)
        dp1=_join(DP,PI,edge,k,a0*c0,a1*c1,a2*c2,a3*c3,b0*d0,b1*d1,b2*d2,b3*d3,K)
        fi1=_join(FI,PI,edge,k,a0*c0,a1*c1,a2*c2,a3*c3,b0*d0,b1*d1,b2*d2,b3*d3,K)
        p2=_join(P,PI,edge,k,a0*d0,a1*d1,a2*d2,a3*d3,b0*c0,b1*c1,b2*c2,b3*c3,K)
        dp2=_join(DP,PI,edge,k,a0*d0,a1*d1,a2*d2,a3*d3,b0*c0,b1*c1,b2*c2,b3*c3,K)
        fi2=_join(FI,PI,edge,k,a0*d0,a1*d1,a2*d2,a3*d3,b0*c0,b1*c1,b2*c2,b3*c3,K)
        e=tl.load(E+(ra*K+k)*W+s,ok,other=0)+tl.load(E+(rb*K+k)*W+s,ok,other=0)
        e+=tl.load(E+(rc*K+k)*W+s,ok,other=0)+tl.load(E+(rd*K+k)*W+s,ok,other=0)
        weight=tl.load(CW+k)*ld.exp2((e-maximum).to(tl.float64))
        t0+=p0*weight;g0+=dp0*weight;h0+=fi0*weight
        t1+=p1*weight;g1+=dp1*weight;h1+=fi1*weight
        t2+=p2*weight;g2+=dp2*weight;h2+=fi2*weight
    multiplicity=tl.load(WGT+s,ok,other=0).to(tl.float64)
    denominator0=tl.maximum(t0,1.e-300)
    slope0=g0/denominator0
    tl.store(OUT+((pos*3+0)*2)*PARTS+tile,tl.sum(tl.where(ok,slope0*multiplicity,0.),0))
    tl.store(OUT+((pos*3+0)*2+1)*PARTS+tile,tl.sum(tl.where(ok,(slope0*slope0-h0/denominator0)*multiplicity,0.),0))
    denominator1=tl.maximum(t1,1.e-300)
    slope1=g1/denominator1
    tl.store(OUT+((pos*3+1)*2)*PARTS+tile,tl.sum(tl.where(ok,slope1*multiplicity,0.),0))
    tl.store(OUT+((pos*3+1)*2+1)*PARTS+tile,tl.sum(tl.where(ok,(slope1*slope1-h1/denominator1)*multiplicity,0.),0))
    denominator2=tl.maximum(t2,1.e-300)
    slope2=g2/denominator2
    tl.store(OUT+((pos*3+2)*2)*PARTS+tile,tl.sum(tl.where(ok,slope2*multiplicity,0.),0))
    tl.store(OUT+((pos*3+2)*2+1)*PARTS+tile,tl.sum(tl.where(ok,(slope2*slope2-h2/denominator2)*multiplicity,0.),0))
    d1=ld.log(denominator1/denominator0);d2=ld.log(denominator2/denominator0)
    l0=ld.log(denominator0)+maximum.to(tl.float64)*tl.full((),0.6931471805599453,tl.float64)
    tl.store(LIK+(pos*3)*PARTS+tile,tl.sum(tl.where(ok,d1*multiplicity,0.),0))
    tl.store(LIK+(pos*3+1)*PARTS+tile,tl.sum(tl.where(ok,d2*multiplicity,0.),0))
    tl.store(LIK+(pos*3+2)*PARTS+tile,tl.sum(tl.where(ok,l0*multiplicity,0.),0))

@triton.jit
def _central_trial(U,E,P,NEWP,PI,CW,EDGES,INCOMING,OUT,CENTER,WGT,IV,
                   K:tl.constexpr,W:tl.constexpr,T:tl.constexpr,PARTS:tl.constexpr,INVARIANT:tl.constexpr):
    pos=tl.program_id(0);tile=tl.program_id(1);edge=tl.load(EDGES+pos)
    ra=tl.load(INCOMING+pos*4);rb=tl.load(INCOMING+pos*4+1);rc=tl.load(INCOMING+pos*4+2);rd=tl.load(INCOMING+pos*4+3)
    s=tile*T+tl.arange(0,T);ok=s<W;maximum=tl.full((T,),-2147483647,tl.int32)
    if INVARIANT:
        iv=tl.load(IV+s,ok,other=0.);maximum=tl.where(iv>0,0,maximum)
    for k in tl.static_range(K):
        e=tl.load(E+(ra*K+k)*W+s,ok,other=0)+tl.load(E+(rb*K+k)*W+s,ok,other=0)
        e+=tl.load(E+(rc*K+k)*W+s,ok,other=0)+tl.load(E+(rd*K+k)*W+s,ok,other=0)
        maximum=tl.maximum(maximum,e)
    base=tl.full((T,),0.,tl.float64);t0=base;t1=base;t2=base
    if INVARIANT:
        initial=iv*ld.exp2(tl.where(iv>0,-maximum,0).to(tl.float64));base=initial;t0=initial;t1=initial;t2=initial
    for k in tl.static_range(K):
        a0,a1,a2,a3=_read4(U,ra,k*W+s,K*W,ok);b0,b1,b2,b3=_read4(U,rb,k*W+s,K*W,ok)
        c0,c1,c2,c3=_read4(U,rc,k*W+s,K*W,ok);d0,d1,d2,d3=_read4(U,rd,k*W+s,K*W,ok)
        original=_join(P,PI,edge,k,a0*b0,a1*b1,a2*b2,a3*b3,c0*d0,c1*d1,c2*d2,c3*d3,K)
        p0=_join(NEWP,PI,pos*3,k,a0*b0,a1*b1,a2*b2,a3*b3,c0*d0,c1*d1,c2*d2,c3*d3,K)
        p1=_join(NEWP,PI,pos*3+1,k,a0*c0,a1*c1,a2*c2,a3*c3,b0*d0,b1*d1,b2*d2,b3*d3,K)
        p2=_join(NEWP,PI,pos*3+2,k,a0*d0,a1*d1,a2*d2,a3*d3,b0*c0,b1*c1,b2*c2,b3*c3,K)
        e=tl.load(E+(ra*K+k)*W+s,ok,other=0)+tl.load(E+(rb*K+k)*W+s,ok,other=0)
        e+=tl.load(E+(rc*K+k)*W+s,ok,other=0)+tl.load(E+(rd*K+k)*W+s,ok,other=0)
        weight=tl.load(CW+k)*ld.exp2((e-maximum).to(tl.float64))
        base+=original*weight;t0+=p0*weight;t1+=p1*weight;t2+=p2*weight
    base=tl.maximum(base,1.e-300);mult=tl.load(WGT+s,ok,other=0).to(tl.float64)
    d0=ld.log(tl.maximum(t0,1.e-300)/base);d1=ld.log(tl.maximum(t1,1.e-300)/base);d2=ld.log(tl.maximum(t2,1.e-300)/base)
    baseline=ld.log(base)+maximum.to(tl.float64)*tl.full((),0.6931471805599453,tl.float64)
    tl.store(OUT+(pos*3)*PARTS+tile,tl.sum(tl.where(ok,d1*mult,0.),0))
    tl.store(OUT+(pos*3+1)*PARTS+tile,tl.sum(tl.where(ok,d2*mult,0.),0))
    tl.store(OUT+(pos*3+2)*PARTS+tile,tl.sum(tl.where(ok,baseline*mult,0.),0))
    tl.store(CENTER+pos*PARTS+tile,tl.sum(tl.where(ok,d0*mult,0.),0))


class CentralStep:
    def __init__(self,tree,names,metrics,**kwargs):
        super().__init__(tree,names,metrics,**kwargs)
        self.phase=0;self.metrics=metrics
        if not self.nni:raise ValueError('central profile requires NNI messages')
        q=torch.tensor(metrics['Q_acgt'],device=self.device,dtype=torch.float64)
        rates=torch.tensor([r for r in metrics['category_rates'] if r>0],device=self.device,dtype=torch.float64)
        self.dp=((q[None]*rates[:,None,None])[None]@self.p.double()).float()
        self.fi=(((q[None]*rates[:,None,None])[None])@self.dp.double()).float()
        self.stat_partial=torch.empty((len(self.internal_edges),3,2,self.parts),device=self.device,dtype=torch.float64)
        self.center_partial=torch.empty((len(self.internal_edges),self.parts),device=self.device,dtype=torch.float64)

    def _mixture(self,weights):
        if self.phase==0:
            iv=self.iv if self.binary_invariant else self.rootpartial
            _common_root[(self.parts,)](self.u,self.logs,self.pi,self.category_weights,self.rootrows,self.rootpartial,weights,iv,
                self.k,self.width,128,self.binary_invariant,enable_fp_fusion=False)
            _central_stats[(len(self.internal_edges),self.parts)](self.u,self.logs,self.p,self.dp,self.fi,self.pi,self.category_weights,
                self.score_edges,self.incoming,self.stat_partial,self.partial,weights,iv,self.k,self.width,128,self.parts,self.binary_invariant,enable_fp_fusion=False)
        else:
            iv=self.iv if self.binary_invariant else self.rootpartial
            _common_root[(self.parts,)](self.u,self.logs,self.pi,self.category_weights,self.rootrows,self.rootpartial,weights,iv,
                self.k,self.width,128,self.binary_invariant,enable_fp_fusion=False)
            _central_trial[(len(self.internal_edges),self.parts)](self.u,self.logs,self.p,self.new_p,self.pi,self.category_weights,
                self.score_edges,self.incoming,self.partial,self.center_partial,weights,iv,
                self.k,self.width,128,self.parts,self.binary_invariant,enable_fp_fusion=False)

    def _kernels(self,slab,weights):
        super()._kernels(slab,weights)
        if self.phase==0:self.stat_totals+=self.stat_partial.sum(-1)
        else:self.center_totals+=self.center_partial.sum(-1)

    @torch.no_grad()
    def score_patterns(self,alignment,*,nni=True):
        start=time.perf_counter();n=len(self.internal_edges);self.phase=0
        self.stat_totals=torch.zeros((n,3,2),device=self.device,dtype=torch.float64)
        frozen=super().score_patterns(alignment,nni=nni)
        stats=self.stat_totals.cpu().numpy();old=np.array([self.tree.lengths[e] for e in self.internal_edges])[:,None]
        delta=np.divide(stats[:,:,0],np.abs(stats[:,:,1]),out=np.zeros((n,3)),where=np.abs(stats[:,:,1])>0)
        variable_mass=1-self.metrics.get('invariant_proportion',0.)
        self.new_lengths=np.clip(old+delta,variable_mass*1e-12,variable_mass*200)
        from fixed_transitions import fixed_transitions
        self.new_p=fixed_transitions(self.metrics['Q_acgt'],self.metrics['frequencies_acgt'],
            self.new_lengths.reshape(-1),[r for r in self.metrics['category_rates'] if r>0],self.device).float()
        self.phase=1;self.center_totals=torch.zeros(n,device=self.device,dtype=torch.float64)
        trial=super().score_patterns(alignment,nni=nni)
        use_new=trial['deltas']>frozen['deltas'];best=np.maximum(trial['deltas'],frozen['deltas'])
        self.proposed_lengths=np.where(use_new,self.new_lengths[:,1:],old)
        self.profiled_baseline=np.maximum(self.center_totals.cpu().numpy(),0.)
        self.last_stats=stats
        result=dict(trial,deltas=best,seconds=time.perf_counter()-start,observed_curvature_step=True,
            approximate_profile=True,old_deltas=frozen['deltas'],new_deltas=trial['deltas'],
            profiled_baseline_gain=self.profiled_baseline,proposed_lengths=self.proposed_lengths)
        return result

    def proposals(self,result,min_gain=1e-4):
        out=[]
        for j,(e,branches,delta) in enumerate(zip(self.internal_edges,self.branches,result['deltas'])):
            alt=int(np.argmax(delta));gain=float(delta[alt]);local=gain-float(self.profiled_baseline[j])
            if local>min_gain:
                out.append(dict(edge=e,branches=branches,alternative=alt+1,length=float(self.proposed_lengths[j,alt]),
                    gain=gain,local_gain=local,log_likelihood=result['log_likelihood']+gain))
        return sorted(out,key=lambda p:(-p['gain'],p['edge']))

class CurvatureInvariant(CentralStep,SharedMixture,BinaryMessages,FlatInvariantLikelihood):pass
class CurvatureVariable(CentralStep,SharedMixture,BinaryMessages,PrunedWeightedLikelihood):pass
