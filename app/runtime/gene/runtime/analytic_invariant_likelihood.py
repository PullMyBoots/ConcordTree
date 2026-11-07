"""Exact analytical invariant class; no rate-zero pruning/message allocation.

All variable-rate classes, original-site weights and global category mixing stay.
A zero-rate site's likelihood is topology independent: pi[state] for a single
observed state,1 for all unknown,0 for incompatible observed states.
"""
import math,time
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as ld
from concordtree.jc69_cuda import _read4
from streamed_likelihood import _leaf,_send,_join,_lse_update
from pruned_weighted_likelihood_eager import PrunedWeightedLikelihood

@triton.jit
def _invariant_values(CODES,PI,OUT,WEIGHT:tl.constexpr,W:tl.constexpr,T:tl.constexpr,LOG2:tl.constexpr=False):
    s=tl.program_id(0)*T+tl.arange(0,T);ok=s<W
    code=tl.load(CODES+s,ok,other=5)
    p=tl.where(code<4,tl.load(PI+tl.minimum(code,3)),tl.where(code==4,1.,0.))
    value=tl.where(p>0,ld.log(tl.maximum(p.to(tl.float64)*WEIGHT,1.e-300)),-1.e300)
    tl.store(OUT+s,value,ok)

@triton.jit
def _invariant_root(U,LOG,PI,LW,ROWS,OUT,WGT,IV,K:tl.constexpr,W:tl.constexpr,S:tl.constexpr,T:tl.constexpr,LOG2:tl.constexpr=False):
    s=tl.program_id(0)*T+tl.arange(0,T);ok=s<S
    r0=tl.load(ROWS);r1=tl.load(ROWS+1);r2=tl.load(ROWS+2)
    maximum=tl.load(IV+s,ok,other=-1.e300);total=tl.where(maximum>-1.e299,1.,0.).to(tl.float64)
    for k in tl.static_range(K):
        a,b,c,d=_read4(U,r0,k*W+s,K*W,ok);x,y,z,w=_read4(U,r1,k*W+s,K*W,ok)
        f,g,h,i=_read4(U,r2,k*W+s,K*W,ok)
        p=(a*x*f*tl.load(PI)+b*y*g*tl.load(PI+1)+c*z*h*tl.load(PI+2)+d*w*i*tl.load(PI+3)).to(tl.float64)
        scale=tl.load(LOG+(r0*K+k)*W+s,ok,other=0.)+tl.load(LOG+(r1*K+k)*W+s,ok,other=0.)+tl.load(LOG+(r2*K+k)*W+s,ok,other=0.)
        if LOG2:scale=scale.to(tl.float64)*tl.full((),0.6931471805599453,tl.float64)
        term=tl.where(p>0,ld.log(tl.maximum(p,1.e-300))+scale+tl.load(LW+k),-1.e300)
        maximum,total=_lse_update(maximum,total,term)
    value=maximum+ld.log(total)
    tl.store(OUT+tl.program_id(0),tl.sum(tl.where(ok,value*tl.load(WGT+s,ok,other=0).to(tl.float64),0.),0))

@triton.jit
def _invariant_nni(U,LOG,P,PI,LW,EDGES,INCOMING,OUT,WGT,IV,K:tl.constexpr,W:tl.constexpr,S:tl.constexpr,T:tl.constexpr,PARTS:tl.constexpr,LOG2:tl.constexpr=False):
    pos=tl.program_id(0);tile=tl.program_id(1);e=tl.load(EDGES+pos)
    ra=tl.load(INCOMING+pos*4);rb=tl.load(INCOMING+pos*4+1)
    rc=tl.load(INCOMING+pos*4+2);rd=tl.load(INCOMING+pos*4+3)
    s=tile*T+tl.arange(0,T);ok=s<S
    m0=tl.load(IV+s,ok,other=-1.e300);m1=m0;m2=m0
    q0=tl.where(m0>-1.e299,1.,0.).to(tl.float64);q1=q0;q2=q0
    for k in tl.static_range(K):
        a0,a1,a2,a3=_read4(U,ra,k*W+s,K*W,ok);b0,b1,b2,b3=_read4(U,rb,k*W+s,K*W,ok)
        c0,c1,c2,c3=_read4(U,rc,k*W+s,K*W,ok);d0,d1,d2,d3=_read4(U,rd,k*W+s,K*W,ok)
        p0=_join(P,PI,e,k,a0*b0,a1*b1,a2*b2,a3*b3,c0*d0,c1*d1,c2*d2,c3*d3,K)
        p1=_join(P,PI,e,k,a0*c0,a1*c1,a2*c2,a3*c3,b0*d0,b1*d1,b2*d2,b3*d3,K)
        p2=_join(P,PI,e,k,a0*d0,a1*d1,a2*d2,a3*d3,b0*c0,b1*c1,b2*c2,b3*c3,K)
        if LOG2:
            exponent=tl.load(LOG+(ra*K+k)*W+s,ok,other=0)+tl.load(LOG+(rb*K+k)*W+s,ok,other=0)
            exponent+=tl.load(LOG+(rc*K+k)*W+s,ok,other=0)+tl.load(LOG+(rd*K+k)*W+s,ok,other=0)
            scale=exponent.to(tl.float64)*tl.full((),0.6931471805599453,tl.float64)+tl.load(LW+k)
        else:
            scale=tl.load(LOG+(ra*K+k)*W+s,ok,other=0.)+tl.load(LOG+(rb*K+k)*W+s,ok,other=0.)
            scale+=tl.load(LOG+(rc*K+k)*W+s,ok,other=0.)+tl.load(LOG+(rd*K+k)*W+s,ok,other=0.)+tl.load(LW+k)
        t0=tl.where(p0>0,ld.log(tl.maximum(p0,1.e-300))+scale,-1.e300)
        t1=tl.where(p1>0,ld.log(tl.maximum(p1,1.e-300))+scale,-1.e300)
        t2=tl.where(p2>0,ld.log(tl.maximum(p2,1.e-300))+scale,-1.e300)
        m0,q0=_lse_update(m0,q0,t0);m1,q1=_lse_update(m1,q1,t1);m2,q2=_lse_update(m2,q2,t2)
    l0=m0+ld.log(q0);l1=m1+ld.log(q1);l2=m2+ld.log(q2)
    multiplicity=tl.load(WGT+s,ok,other=0).to(tl.float64)
    tl.store(OUT+(pos*3)*PARTS+tile,tl.sum(tl.where(ok,(l1-l0)*multiplicity,0.),0))
    tl.store(OUT+(pos*3+1)*PARTS+tile,tl.sum(tl.where(ok,(l2-l0)*multiplicity,0.),0))
    tl.store(OUT+(pos*3+2)*PARTS+tile,tl.sum(tl.where(ok,l0*multiplicity,0.),0))


class AnalyticInvariantLikelihood(PrunedWeightedLikelihood):
    def __init__(self,tree,names,metrics,**kw):
        rates=np.asarray(metrics['category_rates']);weights=np.asarray(metrics['category_weights'])
        zero=rates==0
        if not zero.any() or (rates<0).any():raise ValueError('positive classes plus explicit invariant class required')
        self.invariant_weight=float(weights[zero].sum())
        active=dict(metrics,category_rates=rates[~zero].tolist(),category_weights=weights[~zero].tolist())
        self.declared_workspace=kw.get('workspace_bytes',512<<20)
        # The caller sizes sites by ACTUAL retained rows. Parent old 4N estimate
        # otherwise needlessly constrains root-only checks; allocation audited.
        kw['workspace_bytes']=2*self.declared_workspace
        super().__init__(tree,names,active,**kw)
        if self.message_buffer_bytes>self.declared_workspace:raise MemoryError('actual message workspace exceeded')
        self.iv=torch.empty(self.width,device=self.device,dtype=torch.float64)
        self.codes=torch.empty(self.width,device=self.device,dtype=torch.uint8)

    def _kernels(self,slab,weights):
        k,w,parts=self.k,self.width,self.parts
        _invariant_values[(triton.cdiv(w,256),)](self.codes,self.pi,self.iv,self.invariant_weight,w,256)
        _leaf[(len(self.leafrows),k,parts)](self.u,self.logs,self.p,slab,self.leafrows,self.leafindex,self.rowedge,k,w,w,128,1e-30,enable_fp_fusion=False)
        for level in self.levels:
            _send[(len(level),k,parts)](self.u,self.logs,self.p,level,self.deps,self.rowedge,k,w,w,128,1e-30,enable_fp_fusion=False)
        _invariant_root[(parts,)](self.u,self.logs,self.pi,self.logweights,self.rootrows,self.rootpartial,weights,self.iv,k,w,w,128,enable_fp_fusion=False)
        if self.nni:
            _invariant_nni[(len(self.internal_edges),parts)](self.u,self.logs,self.p,self.pi,self.logweights,self.score_edges,self.incoming,self.partial,weights,self.iv,k,w,w,128,parts,enable_fp_fusion=False)

    @torch.no_grad()
    def score_patterns(self,alignment,*,nni=True):
        if nni!=self.nni:raise ValueError('plan mode mismatch')
        start=time.perf_counter()
        totals=torch.zeros((len(self.internal_edges),3),device=self.device,dtype=torch.float64)
        root=torch.zeros((),device=self.device,dtype=torch.float64)
        host=np.full((len(self.names),self.width),4,np.uint8);counts=np.zeros(self.width,np.int64);codes=np.full(self.width,5,np.uint8)
        slab=torch.tensor(host,device=self.device);weights=torch.tensor(counts,device=self.device)
        sites=patterns=blocks=0
        for obs,multiplicity,iv in alignment.blocks_with_invariant(self.width):
            host.fill(4);counts.fill(0);codes.fill(5)
            host[:,:obs.shape[1]]=obs;counts[:len(multiplicity)]=multiplicity;codes[:len(iv)]=iv
            slab.copy_(torch.from_numpy(host));weights.copy_(torch.from_numpy(counts));self.codes.copy_(torch.from_numpy(codes))
            self._kernels(slab,weights)
            if nni:totals+=self.partial.sum(-1)
            root+=self.rootpartial.sum();blocks+=1;sites+=int(multiplicity.sum());patterns+=obs.shape[1]
        root=float(root);output=totals.cpu().numpy()
        if sites!=alignment.original_sites:raise ValueError('lost site multiplicity')
        if not math.isfinite(root) or not np.isfinite(output).all():raise ArithmeticError('nonfinite likelihood')
        return dict(log_likelihood=root,deltas=output[:,:2] if nni else None,
            max_baseline_error=float(np.max(np.abs(output[:,2]-root))) if nni else None,
            seconds=time.perf_counter()-start,site_blocks=blocks,all_sites=sites,evaluated_patterns=patterns,
            block_sites=self.width,message_buffer_bytes=self.message_buffer_bytes,
            analytical_invariant_class=True,pruned_rate_classes=self.k)
