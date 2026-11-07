"""Exact site-multiplicity variant of retained fixed-model streamed likelihood.

Only final per-site log likelihood/delta reductions are weighted. Rate categories
still mix AFTER whole-tree pruning. Original core is untouched. Numerical tests
must compare raw sites and weights; altered summation order is not bit equivalence.
"""
import math,time
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as ld
from concordtree.jc69_cuda import _read4
from streamed_likelihood import StreamedLikelihood,_leaf,_send,_lse_update,_join

@triton.jit
def _weighted_root(U,LOG,PI,LW,ROWS,OUT,WGT,K:tl.constexpr,W:tl.constexpr,S:tl.constexpr,T:tl.constexpr,LOG2:tl.constexpr=False):
    s=tl.program_id(0)*T+tl.arange(0,T);ok=s<S
    r0=tl.load(ROWS);r1=tl.load(ROWS+1);r2=tl.load(ROWS+2)
    maximum=tl.full((T,),-1.e300,tl.float64);total=tl.full((T,),0.,tl.float64)
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
def _weighted_nni(U,LOG,P,PI,LW,EDGES,INCOMING,OUT,WGT,K:tl.constexpr,W:tl.constexpr,S:tl.constexpr,T:tl.constexpr,PARTS:tl.constexpr,LOG2:tl.constexpr=False):
    pos=tl.program_id(0);tile=tl.program_id(1);e=tl.load(EDGES+pos)
    ra=tl.load(INCOMING+pos*4);rb=tl.load(INCOMING+pos*4+1)
    rc=tl.load(INCOMING+pos*4+2);rd=tl.load(INCOMING+pos*4+3)
    s=tile*T+tl.arange(0,T);ok=s<S
    m0=tl.full((T,),-1.e300,tl.float64);m1=m0;m2=m0
    q0=tl.full((T,),0.,tl.float64);q1=q0;q2=q0
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

class WeightedLikelihood(StreamedLikelihood):
    @torch.no_grad()
    def score_patterns(self,alignment,*,nni=True):
        start=time.perf_counter()
        totals=torch.zeros((len(self.internal_edges),3),device=self.device,dtype=torch.float64)
        root_ll=torch.zeros((),device=self.device,dtype=torch.float64)
        nblocks=0;all_sites=0;pattern_count=0
        floor=1e-300 if self.dtype==torch.float64 else 1e-30
        for obs,weights in alignment.blocks(self.width):
            if obs.shape[0]!=len(self.names) or obs.shape[1]!=len(weights):raise ValueError('pattern dimensions')
            if not np.issubdtype(weights.dtype,np.integer) or (weights<=0).any():raise ValueError('positive integer multiplicity required')
            if obs.min()<0 or obs.max()>4:raise ValueError('DNA states must be0..4')
            # Stable padded slabs prevent a fresh expensive Triton specialization
            # for every data-dependent final pattern count. Padding has missing
            # observations and ZERO multiplicity; it carries no data evidence.
            observed=obs.shape[1]
            host=np.full((obs.shape[0],self.width),4,dtype=np.uint8);host[:,:observed]=obs
            multiplicity=np.zeros(self.width,dtype=np.int64);multiplicity[:observed]=weights
            slab=torch.tensor(host,device=self.device)
            w=torch.tensor(multiplicity,device=self.device)
            s=self.width;parts=triton.cdiv(s,128)
            _leaf[(len(self.leafrows),self.k,parts)](self.u,self.logs,self.p,slab,self.leafrows,self.leafindex,self.rowedge,self.k,self.width,s,128,floor,enable_fp_fusion=False)
            for level in (self.levels if nni else self.up_levels):
                _send[(len(level),self.k,parts)](self.u,self.logs,self.p,level,self.deps,self.rowedge,self.k,self.width,s,128,floor,enable_fp_fusion=False)
            _weighted_root[(parts,)](self.u,self.logs,self.pi,self.logweights,self.rootrows,self.rootpartial,w,self.k,self.width,s,128,enable_fp_fusion=False)
            if nni:
                _weighted_nni[(len(self.internal_edges),parts)](self.u,self.logs,self.p,self.pi,self.logweights,self.score_edges,self.incoming,self.partial,w,self.k,self.width,s,128,self.parts,enable_fp_fusion=False)
                totals+=self.partial[:,:,:parts].sum(-1)
            root_ll+=self.rootpartial[:parts].sum();nblocks+=1
            all_sites+=int(weights.sum());pattern_count+=observed
        root=float(root_ll);output=totals.cpu().numpy()
        if not np.isfinite(output).all() or not math.isfinite(root):raise RuntimeError('nonfinite weighted likelihood')
        if all_sites!=alignment.original_sites:raise ValueError('lost/duplicated site multiplicity')
        return dict(log_likelihood=root,deltas=output[:,:2] if nni else None,
            max_baseline_error=float(np.max(np.abs(output[:,2]-root))) if nni else None,
            seconds=time.perf_counter()-start,site_blocks=nblocks,block_sites=self.width,depth=self.depth,
            message_buffer_bytes=self.message_buffer_bytes,precision=str(self.dtype),all_sites=all_sites,evaluated_patterns=pattern_count)
