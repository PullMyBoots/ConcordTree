"""One-pass uniformization-EM lower-bound NNI/branch proposals.

For lambda=max(-diag(Q)), latent Poisson count C_e=t_e*(g_e+lambda*R),
R=sum_sites E[rate|data]. EM proposes t'=C_e/(lambda*R). Its complete-data
Q gain lower-bounds observed LL gain on that conditional topology. No second
scan per edge, Hessian, series expansion, neural model or site subsampling.
Whole-tree batch LL verification is still mandatory; floating arithmetic and
simultaneous topology moves do not inherit a blanket exact-arithmetic guarantee.
"""
import time,math
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as ld
from concordtree.jc69_cuda import _read4
from concordtree.treeio import edge
from streamed_likelihood import _leaf,_send,_join
from analytic_invariant_likelihood import AnalyticInvariantLikelihood,_invariant_root,_invariant_values

@triton.jit
def _em_nni(U,LOG,P,DP,PI,LW,RATES,INC,EDGES,OUT,WGT,IV,
            K:tl.constexpr,W:tl.constexpr,T:tl.constexpr,PARTS:tl.constexpr):
    pos=tl.program_id(0);alt=tl.program_id(1);tile=tl.program_id(2);e=tl.load(EDGES+pos)
    ra=tl.load(INC+pos*4);rb=tl.load(INC+pos*4+1);rc=tl.load(INC+pos*4+2);rd=tl.load(INC+pos*4+3)
    s=tile*T+tl.arange(0,T);ok=s<W
    m=tl.load(IV+s,ok,other=-1.e300);q=tl.where(m>-1.e299,1.,0.).to(tl.float64)
    g=tl.full((T,),0.,tl.float64);rate=g
    for k in tl.static_range(K):
        a0,a1,a2,a3=_read4(U,ra,k*W+s,K*W,ok);b0,b1,b2,b3=_read4(U,rb,k*W+s,K*W,ok)
        c0,c1,c2,c3=_read4(U,rc,k*W+s,K*W,ok);d0,d1,d2,d3=_read4(U,rd,k*W+s,K*W,ok)
        l0=tl.where(alt==0,a0*b0,tl.where(alt==1,a0*c0,a0*d0));l1=tl.where(alt==0,a1*b1,tl.where(alt==1,a1*c1,a1*d1))
        l2=tl.where(alt==0,a2*b2,tl.where(alt==1,a2*c2,a2*d2));l3=tl.where(alt==0,a3*b3,tl.where(alt==1,a3*c3,a3*d3))
        r0=tl.where(alt==0,c0*d0,tl.where(alt==1,b0*d0,b0*c0));r1=tl.where(alt==0,c1*d1,tl.where(alt==1,b1*d1,b1*c1))
        r2=tl.where(alt==0,c2*d2,tl.where(alt==1,b2*d2,b2*c2));r3=tl.where(alt==0,c3*d3,tl.where(alt==1,b3*d3,b3*c3))
        p=_join(P,PI,e,k,l0,l1,l2,l3,r0,r1,r2,r3,K);dp=_join(DP,PI,e,k,l0,l1,l2,l3,r0,r1,r2,r3,K)
        scale=tl.load(LOG+(ra*K+k)*W+s,ok,other=0.)+tl.load(LOG+(rb*K+k)*W+s,ok,other=0.)
        scale+=tl.load(LOG+(rc*K+k)*W+s,ok,other=0.)+tl.load(LOG+(rd*K+k)*W+s,ok,other=0.)+tl.load(LW+k)
        term=tl.where(p>0,ld.log(tl.maximum(p,1.e-300))+scale,-1.e300)
        new=tl.maximum(m,term);prev=ld.exp(m-new);weight=ld.exp(term-new)
        q=q*prev+weight;g=g*prev+weight*tl.where(p>0,dp/tl.maximum(p,1.e-300),0.)
        rate=rate*prev+weight*tl.load(RATES+k);m=new
    multiplicity=tl.load(WGT+s,ok,other=0).to(tl.float64);base=((pos*3+alt)*3)*PARTS+tile
    tl.store(OUT+base,tl.sum(tl.where(ok,(m+ld.log(q))*multiplicity,0.),0))
    tl.store(OUT+base+PARTS,tl.sum(tl.where(ok,g/q*multiplicity,0.),0))
    tl.store(OUT+base+2*PARTS,tl.sum(tl.where(ok,rate/q*multiplicity,0.),0))

@triton.jit
def _em_leaf(U,LOG,P,DP,PI,LW,RATES,OBS,LEAVES,INDEX,ROWEDGE,INC,OUT,WGT,IV,
             K:tl.constexpr,W:tl.constexpr,T:tl.constexpr,PARTS:tl.constexpr):
    pos=tl.program_id(0);tile=tl.program_id(1);ra=tl.load(INC+pos*2);rb=tl.load(INC+pos*2+1)
    e=tl.load(ROWEDGE+tl.load(LEAVES+pos));tax=tl.load(INDEX+pos)
    s=tile*T+tl.arange(0,T);ok=s<W;code=tl.load(OBS+tax*W+s,ok,other=4)
    c0=((code==0)|(code==4)).to(tl.float32);c1=((code==1)|(code==4)).to(tl.float32)
    c2=((code==2)|(code==4)).to(tl.float32);c3=((code==3)|(code==4)).to(tl.float32)
    m=tl.load(IV+s,ok,other=-1.e300);q=tl.where(m>-1.e299,1.,0.).to(tl.float64)
    g=tl.full((T,),0.,tl.float64);rate=g
    for k in tl.static_range(K):
        a0,a1,a2,a3=_read4(U,ra,k*W+s,K*W,ok);b0,b1,b2,b3=_read4(U,rb,k*W+s,K*W,ok)
        p=_join(P,PI,e,k,a0*b0,a1*b1,a2*b2,a3*b3,c0,c1,c2,c3,K)
        dp=_join(DP,PI,e,k,a0*b0,a1*b1,a2*b2,a3*b3,c0,c1,c2,c3,K)
        scale=tl.load(LOG+(ra*K+k)*W+s,ok,other=0.)+tl.load(LOG+(rb*K+k)*W+s,ok,other=0.)+tl.load(LW+k)
        term=tl.where(p>0,ld.log(tl.maximum(p,1.e-300))+scale,-1.e300)
        new=tl.maximum(m,term);prev=ld.exp(m-new);weight=ld.exp(term-new)
        q=q*prev+weight;g=g*prev+weight*tl.where(p>0,dp/tl.maximum(p,1.e-300),0.)
        rate=rate*prev+weight*tl.load(RATES+k);m=new
    multiplicity=tl.load(WGT+s,ok,other=0).to(tl.float64)
    tl.store(OUT+(pos*3)*PARTS+tile,tl.sum(tl.where(ok,(m+ld.log(q))*multiplicity,0.),0))
    tl.store(OUT+(pos*3+1)*PARTS+tile,tl.sum(tl.where(ok,g/q*multiplicity,0.),0))
    tl.store(OUT+(pos*3+2)*PARTS+tile,tl.sum(tl.where(ok,rate/q*multiplicity,0.),0))


class UniformizedLikelihood(AnalyticInvariantLikelihood):
    def __init__(self,tree,names,metrics,**kw):
        if not kw.get('nni',True):raise ValueError('use analytical parent for root checks')
        super().__init__(tree,names,metrics,**kw)
        rates=np.asarray(metrics['category_rates']);rates=rates[rates>0]
        self.rates=torch.tensor(rates,device=self.device,dtype=torch.float64)
        q=torch.tensor(metrics['Q_acgt'],device=self.device,dtype=torch.float32)
        self.uniformization_rate=float(-np.diag(np.array(metrics['Q_acgt'])).min())
        self.dp=(q[None]*self.rates.float()[:,None,None])[None]@self.p
        rows={}
        for u,v in self.edges:
            for a,b in ((u,v),(v,u)):
                if b not in tree.names:rows[a,b]=len(rows)
        leaves=sorted(tree.names);incoming=[]
        for u in leaves:
            v=next(iter(tree.graph[u]));incoming.append([rows[w,v] for w in sorted(tree.graph[v]-{u})])
        self.leafincoming=torch.tensor(incoming,device=self.device,dtype=torch.int32)
        self.partial=torch.empty((len(self.internal_edges),3,3,self.parts),device=self.device,dtype=torch.float64)
        self.leafpartial=torch.empty((len(leaves),3,self.parts),device=self.device,dtype=torch.float64)
        self.leafedges=[edge(u,next(iter(tree.graph[u]))) for u in leaves]

    def _kernels(self,slab,weights):
        k,w,parts=self.k,self.width,self.parts
        _invariant_values[(triton.cdiv(w,256),)](self.codes,self.pi,self.iv,self.invariant_weight,w,256)
        _leaf[(len(self.leafrows),k,parts)](self.u,self.logs,self.p,slab,self.leafrows,self.leafindex,self.rowedge,k,w,w,128,1e-30,enable_fp_fusion=False)
        for level in self.levels:
            _send[(len(level),k,parts)](self.u,self.logs,self.p,level,self.deps,self.rowedge,k,w,w,128,1e-30,enable_fp_fusion=False)
        _invariant_root[(parts,)](self.u,self.logs,self.pi,self.logweights,self.rootrows,self.rootpartial,weights,self.iv,k,w,w,128,enable_fp_fusion=False)
        _em_nni[(len(self.internal_edges),3,parts)](self.u,self.logs,self.p,self.dp,self.pi,self.logweights,self.rates,self.incoming,self.score_edges,self.partial,weights,self.iv,k,w,128,parts,enable_fp_fusion=False)
        _em_leaf[(len(self.leafrows),parts)](self.u,self.logs,self.p,self.dp,self.pi,self.logweights,self.rates,slab,self.leafrows,self.leafindex,self.rowedge,self.leafincoming,self.leafpartial,weights,self.iv,k,w,128,parts,enable_fp_fusion=False)

    @torch.no_grad()
    def score_patterns(self,alignment,*,nni=True):
        start=time.perf_counter()
        totals=torch.zeros((len(self.internal_edges),3,3),device=self.device,dtype=torch.float64)
        leaf_totals=torch.zeros((len(self.leafrows),3),device=self.device,dtype=torch.float64)
        root=torch.zeros((),device=self.device,dtype=torch.float64)
        host=np.full((len(self.names),self.width),4,np.uint8);counts=np.zeros(self.width,np.int64);codes=np.full(self.width,5,np.uint8)
        slab=torch.tensor(host,device=self.device);weights=torch.tensor(counts,device=self.device)
        sites=patterns=blocks=0
        for obs,multiplicity,iv in alignment.blocks_with_invariant(self.width):
            host.fill(4);counts.fill(0);codes.fill(5);host[:,:obs.shape[1]]=obs;counts[:len(multiplicity)]=multiplicity;codes[:len(iv)]=iv
            slab.copy_(torch.from_numpy(host));weights.copy_(torch.from_numpy(counts));self.codes.copy_(torch.from_numpy(codes))
            self._kernels(slab,weights);totals+=self.partial.sum(-1);leaf_totals+=self.leafpartial.sum(-1)
            root+=self.rootpartial.sum();blocks+=1;sites+=int(multiplicity.sum());patterns+=obs.shape[1]
        root=float(root);output=totals.cpu().numpy();leaves=leaf_totals.cpu().numpy()
        if sites!=alignment.original_sites:raise ValueError('lost site multiplicity')
        if not math.isfinite(root) or not np.isfinite(output).all() or not np.isfinite(leaves).all():raise ArithmeticError('nonfinite LL/score/rate posterior')
        lengths=np.array([self.tree.lengths[e] for e in self.internal_edges])[:,None]
        denominator=self.uniformization_rate*output[:,:,2]
        latent=lengths*(output[:,:,1]+denominator)
        if latent.min() < -max(1e-4,1e-6*sites):raise ArithmeticError('negative expected uniformization count')
        latent=np.maximum(latent,0.)
        self.central_lengths=np.clip(latent/denominator,1e-6,10.)
        qgain=latent*np.log(self.central_lengths/lengths)-denominator*(self.central_lengths-lengths)
        lower=output[:,:,0]+np.maximum(qgain,0.)
        byedge=dict(zip(self.internal_edges,output[:,0,1:]));byedge.update(zip(self.leafedges,leaves[:,1:3]))
        stats=np.array([byedge[e] for e in self.edges]);old=np.array([self.tree.lengths[e] for e in self.edges])
        den=self.uniformization_rate*stats[:,1];count=old*(stats[:,0]+den)
        if count.min() < -max(1e-4,1e-6*sites):raise ArithmeticError('negative all-branch expected count')
        proposed=np.clip(np.maximum(count,0.)/den,1e-6,10.)
        rate_error=max(float(np.ptp(output[:,0,2])),float(np.max(np.abs(leaves[:,2]-output[0,0,2]))))
        if rate_error>1e-5*sites:raise ArithmeticError('category posterior depends on root edge')
        return dict(log_likelihood=root,deltas=lower[:,1:]-root,lower_scores=lower,conditional_scores=output[:,:,0],
            conditional_gradient=output[:,:,1],conditional_rate_sum=output[:,:,2],
            max_baseline_error=max(float(np.max(np.abs(output[:,0,0]-root))),float(np.max(np.abs(leaves[:,0]-root)))),
            branch_edges=self.edges,branch_gradient=stats[:,0],branch_rate_sum=stats[:,1],branch_step=proposed-old,
            seconds=time.perf_counter()-start,site_blocks=blocks,all_sites=sites,evaluated_patterns=patterns,
            block_sites=self.width,message_buffer_bytes=self.message_buffer_bytes,uniformization_rate=self.uniformization_rate,
            posterior_rate_root_error=rate_error,minimum_unclipped_expected_count=float(min(count.min(),latent.min())))

    def proposals(self,result,min_gain=1e-4):
        out=[]
        for i,(e,b,scores) in enumerate(zip(self.internal_edges,self.branches,result['lower_scores'])):
            alt=int(np.argmax(scores));gain=float(scores[alt]-result['log_likelihood'])
            if alt and gain>min_gain:
                out.append(dict(edge=e,branches=b,alternative=alt,length=float(self.central_lengths[i,alt]),gain=gain,
                    log_likelihood_lower_bound=float(scores[alt]),local_lower_bound_gain=float(scores[alt]-scores[0]),
                    method='one uniformization-EM central-length update; observed-LL lower bound'))
        return sorted(out,key=lambda p:(-p['gain'],p['edge']))
