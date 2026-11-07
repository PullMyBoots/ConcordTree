"""Experimental all-taxon, all-site block-streamed fixed-model NNI likelihood.

NOT a complete de novo estimator: caller supplies topology, positive lengths and
valid reversible-model parameters. Shared categories are mixed AFTER whole-tree
pruning. O(N*K*L) work per scan, O(N*K*B) CLV state; depth-batched launch span.
No neural model, site subsampling, global distance matrix or per-edge tree fit.
"""
from __future__ import annotations
import math,time
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as ld
from concordtree.treeio import edge,rooted
from concordtree.dna_models import transition_matrices
from concordtree.jc69_cuda import _read4,_write4

@triton.jit
def _p4(P,e,k,a,b,c,d,K:tl.constexpr):
    o=(e*K+k)*16
    z0=tl.load(P+o)*a+tl.load(P+o+1)*b+tl.load(P+o+2)*c+tl.load(P+o+3)*d
    z1=tl.load(P+o+4)*a+tl.load(P+o+5)*b+tl.load(P+o+6)*c+tl.load(P+o+7)*d
    z2=tl.load(P+o+8)*a+tl.load(P+o+9)*b+tl.load(P+o+10)*c+tl.load(P+o+11)*d
    z3=tl.load(P+o+12)*a+tl.load(P+o+13)*b+tl.load(P+o+14)*c+tl.load(P+o+15)*d
    return z0,z1,z2,z3

@triton.jit
def _leaf(U,LOG,P,OBS,LEAF_ROWS,LEAF_INDEX,ROW_EDGE,K:tl.constexpr,W:tl.constexpr,S:tl.constexpr,T:tl.constexpr,FLOOR:tl.constexpr):
    row=tl.load(LEAF_ROWS+tl.program_id(0));tax=tl.load(LEAF_INDEX+tl.program_id(0))
    k=tl.program_id(1);s=tl.program_id(2)*T+tl.arange(0,T);ok=s<S
    code=tl.load(OBS+tax*S+s,ok,other=4);e=tl.load(ROW_EDGE+row)
    v0,v1,v2,v3=_p4(P,e,k,((code==0)|(code==4)).to(U.dtype.element_ty),
        ((code==1)|(code==4)).to(U.dtype.element_ty),((code==2)|(code==4)).to(U.dtype.element_ty),
        ((code==3)|(code==4)).to(U.dtype.element_ty),K)
    norm=tl.maximum(tl.maximum(tl.maximum(v0,v1),tl.maximum(v2,v3)),FLOOR)
    _write4(U,row,k*W+s,K*W,v0/norm,v1/norm,v2/norm,v3/norm,ok)
    tl.store(LOG+(row*K+k)*W+s,ld.log(norm.to(tl.float64)),ok)

@triton.jit
def _send(U,LOG,P,ROWS,DEPEND,ROW_EDGE,K:tl.constexpr,W:tl.constexpr,S:tl.constexpr,T:tl.constexpr,FLOOR:tl.constexpr):
    row=tl.load(ROWS+tl.program_id(0));r0=tl.load(DEPEND+row*2);r1=tl.load(DEPEND+row*2+1)
    e=tl.load(ROW_EDGE+row);k=tl.program_id(1)
    s=tl.program_id(2)*T+tl.arange(0,T);ok=s<S
    a,b,c,d=_read4(U,r0,k*W+s,K*W,ok);x,y,z,w=_read4(U,r1,k*W+s,K*W,ok)
    a*=x;b*=y;c*=z;d*=w
    norm=tl.maximum(tl.maximum(tl.maximum(a,b),tl.maximum(c,d)),FLOOR)
    x,y,z,w=_p4(P,e,k,a/norm,b/norm,c/norm,d/norm,K)
    norm2=tl.maximum(tl.maximum(tl.maximum(x,y),tl.maximum(z,w)),FLOOR)
    _write4(U,row,k*W+s,K*W,x/norm2,y/norm2,z/norm2,w/norm2,ok)
    scale=tl.load(LOG+(r0*K+k)*W+s,ok,other=0.)+tl.load(LOG+(r1*K+k)*W+s,ok,other=0.)
    tl.store(LOG+(row*K+k)*W+s,scale+ld.log(norm.to(tl.float64))+ld.log(norm2.to(tl.float64)),ok)

@triton.jit
def _lse_update(maximum,total,term):
    new=tl.maximum(maximum,term)
    total=total*ld.exp(maximum-new)+ld.exp(term-new)
    return new,total

@triton.jit
def _root(U,LOG,PI,LW,ROWS,OUT,K:tl.constexpr,W:tl.constexpr,S:tl.constexpr,T:tl.constexpr):
    s=tl.program_id(0)*T+tl.arange(0,T);ok=s<S
    r0=tl.load(ROWS);r1=tl.load(ROWS+1);r2=tl.load(ROWS+2)
    maximum=tl.full((T,),-1.e300,tl.float64);total=tl.full((T,),0.,tl.float64)
    for k in tl.static_range(K):
        a,b,c,d=_read4(U,r0,k*W+s,K*W,ok);x,y,z,w=_read4(U,r1,k*W+s,K*W,ok)
        f,g,h,i=_read4(U,r2,k*W+s,K*W,ok)
        p=(a*x*f*tl.load(PI)+b*y*g*tl.load(PI+1)+c*z*h*tl.load(PI+2)+d*w*i*tl.load(PI+3)).to(tl.float64)
        scale=tl.load(LOG+(r0*K+k)*W+s,ok,other=0.)+tl.load(LOG+(r1*K+k)*W+s,ok,other=0.)+tl.load(LOG+(r2*K+k)*W+s,ok,other=0.)
        term=tl.where(p>0,ld.log(tl.maximum(p,1.e-300))+scale+tl.load(LW+k),-1.e300)
        maximum,total=_lse_update(maximum,total,term)
    value=maximum+ld.log(total)
    tl.store(OUT+tl.program_id(0),tl.sum(tl.where(ok,value,0.),0))

@triton.jit
def _join(P,PI,e,k,a,b,c,d,x,y,z,w,K:tl.constexpr):
    x,y,z,w=_p4(P,e,k,x,y,z,w,K)
    return (a*x*tl.load(PI)+b*y*tl.load(PI+1)+c*z*tl.load(PI+2)+d*w*tl.load(PI+3)).to(tl.float64)

@triton.jit
def _nni(U,LOG,P,PI,LW,EDGES,INCOMING,OUT,K:tl.constexpr,W:tl.constexpr,S:tl.constexpr,T:tl.constexpr,PARTS:tl.constexpr):
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
        scale=tl.load(LOG+(ra*K+k)*W+s,ok,other=0.)+tl.load(LOG+(rb*K+k)*W+s,ok,other=0.)
        scale+=tl.load(LOG+(rc*K+k)*W+s,ok,other=0.)+tl.load(LOG+(rd*K+k)*W+s,ok,other=0.)+tl.load(LW+k)
        t0=tl.where(p0>0,ld.log(tl.maximum(p0,1.e-300))+scale,-1.e300)
        t1=tl.where(p1>0,ld.log(tl.maximum(p1,1.e-300))+scale,-1.e300)
        t2=tl.where(p2>0,ld.log(tl.maximum(p2,1.e-300))+scale,-1.e300)
        m0,q0=_lse_update(m0,q0,t0);m1,q1=_lse_update(m1,q1,t1);m2,q2=_lse_update(m2,q2,t2)
    l0=m0+ld.log(q0);l1=m1+ld.log(q1);l2=m2+ld.log(q2)
    tl.store(OUT+(pos*3)*PARTS+tile,tl.sum(tl.where(ok,l1-l0,0.),0))
    tl.store(OUT+(pos*3+1)*PARTS+tile,tl.sum(tl.where(ok,l2-l0,0.),0))
    tl.store(OUT+(pos*3+2)*PARTS+tile,tl.sum(tl.where(ok,l0,0.),0))

class StreamedLikelihood:
    """Reusable immutable-topology workspace. Rebuild after ANY topology edit."""
    @torch.no_grad()
    def __init__(self,tree,names,metrics,*,device='cuda:0',block_sites=256,precision='float64',workspace_bytes=8<<30):
        t=time.perf_counter();tree.require_binary()
        if set(names)!=set(tree.names.values()) or len(names)!=len(set(names)):raise ValueError('exact taxa match required')
        if precision not in ('float32','float64') or block_sites<1 or workspace_bytes<1:raise ValueError('invalid precision/workspace')
        self.tree=tree;self.names=tuple(names);self.device=torch.device(device)
        if self.device.type!='cuda':raise ValueError('streamed prototype needs CUDA')
        self.k=len(metrics['category_rates']);self.dtype=getattr(torch,precision)
        if not 1<=self.k<=16:raise ValueError('1..16 rate categories required')
        pi=np.asarray(metrics['frequencies_acgt'],float);q=np.asarray(metrics['Q_acgt'],float)
        rates=np.asarray(metrics['category_rates'],float);weights=np.asarray(metrics['category_weights'],float)
        if q.shape!=(4,4) or pi.shape!=(4,) or not np.isfinite(q).all() or not np.isfinite(pi).all():raise ValueError('invalid Q/pi')
        if (pi<=0).any() or not np.isclose(pi.sum(),1) or not np.allclose(q.sum(1),0,atol=1e-9):raise ValueError('invalid frequencies/generator')
        if not np.allclose(pi[:,None]*q,(pi[:,None]*q).T,atol=1e-9):raise ValueError('Q must be reversible')
        if weights.shape!=rates.shape or not np.isfinite(rates).all() or (rates<0).any() or not np.isfinite(weights).all() or (weights<=0).any() or not np.isclose(weights.sum(),1):raise ValueError('invalid rate mixture')
        self.root,parent,order,depth=rooted(tree);self.depth=max(depth.values())
        self.edges=[edge(u,v) for u in order for v in sorted(tree.graph[u]) if u<v]
        lengths=np.asarray([tree.lengths.get(e,np.nan) for e in self.edges],float)
        if not np.isfinite(lengths).all() or (lengths<=0).any():raise ValueError('positive finite supplied lengths required')
        edge_ids={e:i for i,e in enumerate(self.edges)};rows={}
        for i,(u,v) in enumerate(self.edges):rows[u,v]=i*2;rows[v,u]=i*2+1
        self.directed_rows=len(rows);bytes_per_site=len(rows)*self.k*(4*torch.empty((),dtype=self.dtype).element_size()+8)
        self.width=min(block_sites,workspace_bytes//max(1,bytes_per_site))
        if self.width<1:raise MemoryError('workspace cannot hold even one site slab')
        nrows=len(rows);deps=np.zeros((nrows,2),np.int32);rowedge=np.repeat(np.arange(len(self.edges),dtype=np.int32),2)
        for (u,v),r in rows.items():
            if u not in tree.names:deps[r]=[rows[w,u] for w in sorted(tree.graph[u]-{v})]
        lookup={name:i for i,name in enumerate(names)}
        leaves=sorted(tree.names);leafrows=[rows[u,next(iter(tree.graph[u]))] for u in leaves]
        leafindex=[lookup[tree.names[u]] for u in leaves]
        up=[[] for _ in range(self.depth+1)];down=[[] for _ in up]
        for u in order:
            if u not in tree.names:
                if u!=self.root:up[depth[u]].append(rows[u,parent[u]])
                down[depth[u]].extend(rows[u,v] for v in sorted(tree.graph[u]) if parent.get(v)==u)
        def gpu(x):return torch.as_tensor(np.asarray(x,np.int32),device=self.device)
        self.deps=gpu(deps);self.rowedge=gpu(rowedge);self.leafrows=gpu(leafrows);self.leafindex=gpu(leafindex)
        self.up_levels=[gpu(g) for g in reversed(up) if g]
        self.levels=self.up_levels+[gpu(g) for g in down if g]
        self.rootrows=gpu([rows[v,self.root] for v in sorted(tree.graph[self.root])])
        self.internal_edges=[e for e in self.edges if all(u not in tree.names for u in e)]
        incoming=[];self.branches=[]
        for u,v in self.internal_edges:
            a,b=sorted(tree.graph[u]-{v});c,d=sorted(tree.graph[v]-{u});self.branches.append((a,b,c,d))
            incoming.append([rows[a,u],rows[b,u],rows[c,v],rows[d,v]])
        self.score_edges=gpu([edge_ids[e] for e in self.internal_edges]);self.incoming=gpu(incoming)
        qdev=torch.tensor(q,device=self.device,dtype=torch.float64);pdev=torch.tensor(pi,device=self.device,dtype=torch.float64)
        self.p=transition_matrices(qdev,torch.tensor(lengths,device=self.device),torch.tensor(rates,device=self.device)).to(self.dtype)
        self.pi=pdev.to(self.dtype);self.logweights=torch.tensor(np.log(weights),device=self.device)
        self.u=torch.empty((nrows,self.k,self.width,4),device=self.device,dtype=self.dtype)
        self.logs=torch.empty((nrows,self.k,self.width),device=self.device,dtype=torch.float64)
        self.parts=triton.cdiv(self.width,128)
        self.partial=torch.empty((len(self.internal_edges),3,self.parts),device=self.device,dtype=torch.float64)
        self.rootpartial=torch.empty(self.parts,device=self.device,dtype=torch.float64)
        self.message_buffer_bytes=self.u.numel()*self.u.element_size()+self.logs.numel()*self.logs.element_size()
        torch.cuda.synchronize(self.device);self.setup_seconds=time.perf_counter()-t

    @torch.no_grad()
    def score(self,states,*,nni=True):
        """Return fixed-branch NNI log-likelihood deltas, all columns, no fitting."""
        x=np.asarray(states)
        if x.ndim!=2 or x.shape[0]!=len(self.names) or x.shape[1]<1 or not np.issubdtype(x.dtype,np.integer) or x.min()<0 or x.max()>4:raise ValueError('integer DNA matrix0..4 required')
        start=time.perf_counter();totals=torch.zeros((len(self.internal_edges),3),device=self.device,dtype=torch.float64)
        root_ll=torch.zeros((),device=self.device,dtype=torch.float64);nblocks=0
        floor=1e-300 if self.dtype==torch.float64 else 1e-30
        for pos in range(0,x.shape[1],self.width):
            slab=torch.tensor(np.ascontiguousarray(x[:,pos:pos+self.width],dtype=np.uint8),device=self.device)
            s=slab.shape[1];parts=triton.cdiv(s,128)
            _leaf[(len(self.leafrows),self.k,parts)](self.u,self.logs,self.p,slab,self.leafrows,self.leafindex,self.rowedge,self.k,self.width,s,128,floor,enable_fp_fusion=False)
            for level in (self.levels if nni else self.up_levels):
                _send[(len(level),self.k,parts)](self.u,self.logs,self.p,level,self.deps,self.rowedge,self.k,self.width,s,128,floor,enable_fp_fusion=False)
            _root[(parts,)](self.u,self.logs,self.pi,self.logweights,self.rootrows,self.rootpartial,self.k,self.width,s,128,enable_fp_fusion=False)
            if nni:
                _nni[(len(self.internal_edges),parts)](self.u,self.logs,self.p,self.pi,self.logweights,self.score_edges,self.incoming,self.partial,self.k,self.width,s,128,self.parts,enable_fp_fusion=False)
                totals+=self.partial[:,:,:parts].sum(-1)
            root_ll+=self.rootpartial[:parts].sum();nblocks+=1
        root=float(root_ll);output=totals.cpu().numpy()
        if not np.isfinite(output).all() or not math.isfinite(root):raise RuntimeError('nonfinite streamed likelihood')
        return dict(log_likelihood=root,deltas=output[:,:2] if nni else None,max_baseline_error=float(np.max(np.abs(output[:,2]-root))) if nni else None,
            seconds=time.perf_counter()-start,site_blocks=nblocks,block_sites=self.width,depth=self.depth,
            message_buffer_bytes=self.message_buffer_bytes,precision=str(self.dtype),all_sites=x.shape[1])

    def proposals(self,result,min_gain=1e-4):
        """Frozen-branch candidates; no claim of profile-likelihood optimality."""
        out=[]
        for e,b,delta in zip(self.internal_edges,self.branches,result['deltas']):
            alt=int(np.argmax(delta));gain=float(delta[alt])
            if gain>min_gain:out.append(dict(edge=e,branches=b,alternative=alt+1,length=self.tree.lengths[e],gain=gain,log_likelihood=result['log_likelihood']+gain))
        return sorted(out,key=lambda p:(-p['gain'],p['edge']))
