"""Uniform branch-panel Monte Carlo of the full quartet log-potential.

The target is sum over all quartets of their candidate-tree log conditional
probability. Edge gains are sampled approximations, NOT exact global deltas.
No median, confidence, rank mixture, inferred branch length or data router.
"""
from __future__ import annotations
import itertools
import time
import numpy as np
import torch
from .core import compile_plan,select_moves,apply_move
from .kernel_tree import streamed_counts
from .kernel_direct import DirectMSCKernel

PAIR_CLASS=np.array([[-1,0,1,2],[0,-1,2,1],[1,2,-1,0],[2,1,0,-1]])


def component_bottom_k(graph,n,priority,k=24):
    """Rerooted bottom-k priorities in all directed components, O(kN log k)."""
    parent={n:-1};order=[n]
    for u in order:
        for v in sorted(graph[u]):
            if v!=parent[u]:parent[v]=u;order.append(v)
    result={}
    def merge(groups):
        return tuple(sorted(itertools.chain.from_iterable(groups),key=lambda x:priority[x])[:k])
    for u in reversed(order):
        if parent[u]>=0:
            result[u,parent[u]]=(u,) if u<n else merge(result[v,u] for v in graph[u] if v!=parent[u])
    for u in order:
        for v in graph[u]:
            if parent.get(v)==u:
                result[u,v]=(u,) if u<n else merge(result[w,u] for w in graph[u] if w!=v)
    return result


def edge_panel_queries(graph,n,priority):
    plan=compile_plan(graph,n);component=component_bottom_k(graph,n,priority);chunks=[];offsets=[0]
    for edge,branches in zip(plan.edges,plan.branches):
        u,v=map(int,edge);groups=[component[int(b),u if i<2 else v] for i,b in enumerate(branches)]
        chosen=[[] for _ in range(4)];total=0
        for rank in range(24):
            for j,g in enumerate(groups):
                if rank<len(g) and total<min(n,24):chosen[j].append(g[rank]);total+=1
        assert total==min(n,24) and all(chosen)
        q=np.array(list(itertools.product(*chosen)),dtype=np.int64)
        chunks.append(q);offsets.append(offsets[-1]+len(q))
    ordered=np.concatenate(chunks);ranks=np.empty_like(ordered)
    for i in range(4):ranks[:,i]=(ordered[:,i,None]>ordered).sum(-1)
    classes=PAIR_CLASS[ranks[:,0,None],ranks[:,[1,2,3]]]
    canonical=np.sort(ordered,axis=1)
    unique,inverse=np.unique(canonical,axis=0,return_inverse=True)
    return plan,unique,inverse,classes,np.asarray(offsets,dtype=np.int64)


class UniformPanelScorer:
    def __init__(self,packed,length,asset,*,device='cuda:0',priority=None):
        self.packed=packed;self.length=int(length);self.device=torch.device(device)
        self.data=torch.from_numpy(packed)
        if packed.nbytes<=1<<30:self.data=self.data.to(device)
        self.model=DirectMSCKernel(asset,device=device)
        if self.model.output_kind!='logits':raise ValueError('conditional multinomial logits required')
        self.priority=np.random.default_rng(20261025).permutation(len(packed)) if priority is None else priority
        self.timing={}

    def predict(self,q):
        if self.length==0:return np.zeros((len(q),3),dtype=np.float64)
        values=np.empty((len(q),3),dtype=np.float64)
        for start in range(0,len(q),8192):
            ix=torch.as_tensor(q[start:start+8192],device=self.device)
            c=streamed_counts(self.data,ix,self.length)
            values[start:start+len(ix)]=self.model.predict_orbits(c).cpu().numpy()
        return values

    def score(self,graph):
        started=time.perf_counter();plan,q,inverse,classes,offsets=edge_panel_queries(graph,len(self.packed),self.priority)
        planning=time.perf_counter()-started;phase=time.perf_counter()
        weights=self.predict(q);mapped=np.take_along_axis(weights[inverse],classes,axis=1)
        mean_gain=np.add.reduceat(mapped-mapped[:,0,None],offsets[:-1])/np.diff(offsets)[:,None]
        # Total affected quartets times sample mean is an unbiased gain
        # estimator for each FIXED tree under independent priority sampling.
        scores=mean_gain*plan.sizes.astype(np.float64).prod(1)[:,None]
        self.timing=dict(plan_seconds=planning,score_seconds=time.perf_counter()-phase,
            elapsed_seconds=time.perf_counter()-started,quartets=len(q),sampled_edge_rows=int(offsets[-1]))
        return plan,scores


class UniformMLPControl(UniformPanelScorer):
    def __init__(self,packed,length,pattern_backend,*,device='cuda:0',priority=None):
        from concordtree.assets import load_mlp
        self.packed=packed;self.length=int(length);self.device=torch.device(device)
        self.data=torch.from_numpy(packed)
        if packed.nbytes<=1<<30:self.data=self.data.to(device)
        self.model=load_mlp(self.device);self.pattern_backend=pattern_backend
        self.priority=np.random.default_rng(20261025).permutation(len(packed)) if priority is None else priority
        self.timing={}

    def predict(self,q):
        from concordtree.pattern_streaming import pattern_frequencies
        if self.length==0:return np.zeros((len(q),3),dtype=np.float64)
        values=np.empty((len(q),3),dtype=np.float64)
        for start in range(0,len(q),8192):
            ix=torch.as_tensor(q[start:start+8192],device=self.device)
            freq=pattern_frequencies(self.pattern_backend,self.data,ix,self.length)
            with torch.inference_mode():score=self.model(freq)
            values[start:start+len(ix)]=score.cpu().numpy()
        return values


def refine_uniform(graph,scorer,*,stop_ratio=.005,callback=None):
    graph={u:set(v) for u,v in graph.items()};n=len(scorer.packed);history=[];seen=set()
    while True:
        # Internal IDs determine tie scheduling. Retain the FULL adjacency
        # state, not merely a canonical leaf-labelled topology signature.
        signature=tuple((u,tuple(sorted(graph[u]))) for u in sorted(graph))
        if signature in seen:raise ArithmeticError('pure sampled search repeats its full nonconverged state')
        seen.add(signature);plan,scores=scorer.score(graph);selected=select_moves(graph,plan,scores)
        for i,c,gain in selected:apply_move(graph,plan.edges[i],plan.branches[i],c)
        row=dict(sweep=len(history)+1,moves=len(selected),move_ratio=len(selected)/(n-3),
            estimated_quartet_log_potential_gain=sum(m[2] for m in selected),**scorer.timing)
        history.append(row)
        if callback:callback(graph,row)
        if not selected:reason='no_positive_sampled_nni';break
        if len(selected)/(n-3)<=stop_ratio:reason='move_ratio';break
    return graph,dict(history=history,stop=reason,all_informative_sites_used=True,branch_lengths_inferred=False,
        objective='full all-quartet log conditional probability; approximate uniform branch-panel NNI gains',
        exact_global_monotonicity_proved=False,general_msc_consistency_proved=False,
        public_stop_ratio=stop_ratio,internal_panel_taxa=24,randomization_seed=20261025)
