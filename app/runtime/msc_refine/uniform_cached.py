"""Exact dependency reuse for uniform branch-panel quartet search.

This changes execution only. Every edge's estimator uses the same ordered
four components, fixed priorities and panel budget as uniform_search.py.
"""
from __future__ import annotations
import itertools,time
import numpy as np
from .core import compile_plan,select_moves,apply_move
from .uniform_search import UniformPanelScorer,component_bottom_k,PAIR_CLASS
from .state_cycle import StateRecurrence


def subset_queries(plan,rows,component,n):
    chunks=[];offsets=[0]
    for row in rows:
        edge,branches=plan.edges[row],plan.branches[row]
        u,v=map(int,edge)
        groups=[component[int(b),u if j<2 else v] for j,b in enumerate(branches)]
        chosen=[[] for _ in range(4)];total=0
        for rank in range(24):
            for j,g in enumerate(groups):
                if rank<len(g) and total<min(n,24):chosen[j].append(g[rank]);total+=1
        assert total==min(n,24) and all(chosen)
        q=np.array(list(itertools.product(*chosen)),dtype=np.int64)
        chunks.append(q);offsets.append(offsets[-1]+len(q))
    ordered=np.concatenate(chunks);ranks=np.empty_like(ordered)
    for j in range(4):ranks[:,j]=(ordered[:,j,None]>ordered).sum(-1)
    classes=PAIR_CLASS[ranks[:,0,None],ranks[:,[1,2,3]]]
    unique,inverse=np.unique(np.sort(ordered,axis=1),axis=0,return_inverse=True)
    return unique,inverse,classes,np.asarray(offsets,dtype=np.int64)


class CachedUniformScorer(UniformPanelScorer):
    def __init__(self,*args,probability_objective=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.cache={};self.probability_objective=bool(probability_objective)

    def predict(self,q):
        scores=super().predict(q)
        if self.probability_objective:
            scores-=scores.max(1,keepdims=True);scores=np.exp(scores)
            scores/=scores.sum(1,keepdims=True)
        return scores

    def invalidate(self,plan,selected):
        changed=set()
        for row,choice,_ in selected:
            changed.update(map(int,plan.edges[row]))
            changed.add(int(plan.branches[row,1]))
            changed.add(int(plan.branches[row,2 if choice==1 else 3]))
        for edge in list(self.cache):
            if changed.intersection(edge):del self.cache[edge]

    def score(self,graph):
        started=time.perf_counter();n=len(self.packed);plan=compile_plan(graph,n)
        result=np.empty((len(plan.edges),3),dtype=np.float64);active=[]
        for row,(edge,branches) in enumerate(zip(plan.edges,plan.branches)):
            cached=self.cache.get(tuple(map(int,edge)))
            if cached is not None and cached[0]==tuple(map(int,branches)):result[row]=cached[1]
            else:active.append(row)
        if active:
            component=component_bottom_k(graph,n,self.priority)
            q,inverse,classes,offsets=subset_queries(plan,active,component,n)
            planning=time.perf_counter()-started;phase=time.perf_counter()
            weights=self.predict(q);mapped=np.take_along_axis(weights[inverse],classes,axis=1)
            mean=np.add.reduceat(mapped-mapped[:,0,None],offsets[:-1])/np.diff(offsets)[:,None]
            scores=mean*plan.sizes[active].astype(np.float64).prod(1)[:,None]
            for i,row in enumerate(active):
                result[row]=scores[i]
                self.cache[tuple(map(int,plan.edges[row]))]=(tuple(map(int,plan.branches[row])),scores[i].copy())
            count=len(q);sampled=int(offsets[-1]);scoring=time.perf_counter()-phase
        else:planning=time.perf_counter()-started;scoring=0.;count=sampled=0
        self.timing=dict(plan_seconds=planning,score_seconds=scoring,elapsed_seconds=time.perf_counter()-started,
            quartets=count,sampled_edge_rows=sampled,active_edges=len(active),cached_edges=len(plan.edges)-len(active))
        return plan,result


def refine_cached(graph,scorer,*,stop_ratio=.005,callback=None):
    graph={u:set(v) for u,v in graph.items()};n=len(scorer.packed);history=[];recurrence=StateRecurrence()
    while True:
        signature=tuple((u,tuple(sorted(graph[u]))) for u in sorted(graph))
        if recurrence.repeated(signature):raise ArithmeticError('pure sampled search repeats its full nonconverged state')
        plan,scores=scorer.score(graph);selected=select_moves(graph,plan,scores)
        scorer.invalidate(plan,selected)
        for i,c,gain in selected:apply_move(graph,plan.edges[i],plan.branches[i],c)
        row=dict(sweep=len(history)+1,moves=len(selected),move_ratio=len(selected)/(n-3),
            estimated_quartet_potential_gain=sum(m[2] for m in selected),**scorer.timing)
        history.append(row)
        if callback:callback(graph,row)
        if not selected:reason='no_positive_sampled_nni';break
        if len(selected)/(n-3)<=stop_ratio:reason='move_ratio';break
    return graph,dict(history=history,stop=reason,all_informative_sites_used=True,branch_lengths_inferred=False,
        objective=('full all-quartet expected posterior agreement' if scorer.probability_objective else
            'full all-quartet log conditional probability')+'; approximate uniform branch-panel NNI gains',
        exact_global_monotonicity_proved=False,general_msc_consistency_proved=False,
        public_stop_ratio=stop_ratio,internal_panel_taxa=24,randomization_seed=20261025,
        cache='exact dependency cache, no statistical pruning',
        recurrence_detection='Brent full adjacency including internal IDs; one O(N) checkpoint')
