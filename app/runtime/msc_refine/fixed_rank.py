"""Fixed sparse quartet rank objective; exact monotone topology search.

Construct evidence once from the unmodified scaffold; every subsequent NNI
is evaluated against ALL affected rows of that same objective, not a changing
representative panel. No statistical confidence gates or score mixtures.
"""
from __future__ import annotations
import time
import numpy as np
import torch
from .core import compile_plan,select_moves,apply_move
from . import _msc_sparse


def scaffold_queries(graph,n):
    from concordtree.assets import load_panel_score_backend
    from concordtree.inference import _saturated_panel,LOCAL_TEMPLATE,LOCAL_ROW_INDEX
    from concordtree._core.sctb_contextual_patch import build_contextual_tree_index,compile_sparse_contextual_panel_plan
    backend=load_panel_score_backend()
    index=build_contextual_tree_index(graph,n,representative_limit=24,plan_backend=backend)
    plan=compile_plan(graph,n);chunks=[]
    for i,edge in enumerate(map(tuple,plan.edges)):
        panel=_saturated_panel(graph,edge,index,i,n)
        sparse=compile_sparse_contextual_panel_plan(graph,panel,LOCAL_TEMPLATE,
                   tree_index=index,row_lookup=LOCAL_ROW_INDEX,plan_backend=backend)
        q=np.array(panel.taxa,dtype=np.int64)[LOCAL_TEMPLATE[sparse.row_indices]]
        q.sort(1);chunks.append(q)
    return np.unique(np.concatenate(chunks),axis=0)


class FixedRankScorer:
    make_queries=staticmethod(scaffold_queries)

    def __init__(self,packed,length,graph,pattern_backend,*,device='cuda:0'):
        from concordtree.statistical import rank_residuals
        from concordtree.pattern_streaming import pattern_frequencies
        self.packed=packed;self.length=int(length);self.device=torch.device(device)
        phase=time.perf_counter();n=len(packed);self.queries=self.make_queries(graph,n)
        self.plan_seconds=time.perf_counter()-phase;phase=time.perf_counter()
        data=torch.from_numpy(packed)
        if packed.nbytes<=1<<30:data=data.to(self.device)
        self.weights=np.empty((len(self.queries),3),dtype=np.float64)
        for start in range(0,len(self.queries),16384):
            q=torch.as_tensor(self.queries[start:start+16384],device=self.device)
            freq=pattern_frequencies(pattern_backend,data,q,length).contiguous()
            score=rank_residuals(freq,10)
            self.weights[start:start+len(q)]=-score.square().cpu().numpy()
        self.score_seconds=time.perf_counter()-phase
        self.timing={}

    def score(self,graph):
        start=time.perf_counter();n=len(self.packed);plan=compile_plan(graph,n)
        edges=np.array([(u,v) for u in sorted(graph) for v in sorted(graph[u]) if u<v],dtype=np.int64)
        result=_msc_sparse.evaluate(edges,n,self.queries,self.weights)
        assert np.array_equal(plan.edges,result['edges'])
        self.timing=dict(score_seconds=time.perf_counter()-start,objective=result['objective'],
                         active_quartets=result['active_quartets'],fixed_quartets=len(self.queries))
        return plan,result['scores']


class QuartetAgreementScorer(FixedRankScorer):
    """Textbook equal-vote SVDquartets assembly, with a fixed sparse query set.

    A quartet votes for its unique minimum-rank-residual resolution. Exact
    ties abstain. No magnitude-dependent confidence weights or temperature.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        best=self.weights.argmax(1)
        unique=(self.weights==self.weights.max(1,keepdims=True)).sum(1)==1
        self.weights=np.eye(3,dtype=np.float64)[best]*unique[:,None]


def refine_fixed(graph,scorer,*,stop_ratio=.005,max_sweeps=None,callback=None):
    graph={u:set(v) for u,v in graph.items()};n=len(scorer.packed);history=[];previous=None
    while True:
        plan,scores=scorer.score(graph);selected=select_moves(graph,plan,scores)
        value=scorer.timing['objective']
        if previous is not None:
            assert np.isclose(value,previous,rtol=1e-9,atol=1e-10),'global objective delta failed'
        for i,c,gain in selected:apply_move(graph,plan.edges[i],plan.branches[i],c)
        gain=sum(m[2] for m in selected);previous=value+gain
        row=dict(sweep=len(history)+1,moves=len(selected),move_ratio=len(selected)/(n-3),
                 summed_objective_gain=gain,**scorer.timing)
        history.append(row)
        if callback:callback(graph,row)
        if not selected:reason='no_improving_nni';break
        if len(selected)/(n-3)<=stop_ratio:reason='move_ratio';break
        if max_sweeps is not None and len(history)>=max_sweeps:reason='max_sweeps';break
    return graph,dict(history=history,stop=reason,all_sites_used=True,branch_lengths_inferred=False,
                 objective='sum of fixed sparse normalized rank10 residual squared',
                 fixed_quartets=len(scorer.queries),quartet_plan_seconds=scorer.plan_seconds,
                 one_time_scoring_seconds=scorer.score_seconds)
