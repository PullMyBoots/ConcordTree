"""Same accumulated target with GPU integer queries and native NNI plans."""
import time
import numpy as np
from concordtree.assets import load_panel_score_backend
from .active_quartet import ActiveQuartetScorer,refine_active,tree_edges
from .core import EdgePlan
from . import _msc_plan
from .topology_gpu_streamsum import evaluate_stream
from . import _msc_active_rmq as _msc_active
from . import _msc_active_plan,_msc_active_norm
from ._msc_incremental_plan import IncrementalPlanner


class StreamSumActiveScorer(ActiveQuartetScorer):
    def __init__(self,base,*,collect_support=False):
        super().__init__(base)
        self.panel_backend=load_panel_score_backend()
        self.incremental_planner=IncrementalPlanner()
        self.collect_support=collect_support
        self.last_support_scan=None

    def score(self,graph):
        started=time.perf_counter();n=len(self.packed)
        edges=tree_edges(graph).astype(np.int32)
        messages=self.panel_backend.compile_directed_nearest_messages(edges,2*n-2,n,24)
        proposal=self.incremental_planner.propose(edges,n,messages)
        proposed=proposal["queries"]
        new=_msc_active_plan.missing(self.queries,proposed)
        planning=time.perf_counter()-started;phase=time.perf_counter()
        newweights=self.base.predict(new) if len(new) else np.empty((0,3))
        if newweights.shape!=(len(new),3) or not np.isfinite(newweights).all():
            raise ArithmeticError('finite canonical quartet scores required')
        scoring=time.perf_counter()-phase;phase=time.perf_counter()
        if len(new):
            self.queries,self.weights,is_new=_msc_active_plan.merge(
                self.queries,self.weights,new,newweights)
        else:is_new=np.zeros(len(self.queries),dtype=bool)
        plan=EdgePlan(**_msc_plan.compile(graph,n))
        result=evaluate_stream(edges,n,self.queries,self.weights,is_new,device=self.base.device,
                               return_classes=self.collect_support)
        if self.collect_support:
            self.last_support_scan=(edges,result['edges'],result['classes'])
        if not np.array_equal(plan.edges,result['edges']):
            raise AssertionError('native exact NNI edge order changed')
        new_value=float(result['new_sum'])
        bound=512*np.finfo(np.float64).eps*max(1.,result['magnitude_sum'])
        scores=np.asarray(result['scores']);scores[np.abs(scores)<=bound]=0.
        self.timing=dict(objective=float(result['objective']),new_evidence_objective=new_value,
            new_quartets=len(new),proposed_quartets=int(proposal["proposed_quartets"]),accumulated_quartets=len(self.queries),
            evidence_bytes=self.queries.nbytes+self.weights.nbytes,active_quartets=int(result['active_quartets']),
            planning_seconds=planning,quartet_seconds=scoring,
            materialized_quartets=len(proposed),changed_panels=int(proposal["changed_edges"]),exact_reduction_seconds=time.perf_counter()-phase,
            arithmetic_gain_bound=float(bound),elapsed_seconds=time.perf_counter()-started)
        return plan,scores

    def branch_support(self,graph):
        from branch_support import species_records
        from .topology_gpu_streamsum import classify_stream
        started=time.perf_counter();edges=tree_edges(graph).astype(np.int32)
        if self.last_support_scan is None:raise RuntimeError('no captured search evidence')
        before,internal,codes=self.last_support_scan
        remap=not np.array_equal(before,edges)
        if remap:
            # Only remap existing evidence to the final tree. No quartet model,
            # alignment score, topology search or new adaptive queries are run.
            idx,codes=classify_stream(edges,len(self.packed),self.queries,device=self.base.device)
            internal=idx['edges']
        records=species_records(internal,codes,self.weights)
        return dict(branches=records,source='remapped_cached_quartets' if remap else 'last_search_scan',
            additional_inference_runs=0,additional_quartet_scores=0,
            additional_classification_passes=int(remap),cached_unique_quartets=len(self.queries),
            retained_classification_bytes=int(codes.nbytes),final_state_verified=True,
            summary_seconds=time.perf_counter()-started)
