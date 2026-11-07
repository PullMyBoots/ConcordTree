"""Same cached quartet criterion with native bounded component/query planning."""
import time
import numpy as np
from .core import EdgePlan
from .uniform_cached import CachedUniformScorer
from . import _msc_plan,_uniform_plan

class NativeUniformScorer(CachedUniformScorer):
    EDGE_CHUNK=128
    def score(self,graph):
        started=time.perf_counter();n=len(self.packed);plan=EdgePlan(**_msc_plan.compile(graph,n))
        result=np.empty((len(plan.edges),3),dtype=np.float64);active=[]
        for row,(edge,branches) in enumerate(zip(plan.edges,plan.branches)):
            prior=self.cache.get(tuple(map(int,edge)))
            if prior is not None and prior[0]==tuple(map(int,branches)):result[row]=prior[1]
            else:active.append(row)
        component=_uniform_plan.components(graph,n,self.priority) if active else None
        planning=time.perf_counter()-started;scoring=0.;count=sampled=peak_rows=0
        for first in range(0,len(active),self.EDGE_CHUNK):
            rows=np.asarray(active[first:first+self.EDGE_CHUNK],dtype=np.int64);phase=time.perf_counter()
            q,classes,offsets=_uniform_plan.queries(*component,plan.edges,plan.branches,rows)
            planning+=time.perf_counter()-phase;phase=time.perf_counter()
            weights=self.predict(q);mapped=np.take_along_axis(weights,classes,axis=1)
            mean=np.add.reduceat(mapped-mapped[:,0,None],offsets[:-1])/np.diff(offsets)[:,None]
            scores=mean*plan.sizes[rows].astype(np.float64).prod(1)[:,None]
            for j,row in enumerate(rows):
                result[row]=scores[j]
                self.cache[tuple(map(int,plan.edges[row]))]=(tuple(map(int,plan.branches[row])),scores[j].copy())
            scoring+=time.perf_counter()-phase;count+=len(q);sampled+=int(offsets[-1]);peak_rows=max(peak_rows,int(offsets[-1]))
        self.timing=dict(plan_seconds=planning,score_seconds=scoring,elapsed_seconds=time.perf_counter()-started,
            quartets=count,sampled_edge_rows=sampled,active_edges=len(active),cached_edges=len(plan.edges)-len(active),
            peak_query_chunk_rows=peak_rows,query_chunk_row_bound=self.EDGE_CHUNK*1296,native_component_and_queries=True)
        return plan,result
