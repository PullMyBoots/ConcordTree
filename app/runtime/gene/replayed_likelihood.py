"""Bounded exact CUDA graph templates over a shared message arena.

Only addresses and launch capacities persist. Every changed tree/model/input is
copied into the template before replay; no conditional likelihood is reused.
Capture templates are discarded before fitting, so fit and score do not retain
separate full message workspaces. LRU8 holds small metadata/output buffers only.
"""
import math,time
import torch
from capture_scope import capture_scope
from flat_uniformized_likelihood import FlatInvariantLikelihood
from pruned_weighted_likelihood_eager import PrunedWeightedLikelihood

class Replay:
    def score_patterns(self,alignment,*,nni=True):
        blocks=sum(math.ceil(x.shape[1]/self.width) for x,_ in alignment.chunks)
        self.use_replay=blocks>=4;self.replay_entry=None
        self.capture_seconds=0.;self.capture_hit=False
        result=super().score_patterns(alignment,nni=nni)
        result.update(graph_replay=self.use_replay,capture_seconds=self.capture_seconds,capture_cache_hit=self.capture_hit)
        return result

    def _bind(self,slab,weights):
        arena=getattr(self,'message_workspace',None)
        key=(type(self).__name__,self.nni,self.k,self.width,self.directed_rows,
             tuple(map(len,self.levels)),self.u.data_ptr(),self.logs.data_ptr())
        pool=None if arena is None else arena.graph_cache
        entry=None if pool is None else pool.get(key)
        fields=['metadata','deps','rowedge','leafrows','leafindex','rootrows','score_edges','incoming',
                'p','pi','logweights','partial','rootpartial','category_weights']
        if self.binary_invariant:fields+=['codes','iv']
        if hasattr(self,'curvature_partial'):
            fields+=['curvature_rows','curvature_edges','curvature_tax','dp','d2p','curvature_partial']
        if entry is None:
            entry={k:getattr(self,k) for k in fields}
            entry.update(levels=self.levels,up_levels=self.up_levels,slab=slab,weights=weights,graph=None)
            if pool is not None:
                pool[key]=entry
                while len(pool)>8:pool.popitem(last=False)
        else:
            self.capture_hit=True;pool.move_to_end(key)
            # All integer DAG arrays/levels are views of ONE metadata allocation.
            entry['metadata'].copy_(self.metadata)
            for field in ['p','pi','logweights','category_weights']+(['codes'] if self.binary_invariant else []):
                entry[field].copy_(getattr(self,field))
            if hasattr(self,'curvature_partial'):
                for field in ['curvature_rows','curvature_edges','curvature_tax','dp','d2p']:entry[field].copy_(getattr(self,field))
            for field in fields:setattr(self,field,entry[field])
            self.levels,self.up_levels=entry['levels'],entry['up_levels']
        self.invariant_weight_tensor=self.category_weights[-1:]
        self.replay_entry=entry
        return entry

    def _kernels(self,slab,weights):
        if not self.use_replay:return super()._kernels(slab,weights)
        entry=self.replay_entry or self._bind(slab,weights)
        if slab.data_ptr()!=entry['slab'].data_ptr():entry['slab'].copy_(slab)
        if weights.data_ptr()!=entry['weights'].data_ptr():entry['weights'].copy_(weights)
        if hasattr(self,'curvature_slab'):self.curvature_slab=entry['slab']
        if entry['graph'] is None:
            started=time.perf_counter();current=torch.cuda.current_stream(self.device)
            stream=torch.cuda.Stream(device=self.device);stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):super()._kernels(entry['slab'],entry['weights'])
            current.wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with capture_scope(graph,stream):super()._kernels(entry['slab'],entry['weights'])
            current.wait_stream(stream);entry['graph']=graph
            self.capture_seconds=time.perf_counter()-started
        entry['graph'].replay()

class InvariantReplay(Replay,FlatInvariantLikelihood):pass
class VariableReplay(Replay,PrunedWeightedLikelihood):pass
