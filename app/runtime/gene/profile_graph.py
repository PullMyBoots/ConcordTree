"""Graph replay of the same two local-profile passes, bounded shared arena."""
import time
import torch
from replayed_likelihood import Replay
from local_profile import CentralStep
from shared_mixture import SharedMixture
from binary_likelihood import BinaryMessages
from flat_uniformized_likelihood import FlatInvariantLikelihood
from pruned_weighted_likelihood_eager import PrunedWeightedLikelihood


class ProfileReplay(Replay):
    def _bind(self,slab,weights):
        arena=getattr(self,'message_workspace',None)
        key=(type(self).__name__,self.phase,self.nni,self.k,self.width,self.directed_rows,
             tuple(map(len,self.levels)),self.u.data_ptr(),self.logs.data_ptr())
        pool=None if arena is None else arena.graph_cache
        entry=None if pool is None else pool.get(key)
        fields=['metadata','deps','rowedge','leafrows','leafindex','rootrows','score_edges','incoming',
                'p','pi','logweights','partial','rootpartial','category_weights',
                'dp','fi','stat_partial','center_partial','new_p']
        if self.binary_invariant:fields+=['codes','iv']
        if entry is None:
            entry={name:getattr(self,name) for name in fields}
            entry.update(levels=self.levels,up_levels=self.up_levels,slab=slab,weights=weights,graph=None)
            if pool is not None:
                pool[key]=entry
                while len(pool)>8:pool.popitem(last=False)
        else:
            self.capture_hit=True;pool.move_to_end(key)
            entry['metadata'].copy_(self.metadata)
            for name in ['p','pi','logweights','category_weights','dp','fi','new_p']+(['codes'] if self.binary_invariant else []):
                entry[name].copy_(getattr(self,name))
            for name in fields:setattr(self,name,entry[name])
            self.levels,self.up_levels=entry['levels'],entry['up_levels']
        self.invariant_weight_tensor=self.category_weights[-1:]
        self.replay_entry=entry
        return entry


class ReplayStep(CentralStep):
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        self.new_p=torch.empty((len(self.internal_edges)*3,self.k,4,4),device=self.device,dtype=self.p.dtype)


class GraphInvariant(ReplayStep,ProfileReplay,SharedMixture,BinaryMessages,FlatInvariantLikelihood):pass
class GraphVariable(ReplayStep,ProfileReplay,SharedMixture,BinaryMessages,PrunedWeightedLikelihood):pass
