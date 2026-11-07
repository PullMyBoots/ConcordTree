"""Fixed sparse quartet conditional-mean target; exact monotone NNI search.

The finite statistical scorer has no general MSC consistency certificate.
No gene tree is constructed, no branch length is fitted. Fixed scaffold is
used only to choose a fixed evidence query set, not to train the model.
"""
from __future__ import annotations
import time
import numpy as np
import torch
from .fixed_rank import FixedRankScorer
from .msc_kernel import MSCKernel,simplex_projection
from .orbit_counts import orbit_counts


@torch.inference_mode()
def streamed_counts(data,queries,length,*,slab_sites=1<<20):
    if data.is_cuda:return orbit_counts(data,queries,length)
    total=torch.zeros((len(queries),15),device=queries.device,dtype=torch.int64)
    for start in range(0,length,slab_sites):
        end=min(length,start+slab_sites)
        slab=data[:,start//2:(end+1)//2].contiguous().to(queries.device)
        total+=orbit_counts(slab,queries,end-start).to(torch.int64)
    return total


class FixedKernelScorer(FixedRankScorer):
    def __init__(self,packed,length,graph,asset,*,device='cuda:0'):
        self.packed=packed;self.length=int(length);self.device=torch.device(device)
        phase=time.perf_counter();self.queries=self.make_queries(graph,len(packed))
        self.plan_seconds=time.perf_counter()-phase;phase=time.perf_counter()
        self.model=MSCKernel(asset,device=device)
        data=torch.from_numpy(packed)
        if packed.nbytes<=1<<30:data=data.to(device)
        self.weights=np.empty((len(self.queries),3),dtype=np.float64)
        for start in range(0,len(self.queries),8192):
            q=torch.as_tensor(self.queries[start:start+8192],device=device)
            counts=streamed_counts(data,q,self.length)
            score=self.model.probabilities(counts)
            self.weights[start:start+len(q)]=score.cpu().numpy()
        self.score_seconds=time.perf_counter()-phase;self.timing={}
        self.metadata=dict(input_mode='MSA-only',objective='sum of fixed sparse estimated quartet conditional probabilities',
            all_sites_counted=True,model_consistency_proved=False,model_asset=str(asset),
            missingness='quartet complete cases; no MNAR consistency claim')


class FixedMLPControlScorer(FixedRankScorer):
    """Search-control ablation, same queries/NNI, native MLP preprocessing."""
    def __init__(self,packed,length,graph,pattern_backend,queries,*,device='cuda:0'):
        from concordtree.assets import load_mlp
        from concordtree.pattern_streaming import pattern_frequencies
        self.packed=packed;self.length=int(length);self.device=torch.device(device)
        self.queries=queries;self.plan_seconds=0.;phase=time.perf_counter();model=load_mlp(self.device)
        data=torch.from_numpy(packed)
        if packed.nbytes<=1<<30:data=data.to(self.device)
        self.weights=np.empty((len(queries),3),dtype=np.float64)
        for start in range(0,len(queries),16384):
            q=torch.as_tensor(queries[start:start+16384],device=self.device)
            f=pattern_frequencies(pattern_backend,data,q,length).contiguous()
            with torch.inference_mode():scores=torch.softmax(model(f),-1)
            self.weights[start:start+len(q)]=scores.cpu().numpy()
        self.score_seconds=time.perf_counter()-phase;self.timing={}
