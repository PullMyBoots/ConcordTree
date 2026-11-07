"""Observed-locus barycenter scorer on the unchanged native quartet search.

Same finite conditional learner and fixed24 uniform branch panels. Exact
bit counting changes execution only. The sampled NNI gains do not imply
global monotonicity or generic MSC consistency. No tree branches are fitted.
"""
from __future__ import annotations
import time
import numpy as np
import torch
import triton
from .bit_orbits import encode,counts
from .locus_barycenter import barycenter
from .barycenter_kernel import BarycenterKernel
from .uniform_native import NativeUniformScorer


class BarycenterScorer(NativeUniformScorer):
    @torch.inference_mode()
    def __init__(self,packed,length,regions,asset,*,device='cuda:0'):
        self.length=int(length);self.device=torch.device(device);n=len(packed)
        if self.length<=0 or packed.shape[1]!=(self.length+1)//2 or packed.dtype!=np.uint8:
            raise ValueError('native packed MSA and positive length required')
        rr=np.asarray(regions,dtype=np.int64)
        if (rr.ndim!=2 or rr.shape[1]!=2 or len(rr)==0 or rr[0,0]!=0 or
            rr[-1,1]!=length or np.any(rr[:,1]<=rr[:,0]) or
            np.any(rr[:-1,1]!=rr[1:,0])):
            raise ValueError('observed nonempty regions must exactly cover the MSA')
        self.regions=rr;self.packed=np.empty((n,0),dtype=np.uint8)
        self.planes=torch.empty((n,3,triton.cdiv(length,32)),device=device,dtype=torch.uint32)
        transfer_rows=max(1,(64<<20)//max(1,packed.shape[1]))
        for first in range(0,n,transfer_rows):
            last=min(n,first+transfer_rows)
            slab=torch.as_tensor(packed[first:last],device=device)
            self.planes[first:last].copy_(encode(slab,length))
            del slab
        self.model=BarycenterKernel(asset,device=device)
        self.priority=np.random.default_rng(20261025).permutation(n)
        self.probability_objective=False;self.cache={};self.timing={};self.check=None
        tiles=triton.cdiv(int(((rr[:,1]+31)//32-rr[:,0]//32).max()),128)
        # Layout/memory budget only. No query,locus or site is omitted.
        per_query=len(rr)*15*(4*tiles+80)
        self.count_rows=max(1,min(8192,(64<<20)//per_query))
        self.metadata=dict(observed_loci=len(rr),all_sites_counted=True,
            constant_quartet_patterns='conditioned out within each observed locus',
            count_query_rows=self.count_rows,bitplane_bytes=self.planes.numel()*self.planes.element_size(),
            scratch_layout_budget_bytes=64<<20,branch_lengths_inferred=False,
            general_msc_consistency_proved=False)

    @torch.inference_mode()
    def predict(self,q):
        values=np.empty((len(q),3),dtype=np.float64)
        for start in range(0,len(q),8192):
            query=q[start:start+8192];features=[]
            for first in range(0,len(query),self.count_rows):
                if self.check is not None:self.check()
                c=counts(self.planes,query[first:first+self.count_rows],self.regions)
                features.append(barycenter(c));del c
            x=torch.cat(features)
            values[start:start+len(query)]=self.model.predict_features(x).cpu().numpy()
        return values
