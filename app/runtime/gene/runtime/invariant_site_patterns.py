"""Exact pattern input plus cached invariant-state codes; all sites retained."""
import time
import numpy as np
import torch
import triton
from exact_site_patterns import PackedPatterns,compress_columns,_unpack

class InvariantPatterns(PackedPatterns):
    @torch.no_grad()
    def compress(self):
        if self.chunks: raise ValueError('already compressed; never duplicate cached work')
        start=time.perf_counter();audits=[];self.invariant_codes=[]
        for pos in range(0,self.original_sites,self.chunk_sites):
            end=min(self.original_sites,pos+self.chunk_sites);l=end-pos;n=len(self.names)
            packed=torch.tensor(np.ascontiguousarray(self.packed[:,pos//2:(end+1)//2]),device=self.device)
            x=torch.empty((n,l),device=self.device,dtype=torch.uint8)
            _unpack[(triton.cdiv(n*l,512),)](packed,x,n,l,packed.shape[1],512)
            patterns,weights,audit=compress_columns(x)
            # Separate chunks preserve column weights. No huge global dictionary.
            minimum=patterns.min(0).values
            maximum=torch.where(patterns<4,patterns,0).max(0).values
            codes=torch.where(minimum==4,4,torch.where(minimum==maximum,minimum,5)).to(torch.uint8)
            self.invariant_codes.append(codes.cpu().numpy())
            self.chunks.append((patterns.cpu().numpy(),weights.cpu().numpy()))
            audits.append(audit)
            del x,packed,patterns,weights
        self.metrics=dict(seconds=time.perf_counter()-start,parse_seconds=self.parse_seconds,
            original_sites=self.original_sites,patterns=sum(a['patterns'] for a in audits),
            chunk_sites=self.chunk_sites,chunks=len(audits),chunk_audits=audits,
            host_pattern_bytes=sum(x.nbytes+w.nbytes for x,w in self.chunks),
            raw_packed_bytes=self.packed.nbytes)
        return self.metrics


    def blocks_with_invariant(self,width):
        for (x,w),codes in zip(self.chunks,self.invariant_codes):
            for s in range(0,x.shape[1],width):
                yield x[:,s:s+width],w[s:s+width],codes[s:s+width]
