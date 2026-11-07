"""Uniform-shape exact compression, removing zero-evidence padding multiplicity.

The last slab uses the same GPU shape as full slabs. Its extra unknown columns
have exactly zero retained multiplicity, not imputed or subsampled evidence.
This avoids per-record tail-length JIT specializations and bounds cache growth.
"""
import time
import numpy as np
import torch
import triton
import triton.language as tl
from exact_site_patterns import compress_columns
from invariant_site_patterns import InvariantPatterns

@triton.jit
def unpack_padded(P,X,N:tl.constexpr,W:tl.constexpr,S,STRIDE,B:tl.constexpr):
    pos=tl.program_id(0)*B+tl.arange(0,B);ok=pos<N*W
    row=pos//W;site=pos%W;valid=ok&(site<S)
    byte=tl.load(P+row*STRIDE+site//2,valid,other=0)
    code=tl.where(site<S,(byte>>((site%2)*4))&15,4)
    tl.store(X+pos,code,ok)

class StablePatterns(InvariantPatterns):
    @torch.no_grad()
    def compress(self):
        if self.chunks:raise ValueError('already compressed')
        start=time.perf_counter();audits=[];self.invariant_codes=[]
        n=len(self.names);width=min(self.chunk_sites,1<<(int(self.original_sites-1).bit_length()))
        for pos in range(0,self.original_sites,width):
            end=min(self.original_sites,pos+width);observed=end-pos
            packed=torch.tensor(np.ascontiguousarray(self.packed[:,pos//2:(end+1)//2]),device=self.device)
            x=torch.empty((n,width),device=self.device,dtype=torch.uint8)
            unpack_padded[(triton.cdiv(n*width,512),)](packed,x,n,width,observed,packed.shape[1],512)
            patterns,weights,audit=compress_columns(x)
            minimum=patterns.min(0).values
            maximum=torch.where(patterns<4,patterns,0).max(0).values
            codes=torch.where(minimum==4,4,torch.where(minimum==maximum,minimum,5)).to(torch.uint8)
            padding=width-observed
            if padding:
                all_unknown=codes==4
                # Equal-hash collisions can split an observation class without
                # ever merging unequal columns. Remove padding from its total
                # multiplicity even in that conservative collision case.
                weights=weights.clone();remaining=padding
                for j in torch.nonzero(all_unknown).flatten().tolist():
                    take=min(remaining,int(weights[j]));weights[j]-=take;remaining-=take
                    if not remaining:break
                assert remaining==0,'padding observation missing from exact classes'
                keep=weights>0
                patterns,weights,codes=patterns[:,keep],weights[keep],codes[keep]
            assert int(weights.sum())==observed
            self.chunks.append((patterns.cpu().numpy(),weights.cpu().numpy()))
            self.invariant_codes.append(codes.cpu().numpy())
            audits.append(dict(original_sites=observed,padded_shape=width,zero_evidence_padding=padding,
                patterns=patterns.shape[1],exact_collision_splits=audit['exact_collision_splits']))
            del x,packed,patterns,weights
        self.metrics=dict(seconds=time.perf_counter()-start,parse_seconds=self.parse_seconds,
            original_sites=self.original_sites,patterns=sum(x['patterns'] for x in audits),
            chunk_sites=width,chunks=len(audits),chunk_audits=audits,
            host_pattern_bytes=sum(x.nbytes+w.nbytes for x,w in self.chunks),raw_packed_bytes=self.packed.nbytes,
            equality='exact original observation patterns and multiplicities; padding removed')
        return self.metrics
