"""Exact bounded-chunk pattern compression, with hash-collision verification.

Column hashes are accelerators, never evidence of equality. Within each sorted
hash run, EVERY adjacent full observation column is compared exactly. A collision
can leave duplicates in separate runs (less compression), never merge distinct
observations. Missing DNA code4 is a state of the observation, not an imputation.
No site or taxon sampling. Same-position stationary site model required for
weighted likelihood; do not combine partitions with different model parameters.
"""
from __future__ import annotations
import time
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _unpack(P, X, N:tl.constexpr, L:tl.constexpr, STRIDE:tl.constexpr, B:tl.constexpr):
    pos=tl.program_id(0)*B+tl.arange(0,B); ok=pos<N*L
    row=pos//L; site=pos%L
    byte=tl.load(P+row*STRIDE+site//2,ok,other=0)
    code=(byte>>((site%2)*4))&15
    tl.store(X+pos,code,ok)


@triton.jit
def _hash(X,H,N:tl.constexpr,L:tl.constexpr,BN:tl.constexpr,BS:tl.constexpr,BITS:tl.constexpr):
    row=tl.program_id(0)*BN+tl.arange(0,BN)
    site=tl.program_id(1)*BS+tl.arange(0,BS)
    code=tl.load(X+row[:,None]*L+site[None,:],(row[:,None]<N)&(site[None,:]<L),other=0).to(tl.uint64)
    value=(row[:,None].to(tl.uint64)*5+code+1).to(tl.uint64)
    value=(value^(value>>30))*0xbf58476d1ce4e5b9
    value=(value^(value>>27))*0x94d049bb133111eb
    value=value^(value>>31)
    value=tl.where(row[:,None]<N,value,0)
    out=tl.xor_sum(value,axis=0)
    if BITS<64: out=out&((1<<BITS)-1)
    tl.store(H+tl.program_id(0)*L+site,out,site<L)


@triton.jit
def _reduce_xor(H,OUT,ROWS:tl.constexpr,L:tl.constexpr,BN:tl.constexpr,BS:tl.constexpr):
    row=tl.program_id(0)*BN+tl.arange(0,BN); site=tl.program_id(1)*BS+tl.arange(0,BS)
    value=tl.load(H+row[:,None]*L+site[None,:],(row[:,None]<ROWS)&(site[None,:]<L),other=0)
    tl.store(OUT+tl.program_id(0)*L+site,tl.xor_sum(value.to(tl.uint64),axis=0),site<L)


@triton.jit
def _different(X,KEYS,PERM,BAD,N:tl.constexpr,L:tl.constexpr,BN:tl.constexpr,BS:tl.constexpr):
    row=tl.program_id(0)*BN+tl.arange(0,BN); site=tl.program_id(1)*BS+tl.arange(0,BS)
    here=tl.load(KEYS+site,site<L,other=0); prev=tl.load(KEYS+site-1,(site>0)&(site<L),other=0)
    possible=(site>0)&(site<L)&(here==prev)
    col=tl.load(PERM+site,possible,other=0); old=tl.load(PERM+site-1,possible,other=0)
    mask=(row[:,None]<N)&possible[None,:]
    a=tl.load(X+row[:,None]*L+col[None,:],mask,other=0)
    b=tl.load(X+row[:,None]*L+old[None,:],mask,other=0)
    mismatch=tl.max((a!=b).to(tl.int32),axis=0)
    tl.store(BAD+tl.program_id(0)*L+site,mismatch,site<L)


@torch.no_grad()
def compress_columns(x, *, hash_bits=64):
    """Return GPU uint8 patterns, int64 counts, audit (current device)."""
    if x.ndim!=2 or min(x.shape)<1 or x.dtype!=torch.uint8 or not x.is_cuda:
        raise ValueError('CUDA uint8 nonempty taxa x sites required')
    if not 0<=hash_bits<=64: raise ValueError('hash_bits must be0..64')
    if bool((x>4).any()): raise ValueError('DNA observations must be0..4')
    x=x.contiguous();n,l=x.shape;start=time.perf_counter(); bn=16;bs=128
    rows=triton.cdiv(n,bn); h=torch.empty((rows,l),device=x.device,dtype=torch.int64)
    _hash[(rows,triton.cdiv(l,bs))](x,h,n,l,bn,bs,hash_bits)
    while rows>1:
        newrows=triton.cdiv(rows,bn)
        out=torch.empty((newrows,l),device=x.device,dtype=torch.int64)
        _reduce_xor[(newrows,triton.cdiv(l,bs))](h,out,rows,l,bn,bs)
        h=out;rows=newrows
    keys,perm=torch.sort(h[0]);del h
    boundary=torch.ones(l,device=x.device,dtype=torch.bool)
    boundary[1:]=keys[1:]!=keys[:-1]
    equal_hash_pairs=int((~boundary).sum())
    collision_pairs=0
    if equal_hash_pairs:
        bad=torch.empty((triton.cdiv(n,bn),l),device=x.device,dtype=torch.bool)
        _different[(bad.shape[0],triton.cdiv(l,bs))](x,keys,perm,bad,n,l,bn,bs)
        different=bad.any(0);collision_pairs=int((different&(~boundary)).sum())
        boundary|=different
        del bad,different
    starts=torch.nonzero(boundary).flatten()
    ends=torch.cat((starts[1:],starts.new_tensor([l])))
    weights=ends-starts
    patterns=x.index_select(1,perm[starts])
    assert int(weights.sum())==l and bool((weights>0).all())
    torch.cuda.synchronize(x.device)
    return patterns,weights,dict(original_sites=l,patterns=len(starts),equal_hash_adjacent_pairs=equal_hash_pairs,
        exact_collision_splits=collision_pairs,hash_bits=hash_bits,seconds=time.perf_counter()-start,
        equality='all adjacent columns with equal hash compared over every taxon')


class PackedPatterns:
    """Native original-site packed input and bounded exact pattern construction.

    Persistent compressed chunks on host, no full uncompressed MSA. Different
    chunks need not share a pattern dictionary. GPU construction scratch is
    accounted, not hidden in the score timer. Retained raw packed input can be
    released after a fixed parameter sample is extracted by the caller.
    """
    def __init__(self,path,*,device='cuda:0',max_chunk_sites=1<<20,observation_bytes=256<<20):
        from concordtree.assets import load_backends
        t=time.perf_counter();backend,_=load_backends()
        self.packed,names,self.original_sites=backend.load_phy_to_packed_tensor(str(path),False)
        self.names=list(names);self.device=torch.device(device);self.chunks=[]
        self.parse_seconds=time.perf_counter()-t
        if len(set(names))!=len(names) or len(names)<4 or self.original_sites<1:
            raise ValueError('invalid packed alignment dimensions/names')
        cap=max(2,min(max_chunk_sites,observation_bytes//len(names)))
        self.chunk_sites=1<<(int(cap).bit_length()-1)

    def parameter_sample(self,cap=16384,seed=20261022):
        from bounded_parameter_fit import select_parameter_sites
        idx=select_parameter_sites(self.original_sites,cap,seed)
        from _packed_extract import extract
        sample=extract(self.packed,idx,torch.get_num_threads())
        if sample.max()>4 or not np.all(np.any(sample<4,axis=1)):
            raise ValueError('invalid/entirely-missing sampled taxon; no silent fallback')
        return sample,idx

    @torch.no_grad()
    def compress(self):
        if self.chunks: raise ValueError('already compressed; never duplicate cached work')
        start=time.perf_counter();audits=[]
        for pos in range(0,self.original_sites,self.chunk_sites):
            end=min(self.original_sites,pos+self.chunk_sites);l=end-pos;n=len(self.names)
            packed=torch.tensor(np.ascontiguousarray(self.packed[:,pos//2:(end+1)//2]),device=self.device)
            x=torch.empty((n,l),device=self.device,dtype=torch.uint8)
            _unpack[(triton.cdiv(n*l,512),)](packed,x,n,l,packed.shape[1],512)
            patterns,weights,audit=compress_columns(x)
            # Separate chunks preserve column weights. No huge global dictionary.
            self.chunks.append((patterns.cpu().numpy(),weights.cpu().numpy()))
            audits.append(audit)
            del x,packed,patterns,weights
        self.metrics=dict(seconds=time.perf_counter()-start,parse_seconds=self.parse_seconds,
            original_sites=self.original_sites,patterns=sum(a['patterns'] for a in audits),
            chunk_sites=self.chunk_sites,chunks=len(audits),chunk_audits=audits,
            host_pattern_bytes=sum(x.nbytes+w.nbytes for x,w in self.chunks),
            raw_packed_bytes=self.packed.nbytes)
        return self.metrics

    def release_packed(self):
        self.packed=None

    def blocks(self,width):
        for x,w in self.chunks:
            for pos in range(0,x.shape[1],width):
                yield x[:,pos:pos+width],w[pos:pos+width]
