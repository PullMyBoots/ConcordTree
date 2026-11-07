"""Exact equality reduction across all retained pattern chunks.

The input packed alignment must already have been released by the caller after
all initialization views. Only derived positive multiplicities are modified.
First-occurrence representatives preserve deterministic source identity. Hashes
never equate unequal state vectors, including unknown states.
"""
import time
import numpy as np
import torch
from _global_patterns import fold

def compress_globally(alignment):
    if alignment.packed is not None:raise ValueError('release packed input after extracting initialization views first')
    start=time.perf_counter();old=len(alignment.chunks)
    for i,(x,w) in enumerate(alignment.chunks):
        if not x.flags.c_contiguous:alignment.chunks[i]=(np.ascontiguousarray(x),w)
    stats=fold([x for x,w in alignment.chunks],[w for x,w in alignment.chunks],min(64,torch.get_num_threads()))
    fold_seconds=time.perf_counter()-start
    # Replace one chunk at a time; do not retain old AND new full alignments.
    for i in range(len(alignment.chunks)):
        x,w=alignment.chunks[i];keep=np.flatnonzero(w>0)
        if len(keep)==len(w):continue
        if len(keep):
            out=np.empty((x.shape[0],len(keep)),dtype=np.uint8)
            np.take(x,keep,axis=1,out=out,mode='clip')
            alignment.chunks[i]=(out,w[keep].copy())
            alignment.invariant_codes[i]=alignment.invariant_codes[i][keep].copy()
        else:
            alignment.chunks[i]=None;alignment.invariant_codes[i]=None
    alignment.chunks=[c for c in alignment.chunks if c is not None]
    alignment.invariant_codes=[c for c in alignment.invariant_codes if c is not None]
    if sum(int(w.sum()) for x,w in alignment.chunks)!=alignment.original_sites:raise ArithmeticError('global equality lost site multiplicity')
    stats.update(seconds=time.perf_counter()-start,fold_seconds=fold_seconds,original_chunks=old,
        retained_chunks=len(alignment.chunks),host_pattern_bytes=sum(x.nbytes+w.nbytes for x,w in alignment.chunks),
        input_storage_released_before_fold=True)
    return stats
