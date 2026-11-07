"""Exact bounded-memory observed-base counts, without NumPy's int64 copy."""
import numpy as np
import json
from pathlib import Path
from . import _msc_plan


def byte_histogram(packed):
    return _msc_plan.byte_histogram(packed)


def empirical_pi(packed,length):
    return region_frequencies(packed,0,length)


def region_frequencies(packed,start,end):
    histogram=byte_histogram(packed[:,start//2:(end+1)//2]);codes=np.arange(256)
    counts=np.array([histogram[(codes&15)==b].sum()+histogram[(codes>>4)==b].sum()
                     for b in range(4)],dtype=np.float64)
    if start%2:
        low=packed[:,start//2]&15
        counts-=np.array([(low==b).sum() for b in range(4)])
    if end%2:
        high=packed[:,end//2]>>4
        counts-=np.array([(high==b).sum() for b in range(4)])
    total=counts.sum()
    return counts/total if total else np.zeros(4)


def read_boundaries(path,length):
    """Project only observable ordered alignment_length values from the input."""
    regions=[];start=0
    with Path(path).open() as f:
        for line in f:
            value=json.loads(line)['alignment_length']
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:
                raise ValueError('positive integer observed alignment_length required')
            end=start+value
            if end>length:raise ValueError('observed boundaries exceed MSA length')
            regions.append((start,end));start=end
    if start!=length:raise ValueError('observed boundaries must exactly cover MSA')
    return regions
