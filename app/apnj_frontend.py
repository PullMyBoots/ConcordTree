"""Full-site bounded AP-NJ initialization for the unified gene/species engine."""
import numbers
import time
import numpy as np
from apnj_core import build

MAX_TAXA=(2**31-1)//273  # Largest frozen sparse pool has at most273 entries/taxon.
MAX_SITES=2**31-17

def initialize(names,packed_data,*,task,seed,cpu_threads=16,options=None,cleanup_options=None,checkpoint=None):
    if task not in ('gene','species'):raise ValueError('task must be gene or species')
    if packed_data is None or len(packed_data)!=3:raise ValueError('AP-NJ requires the original packed alignment')
    packed,raw_names,sites=packed_data;names=list(names)
    if names!=list(raw_names):raise ValueError('packed alignment taxon order mismatch')
    if not 4<=len(names)<=MAX_TAXA:raise ValueError(f'AP-NJ requires4..{MAX_TAXA} taxa in its native integer domain')
    if any(not isinstance(x,str) or not x for x in names):raise ValueError('taxon names must be nonempty strings')
    ordered=sorted(names)
    if any(x==y for x,y in zip(ordered,ordered[1:])):raise ValueError('duplicate taxon names')
    if not isinstance(sites,numbers.Integral) or isinstance(sites,bool) or not 1<=sites<=MAX_SITES:raise ValueError('site count is outside the supported positive integer domain')
    if not isinstance(packed,np.ndarray) or packed.dtype!=np.uint8 or not packed.flags.c_contiguous or packed.ndim!=2 or packed.shape!=(len(names),(int(sites)+1)//2):raise ValueError('contiguous uint8 packed original observations required')
    if not isinstance(seed,numbers.Integral) or isinstance(seed,bool) or not 0<=seed<2**32:raise ValueError('seed must be an unsigned32-bit integer')
    if not isinstance(cpu_threads,numbers.Integral) or isinstance(cpu_threads,bool) or cpu_threads<1:raise ValueError('positive integer CPU thread count required')
    workers=min(16,int(cpu_threads));started=time.perf_counter()
    options=dict(options or {});options.pop('seed',None)
    distance_model=options.pop('distance_model',3);options.setdefault('minimum_divisor',514)
    tree,meta=build(names,packed,int(sites),seed=int(seed),real_ann=True,row_control=True,common=True,
                    model=distance_model,canonical=True,cpu_threads=workers,cleanup_options=cleanup_options,checkpoint_fn=checkpoint,**options)
    meta.update(canonicalization_seconds=meta['export_seconds'],constructor_seconds=time.perf_counter()-started,
                constructor=meta['method'],backend='native packed AP-NJ',execution='compact integer keys and exact owner-local walk deduplication',sampled_sites=int(sites),all_sites=int(sites),
                full_site_input=True,requested_cpu_threads=int(cpu_threads),cpu_threads=workers)
    return tree,meta
