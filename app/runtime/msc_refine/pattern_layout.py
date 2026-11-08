"""Canonical equality-pattern identities for the production kernel."""
import itertools
import numpy as np

def canonical_pattern(pattern):
    seen={};result=[]
    for x in pattern:
        if x not in seen:seen[x]=len(seen)
        result.append(seen[x])
    return tuple(result)

def pattern_layout():
    all_patterns=list(itertools.product(range(4),repeat=4))
    representatives=sorted(set(canonical_pattern(p) for p in all_patterns))
    index={p:i for i,p in enumerate(representatives)}
    mapping=np.array([[index[canonical_pattern(tuple(p[i] for i in order))] for p in all_patterns]
                     for order in ((0,1,2,3),(0,2,1,3),(0,3,1,2))])
    degeneracy=np.bincount(mapping[0],minlength=15)
    assert len(representatives)==15 and representatives[0]==(0,0,0,0)
    return np.array(representatives),mapping,degeneracy
