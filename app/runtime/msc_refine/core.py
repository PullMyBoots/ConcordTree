"""Exact aggregate CASTER-site quartet objective and topology-only NNI.

Statistic attribution: Zhang, Nielsen & Mirarab, Science 2025, CASTER;
https://github.com/chaoszhang/ASTER/blob/master/src/sequence.hpp, quadPos.
This re-expression is not a new consistency theorem. Its model assumptions
and local-search limitations are recorded in METHOD.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np


def pair_contrast(x, y, pi):
    """Purine / pyrimidine pair contrasts; final axis is A,C,G,T."""
    a, c, g, t = (pi[i] for i in range(4))
    xr, yr = x[..., 0] + x[..., 2], y[..., 0] + y[..., 2]
    xy, yy = x[..., 1] + x[..., 3], y[..., 1] + y[..., 3]
    r = (a*a+g*g)*xr*yr - (a+g)**2*(x[..., 0]*y[..., 0]+x[..., 2]*y[..., 2])
    q = (c*c+t*t)*xy*yy - (c+t)**2*(x[..., 1]*y[..., 1]+x[..., 3]*y[..., 3])
    return r, q


def quartet_scores(counts, pi):
    """Shape (...,4 groups,4 bases) -> (...,3), larger is stronger."""
    scores = []
    for a, b, c, d in ((0,1,2,3), (0,2,1,3), (0,3,1,2)):
        r1, y1 = pair_contrast(counts[..., a, :], counts[..., b, :], pi)
        r2, y2 = pair_contrast(counts[..., c, :], counts[..., d, :], pi)
        scores.append(r1*y2+y1*r2)
    return np.stack(scores, axis=-1)


@dataclass
class EdgePlan:
    leaf_order: np.ndarray
    edges: np.ndarray
    branches: np.ndarray
    intervals: np.ndarray
    complement: np.ndarray
    sizes: np.ndarray


def compile_plan(graph, n):
    """DFS leaf intervals for all four groups incident to every internal edge.

    Work and state O(N), including on caterpillar trees. No dense distance
    matrix, repeated clade walks, or recursion. Leaf IDs must be 0..N-1.
    """
    if n < 4 or set(graph) != set(range(2*n-2)):
        raise ValueError('requires contiguous leaf/internal binary tree IDs')
    if any(len(v) != (1 if u<n else 3) for u,v in graph.items()):
        raise ValueError('requires fully binary unrooted tree')
    root = n
    parent = {root:-1}; pending=[(root,False)]; left={}; right={}; leaves=[]
    while pending:
        u, done = pending.pop()
        if done:
            right[u] = len(leaves)
            continue
        left[u] = len(leaves)
        if u<n:
            leaves.append(u); right[u]=len(leaves)
            continue
        pending.append((u,True))
        for v in sorted(graph[u], reverse=True):
            if v == parent[u]:
                continue
            if v in parent:
                raise ValueError('cycle')
            parent[v]=u; pending.append((v,False))
    if len(leaves)!=n or len(parent)!=2*n-2:
        raise ValueError('disconnected tree')
    edges=[];branches=[];intervals=[];complement=[];sizes=[]
    for u in range(n,2*n-2):
        for v in sorted(graph[u]):
            if v<=u or v<n:
                continue
            parts=sorted(graph[u]-{v})+sorted(graph[v]-{u})
            centers=[u,u,v,v];ints=[];comp=[];sz=[]
            for center,neighbor in zip(centers,parts):
                inverted = parent[center]==neighbor
                node = center if inverted else neighbor
                ints.append((left[node],right[node]));comp.append(inverted)
                count=right[node]-left[node];sz.append(n-count if inverted else count)
            edges.append((u,v));branches.append(parts);intervals.append(ints)
            complement.append(comp);sizes.append(sz)
    assert len(edges)==n-3
    return EdgePlan(np.array(leaves,dtype=np.int64),np.array(edges,dtype=np.int64),
                    np.array(branches,dtype=np.int64),np.array(intervals,dtype=np.int64),
                    np.array(complement,dtype=bool),np.array(sizes,dtype=np.int64))


def cpu_edge_scores(states, plan, pi):
    x = np.eye(5,dtype=np.float64)[states][..., :4]
    prefix = np.concatenate((np.zeros((1,states.shape[1],4)),x[plan.leaf_order].cumsum(0)),0)
    c = prefix[plan.intervals[:,:,1]]-prefix[plan.intervals[:,:,0]]
    c = np.where(plan.complement[:,:,None,None],prefix[-1]-c,c)
    return quartet_scores(c.transpose(0,2,1,3),pi).sum(1)


def apply_move(graph, edge, branches, choice):
    u,v=map(int,edge);b=int(branches[1]);c=int(branches[2 if choice==1 else 3])
    graph[u].remove(b);graph[b].remove(u);graph[v].remove(c);graph[c].remove(v)
    graph[u].add(c);graph[c].add(u);graph[v].add(b);graph[b].add(v)


def select_moves(graph, plan, scores):
    """Positive exact deltas, deterministic noninteracting radius-one packing.

    This exclusion is a dependency schedule, not a statistical confidence gate.
    Nearby edges are rescored next sweep. No user-facing margin parameter.
    """
    best=scores.argmax(1);gain=scores[np.arange(len(scores)),best]-scores[:,0]
    blocked=set();chosen=[]
    for i in sorted(range(len(scores)),key=lambda i:(-float(gain[i]),tuple(plan.edges[i]))):
        u,v=map(int,plan.edges[i])
        if best[i]==0 or gain[i]<=0 or u in blocked or v in blocked:
            continue
        chosen.append((i,int(best[i]),float(gain[i])))
        blocked.update((u,v));blocked.update(graph[u]);blocked.update(graph[v])
    return chosen


def brute_objective(graph, states, pi):
    """Independent O(N^4) correctness oracle, ONLY for tiny unit tests."""
    n=len(states);distance=np.zeros((n,n),dtype=int)
    for i in range(n):
        stack=[(i,-1,0)]
        while stack:
            u,p,d=stack.pop()
            if u<n:distance[i,u]=d
            stack.extend((v,u,d+1) for v in graph[u] if v!=p)
    result=0.
    eye=np.eye(5,dtype=np.float64)[:,:4]
    for a,b,c,d in itertools.combinations(range(n),4):
        lengths=[distance[a,b]+distance[c,d],distance[a,c]+distance[b,d],distance[a,d]+distance[b,c]]
        code=states[[a,b,c,d]].T
        result+=quartet_scores(eye[code],pi)[:,int(np.argmin(lengths))].sum()
    return float(result)
