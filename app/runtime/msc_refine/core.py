"""Shared binary-tree plans, conflict selection and NNI actions.

The Python plan builder is a small independent reference for native plans.
No sequence statistic or alternative scoring method is defined here.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np






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


