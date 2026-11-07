"""Strict, iterative Newick graph utilities; branch lengths may be absent."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import math
import re

def edge(a, b):
    return (min(a, b), max(a, b))

@dataclass
class Tree:
    graph: dict[int, set[int]]
    names: dict[int, str]
    lengths: dict[tuple[int, int], float | None] = field(default_factory=dict)

    def copy(self):
        return Tree({u:set(v) for u,v in self.graph.items()}, self.names.copy(), self.lengths.copy())

    def require_binary(self):
        if len(self.names) < 4 or any(len(v) != (1 if u in self.names else 3)
                                     for u,v in self.graph.items()):
            raise ValueError("requires an unrooted binary tree with at least 4 taxa")
        if len(self.graph)!=2*len(self.names)-2 or not set(self.names).issubset(self.graph):
            raise ValueError('invalid binary tree node or leaf set')
        visited=set();pending=[next(iter(self.graph))]
        while pending:
            u=pending.pop()
            if u in visited:continue
            visited.add(u)
            for v in self.graph[u]:
                if v not in self.graph or u not in self.graph[v]:
                    raise ValueError('asymmetric or dangling tree edge')
                if v not in visited:pending.append(v)
        if len(visited)!=len(self.graph):raise ValueError('disconnected binary tree')

def parse_newick(text: str) -> Tree:
    # Comments are metadata, not part of names. Preserve quoted names.
    token = re.compile(r"\s*(\[[^\]]*\]|'(?:[^']|'')*'|[(),:;]|[^\s(),:;\[\]]+)")
    tokens = [m.group(1) for m in token.finditer(text)]
    tokens = [t for t in tokens if not t.startswith("[")]
    if not tokens or tokens[-1] != ";" or tokens.count(";") != 1:
        raise ValueError("expected exactly one semicolon-terminated Newick tree")
    graph, names, lengths, stack = {}, {}, {}, []
    current = None
    root = None
    expecting = True
    i = 0
    while i < len(tokens)-1:
        t=tokens[i]; i+=1
        if t == "(":
            if not expecting: raise ValueError("unexpected '('")
            u=len(graph); graph[u]=set()
            if stack:
                graph[u].add(stack[-1]); graph[stack[-1]].add(u); lengths[edge(u,stack[-1])]=None
            else:
                if root is not None: raise ValueError("multiple roots")
                root=u
            stack.append(u); current=None; expecting=True
        elif t == ",":
            if expecting or not stack: raise ValueError("unexpected comma")
            expecting=True; current=None
        elif t == ")":
            if expecting or not stack: raise ValueError("unexpected ')'")
            current=stack.pop(); expecting=False
        elif t == ":":
            if current is None or i >= len(tokens)-1: raise ValueError("unexpected length")
            try: value=float(tokens[i])
            except ValueError as e: raise ValueError("invalid length") from e
            i+=1
            if not math.isfinite(value) or value < 0: raise ValueError("length must be finite and nonnegative")
            if stack: lengths[edge(current,stack[-1])]=value
        else:
            label=t[1:-1].replace("''","'") if t.startswith("'") else t
            if expecting:
                if not stack: raise ValueError("tree must be parenthesized")
                u=len(graph); graph[u]={stack[-1]}; graph[stack[-1]].add(u)
                lengths[edge(u,stack[-1])]=None; names[u]=label; current=u; expecting=False
            elif current in names or current is None:
                raise ValueError("unexpected label")
            # otherwise an internal support/name; do not treat as taxon
    if stack or expecting or root is None: raise ValueError("incomplete Newick")
    if len(set(names.values())) != len(names): raise ValueError("duplicate taxon names")
    # Suppress root of degree two, preserving path lengths only if both known.
    if len(graph[root]) == 2:
        a,b=sorted(graph[root]); x=lengths.pop(edge(a,root)); y=lengths.pop(edge(b,root))
        graph[a].remove(root); graph[b].remove(root); del graph[root]
        graph[a].add(b); graph[b].add(a)
        lengths[edge(a,b)] = None if x is None or y is None else x+y
    if len(names)<2: raise ValueError("too few taxa")
    return Tree(graph,names,lengths)

def quote_name(name):
    return "'" + name.replace("'","''") + "'" if re.search(r"[\s(),:;\[\]']",name) else name

def newick_python(tree: Tree, annotations=None) -> str:
    annotations=annotations or {}
    if not tree.graph:raise ValueError('empty Newick graph')
    root=next((u for u in tree.graph if u not in tree.names), next(iter(tree.graph)))
    parent={root:-1}; order=[root]
    for u in order:
        for v in sorted(tree.graph[u]):
            if v not in tree.graph or u not in tree.graph[v]:
                raise ValueError('asymmetric or dangling Newick edge')
            if v!=parent[u]:
                if v in parent:raise ValueError('cycle or asymmetric Newick graph')
                parent[v]=u; order.append(v)
    if len(parent)!=len(tree.graph):raise ValueError('disconnected Newick graph')
    tokens=[]; pending=[("node",root)]
    while pending:
        kind,value=pending.pop()
        if kind=="text": tokens.append(value); continue
        u=value; children=[v for v in sorted(tree.graph[u]) if v!=parent[u]]
        tail=quote_name(tree.names[u]) if u in tree.names else ""
        if parent[u]>=0:
            e=edge(u,parent[u])
            if u not in tree.names and e in annotations: tail+=str(annotations[e])
            length=tree.lengths.get(e)
            if length is not None: tail+=":"+format(float(length),".10g")
        pending.append(("text",tail))
        if children:
            pending.append(("text",")"))
            for j,v in reversed(list(enumerate(children))):
                pending.append(("node",v))
                if j>0: pending.append(("text",","))
            pending.append(("text","("))
    return "".join(tokens)+";"

def newick(tree: Tree, annotations=None) -> str:
    from _tree_plan_native import newick as native_newick
    return native_newick(tree.graph,tree.names,tree.lengths,annotations or {})

def rooted(tree):
    if not tree.graph:raise ValueError('empty tree graph')
    root=next((u for u in tree.graph if u not in tree.names),next(iter(tree.graph)))
    parent={root:-1}; order=[root]; depth={root:0}
    for u in order:
        for v in sorted(tree.graph[u]):
            if v not in tree.graph or u not in tree.graph[v]:
                raise ValueError('asymmetric or dangling tree edge')
            if v!=parent[u]:
                if v in parent:raise ValueError('cycle in tree graph')
                parent[v]=u; order.append(v); depth[v]=depth[u]+1
    if len(parent)!=len(tree.graph):raise ValueError('disconnected tree graph')
    return root,parent,order,depth

def split_map(tree, name_order=None):
    names=sorted(tree.names.values()) if name_order is None else list(name_order)
    if set(names)!=set(tree.names.values()): raise ValueError("exact taxon set mismatch")
    lookup={n:i for i,n in enumerate(names)}; full=(1<<len(names))-1
    root,parent,order,_=rooted(tree); masks={}; out={}
    for u in reversed(order):
        mask=(1<<lookup[tree.names[u]]) if u in tree.names else 0
        for v in tree.graph[u]:
            if parent.get(v)==u: mask|=masks[v]
        masks[u]=mask
        if u!=root and 1<mask.bit_count()<len(names)-1:
            out[edge(u,parent[u])]=min(mask,full^mask)
    return out

def nrf(actual, expected):
    a=set(split_map(actual).values()); b=set(split_map(expected).values())
    if set(actual.names.values())!=set(expected.names.values()): raise ValueError("taxa mismatch")
    return len(a^b)/(len(a)+len(b)) if a or b else 0.0

def edge_quartets(tree, witnesses=8):
    """O(N K log K) bounded-nearest representative DP, no all-pairs matrix."""
    if witnesses<1: raise ValueError("witnesses must be positive")
    tree.require_binary()
    root,parent,order,_=rooted(tree); reps={}
    def merge(parts):
        return sorted((d+1,t) for part in parts for d,t in part)[:witnesses]
    for u in reversed(order):
        if u in tree.names: reps[u,parent[u]]=[(0,u)]
        elif parent[u]>=0:
            reps[u,parent[u]]=merge([reps[v,u] for v in sorted(tree.graph[u]) if v!=parent[u]])
    for u in order:
        for v in sorted(tree.graph[u]):
            if parent.get(v)==u:
                reps[u,v]=merge([reps[w,u] for w in sorted(tree.graph[u]) if w!=v])
    edges=[]; groups=[]
    for u in sorted(tree.graph):
        if u in tree.names: continue
        for v in sorted(tree.graph[u]):
            if v<=u or v in tree.names: continue
            ends=[(w,u) for w in sorted(tree.graph[u]-{v})]+[(w,v) for w in sorted(tree.graph[v]-{u})]
            choices=[[t for _,t in reps[e]] for e in ends]
            seen=set()
            for k in range(witnesses):
                # Distinct components guarantee distinct taxa. Mixed strides
                # prevent repeated witnesses on two-singleton/two-large edges.
                q=tuple(c[(k*(2*j+1)+k//max(1,len(c)))%len(c)]
                        for j,c in enumerate(choices))
                if q not in seen: edges.append((u,v)); groups.append(q); seen.add(q)
    return edges,groups
