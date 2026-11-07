"""DIPPER placement initializer sharing the release's original packed MSA."""
from pathlib import Path
import hashlib
import time
from functools import lru_cache
import numpy as np

UPSTREAM_COMMIT = 'd80df351107f05ea2ac01076709defc4ab9d702e'

@lru_cache(maxsize=1)
def backend_digest():
    import _dipper_native
    return hashlib.sha256(Path(_dipper_native.__file__).read_bytes()).hexdigest()

def canonical_topology(graph, names, original_row_for_leaf):
    """Stable leaf and internal numbering for the existing MLP-NNI search."""
    from concordtree.treeio import Tree
    n = len(names)
    # Upstream starts from two leaves joined through a degree-two root.
    for u in list(graph):
        if u >= n and len(graph[u]) == 2:
            a, b = graph.pop(u)
            if b in graph[a]: raise RuntimeError('DIPPER returned a cyclic topology')
            graph[a].remove(u); graph[b].remove(u)
            graph[a].add(b); graph[b].add(a)
    leaf_rows = {i: int(row) for i, row in enumerate(original_row_for_leaf)}
    if sorted(leaf_rows.values()) != list(range(n)):
        raise RuntimeError('DIPPER returned an invalid leaf permutation')
    root = next(u for u, row in leaf_rows.items() if row == 0)
    parent = {root: -1}; order = [root]
    for u in order:
        for v in graph[u]:
            if v != parent[u]:
                if v in parent: raise RuntimeError('DIPPER returned a cyclic topology')
                parent[v] = u; order.append(v)
    if len(order) != len(graph): raise RuntimeError('DIPPER returned a disconnected topology')
    minimum = dict(leaf_rows)
    for u in reversed(order):
        if u not in minimum:
            minimum[u] = min(minimum[v] for v in graph[u] if v != parent[u])
    stack = [root]; relabel = {}; internal = n
    while stack:
        u = stack.pop()
        if u in leaf_rows: relabel[u] = leaf_rows[u]
        else: relabel[u] = internal; internal += 1
        stack.extend(sorted((v for v in graph[u] if v != parent[u]), key=minimum.__getitem__, reverse=True))
    tree = Tree({relabel[u]: {relabel[v] for v in adjacent} for u, adjacent in graph.items()}, dict(enumerate(names)))
    tree.require_binary()
    if len(tree.graph) != 2 * n - 2: raise RuntimeError('DIPPER topology node count mismatch')
    return tree

def build_dipper(names, packed_data, *, seed, cpu_threads=16):
    import _dipper_native
    if packed_data is None: raise ValueError('DIPPER requires the original packed alignment')
    packed, raw_names, sites = packed_data
    if list(raw_names) != list(names): raise ValueError('packed MSA taxon order mismatch')
    if not isinstance(packed, np.ndarray) or packed.dtype != np.uint8 or not packed.flags.c_contiguous:
        raise ValueError('DIPPER requires contiguous uint8 packed observations')
    if packed.ndim != 2 or len(set(names)) != len(names) or len(names) != packed.shape[0]:
        raise ValueError('DIPPER taxon names must be unique and match the packed rows')
    start = time.perf_counter()
    result = _dipper_native.build(packed, int(sites), int(seed), int(cpu_threads))
    edges = result.pop('edges'); permutation = result.pop('original_row_for_leaf')
    graph = {}
    for u, v in edges.tolist():
        graph.setdefault(u, set()).add(v); graph.setdefault(v, set()).add(u)
    tree = canonical_topology(graph, names, permutation)
    result.update(constructor='DIPPER GPU k-closest placement', backend='in-process packed native',
                  upstream_commit=UPSTREAM_COMMIT, seed=int(seed), all_sites=int(sites),
                  native_sha256=backend_digest(),
                  taxa=len(names), distance='upstream JC; union of usable sites',
                  sampled_sites=int(sites), full_site_input=True,
                  host_encoding='existing packed nibbles; row padding and deterministic shuffle only',
                  constructor_seconds=time.perf_counter() - start)
    return tree, result
