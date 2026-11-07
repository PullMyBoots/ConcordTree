"""Exact named unrooted topology identity, O(N) tokens, no probabilistic hash.

Root at the neighbour of the lexicographically first leaf. Order disjoint child
subtrees by their smallest named leaf. Emit a uniquely decodable bracket/varint
stream. Internal IDs, adjacency insertion order and branch lengths are ignored.
Python's byte-set handles hash collisions by full byte equality.
"""


def topology_key(tree):
    names = sorted(tree.names.values())
    if len(names) != len(set(names)):
        raise ValueError('unique leaf names required')
    rank = {name: i for i, name in enumerate(names)}
    anchor = next(u for u, name in tree.names.items() if name == names[0])
    root, = tree.graph[anchor]
    parent = {root: None}
    order = [root]
    for u in order:
        for v in tree.graph[u]:
            if v != parent[u]:
                if v in parent:
                    raise ValueError('cycle or asymmetric edge in topology identity')
                parent[v] = u
                order.append(v)
    if len(parent)!=len(tree.graph):
        raise ValueError('disconnected topology identity')
    minimum = {}
    children = {}
    for u in reversed(order):
        child = [v for v in tree.graph[u] if parent.get(v) == u]
        if u in tree.names:
            minimum[u] = rank[tree.names[u]]
        else:
            child.sort(key=minimum.__getitem__)
            minimum[u] = minimum[child[0]]
        children[u] = child
    out = bytearray()
    pending = [root]
    while pending:
        u = pending.pop()
        if u is None:
            out.append(1)  # close an internal node
        elif u in tree.names:
            out.append(2)  # leaf, followed by a self-delimiting unsigned varint
            value = rank[tree.names[u]]
            while value >= 128:
                out.append((value & 127) | 128)
                value >>= 7
            out.append(value)
        else:
            out.append(0)
            pending.append(None)
            pending.extend(reversed(children[u]))
    # Include names: topology equality is named equality, even across inputs.
    return tuple(names), bytes(out)
