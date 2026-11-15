"""Exact unrooted backbone constraints, with linear work/state per NNI sweep.

No scientific dependencies are imported here: upload validation uses the same
bundled Newick parser as inference, without importing CUDA into the web process.
"""
from collections import deque
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys


@lru_cache(maxsize=1)
def treeio():
    path = Path(__file__).resolve().parent / 'runtime/gene/src/concordtree/treeio.py'
    spec = importlib.util.spec_from_file_location('_ct_backbone_treeio', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def restricted_graph(graph, names, keep):
    """Prune other leaves, then suppress degree-two nodes; O(N), iterative."""
    g = {u: set(vs) for u, vs in graph.items()}
    anchors = {u for u, name in names.items() if name in keep}
    pending = deque(u for u in g if len(g[u]) <= 1 and u not in anchors)
    while pending:
        u = pending.popleft()
        if u not in g:
            continue
        for v in g.pop(u):
            g[v].remove(u)
            if len(g[v]) <= 1 and v not in anchors:
                pending.append(v)
    pending = deque(u for u in g if u not in anchors and len(g[u]) == 2)
    while pending:
        u = pending.popleft()
        if u not in g or len(g[u]) != 2:
            continue
        a, b = g.pop(u)
        g[a].remove(u); g[b].remove(u)
        g[a].add(b); g[b].add(a)
    return g, {u: names[u] for u in anchors}


def _signature(graph, names, interner):
    # Root at a named leaf, shared between both trees. Integer interning avoids
    # recursion, growing Newick strings, and O(N^2) split bitsets on long trees.
    root = min(names, key=names.get)
    parent = {root: None}; order = [root]
    for u in order:
        for v in graph[u]:
            if v != parent[u]:
                if v in parent:
                    raise ValueError('Backbone tree has a cycle.')
                parent[v] = u; order.append(v)
    if len(order) != len(graph):
        raise ValueError('Backbone tree is disconnected.')
    codes = {}
    for u in reversed(order):
        children = tuple(sorted(codes[v] for v in graph[u] if parent.get(v) == u))
        key = (names.get(u), children)
        codes[u] = interner.setdefault(key, len(interner))
    return codes[root]


class Backbone:
    """A fully resolved unrooted constraint; input rooting/lengths are ignored."""
    def __init__(self, text, alignment_names):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Backbone must be a nonempty Newick tree.')
        # The bundled parser supports comments and quoted names. Reject lexical
        # gaps (which that parser otherwise ignores) before structural parsing.
        import re
        pattern = re.compile(r"\s*(\[[^\]]*\]|'(?:[^']|'')*'|[(),:;]|[^\s(),:;\[\]']+)")
        pos = 0
        for match in pattern.finditer(text):
            if text[pos:match.start()].strip():
                raise ValueError('Malformed Newick backbone.')
            pos = match.end()
        if text[pos:].strip():
            raise ValueError('Malformed Newick backbone.')
        self.tree = treeio().parse_newick(text)
        self.tree.require_binary()
        self.taxa = frozenset(self.tree.names.values())
        missing = self.taxa - set(alignment_names)
        if missing:
            raise ValueError('Backbone taxa missing from alignment: ' + ', '.join(sorted(missing)[:8]))
        self.sha256 = hashlib.sha256(text.encode()).hexdigest()
        self.alignment_names = tuple(alignment_names)
        self.metadata = dict(enabled=True, kind='unrooted_binary', taxa=len(self.taxa),
            added_taxa=len(alignment_names)-len(self.taxa), sha256=self.sha256,
            branch_lengths_fixed=False, root_preserved=False)

    def assert_tree(self, tree):
        tree.require_binary()
        if set(tree.names.values()) != set(self.alignment_names):
            raise RuntimeError('Backbone completion changed the alignment taxon set.')
        graph, names = restricted_graph(tree.graph, tree.names, self.taxa)
        intern = {}
        if _signature(self.tree.graph, self.tree.names, intern) != _signature(graph, names, intern):
            raise RuntimeError('Backbone topology invariant violated.')

    def allowed_edges(self, graph, names, edges):
        """NNI changes the backbone iff all four incident groups have anchors.

        Count anchors in one postorder traversal. Any alternative on an edge
        with an empty incident group is safe. Noninteracting batches retain
        this property. Callers must retain their existing conflict scheduling.
        """
        if not isinstance(graph, dict):
            graph = {u: [int(v) for v in row if v >= 0] for u, row in enumerate(graph)}
        anchors = {u for u, name in names.items() if name in self.taxa}
        root = next(iter(graph)); parent = {root: None}; order = [root]
        for u in order:
            for v in graph[u]:
                if v != parent[u]:
                    if v in parent:
                        raise RuntimeError('Cycle in constrained NNI tree.')
                    parent[v] = u; order.append(v)
        if len(order) != len(graph):
            raise RuntimeError('Disconnected constrained NNI tree.')
        count = {}
        for u in reversed(order):
            count[u] = int(u in anchors) + sum(count[v] for v in graph[u] if parent.get(v) == u)
        total = len(anchors)
        def outward(u, v):
            return count[v] if parent.get(v) == u else total-count[u]
        return [not all(outward(int(u), v) > 0 for u, other in ((a,b),(b,a))
                        for v in graph[int(u)] if v != other) for a,b in edges]

    def complete(self, guide, names):
        """Transfer guide-only subtrees onto the supplied backbone in O(N).

        Maximal missing-taxon components retain their inferred guide topology.
        A multisource walk through the guide core assigns each component to a
        nearest backbone leaf. Constrained distance/MLP/posterior NNI then search
        its attachment. This seed is a heuristic, not optimal placement.
        """
        guide.require_binary()
        anchors = {u for u, name in guide.names.items() if name in self.taxa}
        core = {u: set(vs) for u, vs in guide.graph.items()}
        pending = deque(u for u in core if len(core[u]) == 1 and u not in anchors)
        while pending:
            u = pending.popleft()
            if u not in core:
                continue
            for v in core.pop(u):
                core[v].remove(u)
                if len(core[v]) == 1 and v not in anchors:
                    pending.append(v)
        owners = {u: guide.names[u] for u in anchors}
        pending = deque(u for u in guide.names if u in anchors)
        while pending:
            u = pending.popleft()
            for v in sorted(core[u]):
                if v not in owners:
                    owners[v] = owners[u]; pending.append(v)
        # Keep guide IDs for transferred components, assign distinct IDs to the
        # supplied backbone, and compact once after grafting all components.
        next_id = max(guide.graph)+1
        mapping = {u: next_id+i for i,u in enumerate(self.tree.graph)}
        next_id += len(mapping)
        g = {mapping[u]: {mapping[v] for v in vs} for u,vs in self.tree.graph.items()}
        labels = {mapping[u]: name for u,name in self.tree.names.items()}
        leaf_id = {name: u for u,name in labels.items()}
        removed = set(guide.graph)-set(core)
        for u in removed:
            g[u] = set(guide.graph[u]) & removed
            if u in guide.names:
                labels[u] = guide.names[u]
        for u in core:
            for v in sorted(guide.graph[u]-core[u]):
                anchor = leaf_id[owners[u]]; neighbor = next(iter(g[anchor]))
                center = next_id; next_id += 1
                g[anchor].remove(neighbor); g[neighbor].remove(anchor)
                g[center] = {anchor, neighbor, v}
                g[anchor].add(center); g[neighbor].add(center); g[v].add(center)
        lookup={name:i for i,name in enumerate(names)}
        mapping={u:lookup[name] for u,name in labels.items()}
        mapping.update({u:len(names)+i for i,u in enumerate(u for u in g if u not in labels)})
        tree=type(guide)({mapping[u]:{mapping[v] for v in vs} for u,vs in g.items()},dict(enumerate(names)),{})
        self.assert_tree(tree)
        return tree
