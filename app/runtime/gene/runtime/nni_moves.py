"""Gene NNI proposal application and conflict scheduling."""
from concordtree.treeio import edge

def apply_proposals(tree, proposals):
    out = tree.copy()
    for p in proposals:
        u, v = p['edge']
        a, b, c, d = p['branches']
        x = c if p['alternative'] == 1 else d
        lb, lx = out.lengths.pop(edge(u, b)), out.lengths.pop(edge(v, x))
        out.graph[u].remove(b); out.graph[b].remove(u)
        out.graph[v].remove(x); out.graph[x].remove(v)
        out.graph[u].add(x); out.graph[x].add(u)
        out.graph[v].add(b); out.graph[b].add(v)
        out.lengths[edge(u, x)], out.lengths[edge(v, b)] = lx, lb
        out.lengths[edge(u, v)] = p['length']
    return out

def independent_subset(tree, proposals):
    chosen, blocked = [], set()
    for p in proposals:
        u, v = p['edge']
        if u in blocked or v in blocked:
            continue
        chosen.append(p)
        blocked.update({u, v} | tree.graph[u] | tree.graph[v])
    return chosen
