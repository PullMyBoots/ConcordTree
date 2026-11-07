"""Species topology serializer with bounded malformed-graph rejection."""

from concordtree._core.scaleqf import _newick_label

def graph_to_newick(
    adjacency: dict[int, set[int]], names: list[str], n_taxa: int, annotations=None
) -> str:
    if n_taxa<2 or len(names)!=n_taxa:raise ValueError('invalid taxon name set')
    degree_two = sorted(
        node for node in adjacency if node >= n_taxa and len(adjacency[node]) == 2
    )
    internal = sorted(node for node in adjacency if node >= n_taxa)
    if not internal:
        raise ValueError("Topology has no internal node")
    root = degree_two[0] if degree_two else internal[0]

    output: list[str] = []
    visited=set();leaves=0
    stack: list[tuple[str, int | str, int | None]] = [("node", root, None)]
    while stack:
        kind, value, parent = stack.pop()
        if kind == "text":
            output.append(str(value))
            continue
        node = int(value)
        if node in visited:raise ValueError('cycle in species topology')
        if node<0 or node not in adjacency:raise ValueError('dangling species topology edge')
        if parent is not None and parent not in adjacency[node]:raise ValueError('asymmetric species topology')
        visited.add(node)
        if node < n_taxa:
            if len(adjacency[node])!=1:raise ValueError('invalid species leaf degree')
            leaves+=1
            output.append(_newick_label(names[node]))
            continue
        children = sorted(x for x in adjacency[node] if x != parent)
        if not children:
            raise ValueError(f"Internal node {node} became empty")
        output.append("(")
        label='' if parent is None else (annotations or {}).get(tuple(sorted((node,parent))), '')
        stack.append(("text", ")"+str(label), None))
        for index in range(len(children) - 1, -1, -1):
            stack.append(("node", children[index], node))
            if index > 0:
                stack.append(("text", ",", None))
    if len(visited)!=len(adjacency) or leaves!=n_taxa:raise ValueError('disconnected species topology')
    return "".join(output) + ";"
