"""Topology-oracle utilities for the EAPC candidate-reachability preflight.

These routines are diagnostics, not the production EAPC search.  They make
single-leaf detach/reattach moves, compare them with split-based normalized RF,
and expose bounded candidate policies without writing to the immutable inputs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable


Edge = tuple[int, int]


def canonical_edge(left: int, right: int) -> Edge:
    return (left, right) if left < right else (right, left)


def clone_graph(adjacency: dict[int, set[int]]) -> dict[int, set[int]]:
    return {node: set(neighbors) for node, neighbors in adjacency.items()}


def graph_edges(adjacency: dict[int, set[int]]) -> list[Edge]:
    return sorted(
        canonical_edge(left, right)
        for left, neighbors in adjacency.items()
        for right in neighbors
        if left < right
    )


def detach_leaf(
    adjacency: dict[int, set[int]], leaf: int, n_taxa: int
) -> tuple[dict[int, set[int]], Edge | None]:
    """Detach one leaf and suppress its degree-two internal parent.

    The returned edge is the backbone edge on which reattaching the leaf
    exactly recreates the input topology.
    """

    graph = clone_graph(adjacency)
    if leaf not in graph or len(graph[leaf]) != 1:
        raise ValueError(f"Leaf {leaf} is absent or not pendant")
    parent = next(iter(graph[leaf]))
    graph[parent].remove(leaf)
    del graph[leaf]
    if parent < n_taxa or len(graph[parent]) < 2:
        raise ValueError(
            f"Detaching leaf {leaf} left an invalid parent of degree {len(graph[parent])}"
        )
    # Preserve pre-existing unresolved multifurcations rather than inventing a
    # suppression or binary resolution before candidate search.
    if len(graph[parent]) > 2:
        return graph, None
    left, right = sorted(graph[parent])
    graph[left].remove(parent)
    graph[right].remove(parent)
    graph[left].add(right)
    graph[right].add(left)
    del graph[parent]
    return graph, canonical_edge(left, right)


def reattach_leaf(
    backbone: dict[int, set[int]], leaf: int, edge: Edge
) -> dict[int, set[int]]:
    """Insert one new degree-three node on ``edge`` and attach ``leaf``."""

    left, right = canonical_edge(*edge)
    if left not in backbone or right not in backbone[left]:
        raise ValueError(f"Candidate edge {edge} is absent")
    if leaf in backbone:
        raise ValueError(f"Leaf {leaf} was not detached")
    graph = clone_graph(backbone)
    graph[left].remove(right)
    graph[right].remove(left)
    internal = max(max(graph, default=-1), leaf) + 1
    graph[internal] = {left, right, leaf}
    graph[left].add(internal)
    graph[right].add(internal)
    graph[leaf] = {internal}
    return graph


def split_bitmasks(
    adjacency: dict[int, set[int]], n_taxa: int
) -> frozenset[int]:
    """Return canonical non-trivial unrooted splits as integer bitmasks.

    Python integer masks keep candidate enumeration fast enough for the locked
    30--188-taxon diagnostic without changing the accepted RF definition.
    """

    leaves = sorted(node for node in adjacency if node < n_taxa)
    if len(leaves) < 4:
        return frozenset()
    internals = sorted(node for node in adjacency if node >= n_taxa)
    root = internals[0] if internals else leaves[0]
    parent: dict[int, int | None] = {root: None}
    order = [root]
    for node in order:
        for neighbor in sorted(adjacency[node]):
            if neighbor == parent[node]:
                continue
            if neighbor in parent:
                raise ValueError("Topology contains a cycle")
            parent[neighbor] = node
            order.append(neighbor)
    if len(order) != len(adjacency):
        raise ValueError("Topology is disconnected")

    masks: dict[int, int] = {}
    for node in reversed(order):
        if node < n_taxa:
            masks[node] = 1 << node
        else:
            value = 0
            for neighbor in adjacency[node]:
                if parent.get(neighbor) == node:
                    value |= masks[neighbor]
            masks[node] = value
    total = masks[root]
    leaf_count = total.bit_count()
    splits: set[int] = set()
    for node in order[1:]:
        side = masks[node]
        other = total ^ side
        side_count = side.bit_count()
        other_count = leaf_count - side_count
        if min(side_count, other_count) < 2:
            continue
        if side_count < other_count:
            canonical = side
        elif other_count < side_count:
            canonical = other
        else:
            canonical = min(side, other)
        splits.add(canonical)
    return frozenset(splits)


def normalized_split_rf(
    prediction: frozenset[int], reference: frozenset[int]
) -> float:
    denominator = len(prediction) + len(reference)
    if denominator == 0:
        return 0.0
    return float(len(prediction.symmetric_difference(reference)) / denominator)


def leaf_distance_profiles(
    adjacency: dict[int, set[int]], n_taxa: int
) -> list[list[int]]:
    """All leaf-to-leaf unweighted path lengths for diagnostic prioritization."""

    profiles = [[0] * n_taxa for _ in range(n_taxa)]
    for source in range(n_taxa):
        if source not in adjacency:
            continue
        distances = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        for target in range(n_taxa):
            if target in distances:
                profiles[source][target] = distances[target]
    return profiles


def rank_discrepant_leaves(
    adjacency: dict[int, set[int]],
    reference_profiles: list[list[int]],
    n_taxa: int,
) -> list[tuple[int, int]]:
    current = leaf_distance_profiles(adjacency, n_taxa)
    values = [
        (sum(abs(a - b) for a, b in zip(current[leaf], reference_profiles[leaf])), leaf)
        for leaf in range(n_taxa)
    ]
    values.sort(key=lambda item: (-item[0], item[1]))
    return values


def incident_edges_for_ranked_leaves(
    backbone: dict[int, set[int]],
    query: int,
    nearest_order: Iterable[int],
    k: int,
) -> list[Edge]:
    candidates: set[Edge] = set()
    for leaf in nearest_order:
        if leaf == query or leaf not in backbone:
            continue
        if len(backbone[leaf]) != 1:
            raise ValueError(f"Taxon {leaf} is not pendant in the backbone")
        candidates.add(canonical_edge(leaf, next(iter(backbone[leaf]))))
        if len(candidates) >= k:
            break
    return sorted(candidates)


def edge_neighborhood(
    adjacency: dict[int, set[int]], seeds: Iterable[Edge], radius: int
) -> list[Edge]:
    seeds = [canonical_edge(*edge) for edge in seeds]
    if radius <= 0:
        return sorted(set(seeds))
    distances: dict[int, int] = {}
    queue: deque[int] = deque()
    for left, right in seeds:
        for node in (left, right):
            if node not in distances:
                distances[node] = 0
                queue.append(node)
    while queue:
        node = queue.popleft()
        if distances[node] >= radius:
            continue
        for neighbor in sorted(adjacency[node]):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return sorted({
        canonical_edge(node, neighbor)
        for node, distance in distances.items()
        for neighbor in adjacency[node]
        if min(distance, distances.get(neighbor, radius + 1)) <= radius
    })


def shortest_path_edges(
    adjacency: dict[int, set[int]], source: int, target: int
) -> list[Edge]:
    parent: dict[int, int | None] = {source: None}
    queue = deque([source])
    while queue and target not in parent:
        node = queue.popleft()
        for neighbor in sorted(adjacency[node]):
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    if target not in parent:
        raise ValueError("No path between candidate anchors")
    edges: list[Edge] = []
    node = target
    while parent[node] is not None:
        previous = int(parent[node])
        edges.append(canonical_edge(node, previous))
        node = previous
    return list(reversed(edges))


def rank_doubling_path_candidates(
    backbone: dict[int, set[int]],
    query: int,
    nearest_order: list[int],
    base_k: int = 4,
) -> list[Edge]:
    """Connect near and exponentially ranked representatives through the tree."""

    active = [leaf for leaf in nearest_order if leaf != query and leaf in backbone]
    if not active:
        return []
    selected = active[: min(base_k, len(active))]
    rank = 2 * max(1, base_k)
    while rank <= len(active):
        selected.append(active[rank - 1])
        rank *= 2
    if active[-1] not in selected:
        selected.append(active[-1])
    selected = list(dict.fromkeys(selected))
    root = selected[0]
    edges: set[Edge] = set()
    for leaf in selected[1:]:
        edges.update(shortest_path_edges(backbone, root, leaf))
    if not edges:
        edges.update(incident_edges_for_ranked_leaves(backbone, query, active, 1))
    return sorted(edges)


def policy_edges(
    backbone: dict[int, set[int]],
    query: int,
    nearest_order: list[int],
    policy: str,
) -> list[Edge]:
    all_edges = graph_edges(backbone)
    if policy == "global":
        return all_edges
    if policy.startswith("fixed_k"):
        k = int(policy.removeprefix("fixed_k"))
        return incident_edges_for_ranked_leaves(backbone, query, nearest_order, k)
    if policy.startswith("near4_r"):
        radius = int(policy.removeprefix("near4_r"))
        seeds = incident_edges_for_ranked_leaves(backbone, query, nearest_order, 4)
        return edge_neighborhood(backbone, seeds, radius)
    if policy == "rankdoubling_paths":
        return rank_doubling_path_candidates(backbone, query, nearest_order)
    raise ValueError(f"Unknown candidate policy: {policy}")


@dataclass(frozen=True)
class ReattachmentResult:
    rf: float
    best_edges: tuple[Edge, ...]
    graph: dict[int, set[int]]
    evaluated_edges: int


def best_reattachment(
    backbone: dict[int, set[int]],
    leaf: int,
    candidates: Iterable[Edge],
    reference_splits: frozenset[int],
    n_taxa: int,
) -> ReattachmentResult:
    best_rf = float("inf")
    best_edges: list[Edge] = []
    best_graph: dict[int, set[int]] | None = None
    evaluated = 0
    for edge in sorted(set(canonical_edge(*value) for value in candidates)):
        candidate = reattach_leaf(backbone, leaf, edge)
        rf = normalized_split_rf(split_bitmasks(candidate, n_taxa), reference_splits)
        evaluated += 1
        if rf < best_rf - 1e-12:
            best_rf = rf
            best_edges = [edge]
            best_graph = candidate
        elif abs(rf - best_rf) <= 1e-12:
            best_edges.append(edge)
    if best_graph is None:
        raise ValueError("Candidate policy produced no edges")
    return ReattachmentResult(
        rf=best_rf,
        best_edges=tuple(best_edges),
        graph=best_graph,
        evaluated_edges=evaluated,
    )
