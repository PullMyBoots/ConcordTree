"""Enhanced SparseNJ-style scaffold using ScaleQF's bounded distance oracle.

The implementation follows the mechanism in Algorithm 2 of Kurt et al.
(Bioinformatics, 2024): an exact-NJ seed of size sqrt(n log n), followed by
centroid-guided online insertion using sampled close orienting leaves.  It is a
clean implementation for this project and does not copy the unavailable
official source code.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from concordtree._core.scaleqf import (
    AlignmentSketch,
    BoundedDistanceOracle,
    graph_to_newick,
    load_phylip_sketch,
    neighbor_joining,
    refine_nni,
    tree_to_graph,
    validate_topology,
)


Edge = tuple[int, int]


def farthest_first_order(n_taxa: int, distance: Callable[[int, int], float]) -> list[int]:
    """Return a deterministic maximin taxon order from observable distances.

    This direct accuracy-control implementation performs O(n^2) distance
    lookups; a scalable release would replace it with approximate landmark or
    ANN updates without changing the ordering contract.
    """

    if n_taxa < 2:
        return list(range(n_taxa))
    first, second = max(
        ((left, right) for left in range(n_taxa) for right in range(left + 1, n_taxa)),
        key=lambda pair: (distance(*pair), -pair[0], -pair[1]),
    )
    order = [first, second]
    selected = {first, second}
    minimum = {
        taxon: min(distance(taxon, first), distance(taxon, second))
        for taxon in range(n_taxa)
        if taxon not in selected
    }
    while minimum:
        taxon = max(minimum, key=lambda value: (minimum[value], -value))
        order.append(taxon)
        del minimum[taxon]
        for other in minimum:
            minimum[other] = min(minimum[other], distance(other, taxon))
    return order


def canonical_edge(left: int, right: int) -> Edge:
    return (left, right) if left < right else (right, left)


def graph_edges(adjacency: dict[int, set[int]]) -> set[Edge]:
    return {
        canonical_edge(left, right)
        for left, neighbors in adjacency.items()
        for right in neighbors
        if left < right
    }


def _edge_groups(candidate_edges: set[Edge], centroid: int) -> dict[int, set[Edge]]:
    restricted: dict[int, set[int]] = {}
    for left, right in candidate_edges:
        restricted.setdefault(left, set()).add(right)
        restricted.setdefault(right, set()).add(left)
    groups: dict[int, set[Edge]] = {}
    for neighbor in sorted(restricted.get(centroid, set())):
        group: set[Edge] = {canonical_edge(centroid, neighbor)}
        stack = [(neighbor, centroid)]
        while stack:
            node, parent = stack.pop()
            for child in restricted[node]:
                if child == parent or child == centroid:
                    continue
                edge = canonical_edge(node, child)
                if edge not in group:
                    group.add(edge)
                    stack.append((child, node))
        groups[neighbor] = group
    return groups


def _edge_centroid_bruteforce(
    candidate_edges: set[Edge], n_taxa: int
) -> tuple[int, dict[int, set[Edge]]]:
    """Reference implementation retained for equivalence tests/fallbacks."""

    nodes = sorted({node for edge in candidate_edges for node in edge})
    best_node = nodes[0]
    best_groups = _edge_groups(candidate_edges, best_node)
    def score(node: int, groups: dict[int, set[Edge]]) -> tuple[int, int, int]:
        leaf_counts = [
            len({endpoint for edge in group for endpoint in edge if endpoint < n_taxa})
            for group in groups.values()
        ]
        return (
            max(leaf_counts, default=0),
            max((len(group) for group in groups.values()), default=0),
            node,
        )

    best_score = score(best_node, best_groups)
    for node in nodes[1:]:
        groups = _edge_groups(candidate_edges, node)
        node_score = score(node, groups)
        if node_score < best_score:
            best_node, best_groups, best_score = node, groups, node_score
    return best_node, best_groups


def edge_centroid(
    candidate_edges: set[Edge], n_taxa: int
) -> tuple[int, dict[int, set[Edge]]]:
    """Return the exact leaf-weight centroid using one linear tree DP.

    Candidate regions produced by sparse routing are connected edge subtrees.
    Subtree leaf and edge counts make the score of every possible center
    available in ``O(|E|)`` time; only the winning center's explicit groups are
    materialized.  The fallback preserves historical behavior for a malformed
    disconnected/cyclic edge set.
    """

    if not candidate_edges:
        raise ValueError("Cannot find a centroid of an empty edge subtree")
    edges = {canonical_edge(*edge) for edge in candidate_edges}
    restricted: dict[int, set[int]] = {}
    for left, right in edges:
        restricted.setdefault(left, set()).add(right)
        restricted.setdefault(right, set()).add(left)
    nodes = sorted(restricted)
    root = nodes[0]
    parent: dict[int, int | None] = {root: None}
    order = [root]
    for node in order:
        for neighbor in sorted(restricted[node]):
            if neighbor == parent[node]:
                continue
            if neighbor in parent:
                return _edge_centroid_bruteforce(edges, n_taxa)
            parent[neighbor] = node
            order.append(neighbor)
    if len(order) != len(nodes) or len(edges) != len(nodes) - 1:
        return _edge_centroid_bruteforce(edges, n_taxa)

    subtree_leaves = {node: int(node < n_taxa) for node in nodes}
    subtree_edges = {node: 0 for node in nodes}
    for node in reversed(order):
        for neighbor in restricted[node]:
            if parent.get(neighbor) == node:
                subtree_leaves[node] += subtree_leaves[neighbor]
                subtree_edges[node] += 1 + subtree_edges[neighbor]
    total_leaves = subtree_leaves[root]
    total_edges = len(edges)

    best_node = nodes[0]
    best_score: tuple[int, int, int] | None = None
    for node in nodes:
        leaf_counts: list[int] = []
        edge_counts: list[int] = []
        center_leaf = int(node < n_taxa)
        for neighbor in restricted[node]:
            if parent.get(neighbor) == node:
                leaf_counts.append(subtree_leaves[neighbor] + center_leaf)
                edge_counts.append(1 + subtree_edges[neighbor])
            elif parent[node] == neighbor:
                leaf_counts.append(total_leaves - subtree_leaves[node] + center_leaf)
                edge_counts.append(total_edges - subtree_edges[node])
            else:  # Defensive: the connected acyclic audit above should exclude this.
                return _edge_centroid_bruteforce(edges, n_taxa)
        score = (max(leaf_counts, default=0), max(edge_counts, default=0), node)
        if best_score is None or score < best_score:
            best_node, best_score = node, score
    return best_node, _edge_groups(edges, best_node)


def branch_leaves(
    adjacency: dict[int, set[int]],
    neighbor: int,
    centroid: int,
    n_taxa: int,
) -> list[int]:
    leaves: list[int] = []
    stack = [(neighbor, centroid)]
    while stack:
        node, parent = stack.pop()
        if node < n_taxa:
            leaves.append(node)
            continue
        for child in sorted(adjacency[node], reverse=True):
            if child != parent:
                stack.append((child, node))
    return sorted(leaves)


def _mean_between(
    left: list[int], right: list[int], distance: Callable[[int, int], float]
) -> float:
    return float(np.mean([distance(a, b) for a in left for b in right]))


def choose_branch(
    query: int,
    centroid: int,
    adjacency: dict[int, set[int]],
    distance: Callable[[int, int], float],
    n_taxa: int,
    rng: np.random.Generator,
    sample_size: int,
    orienting_count: int,
) -> tuple[int, dict[int, float]]:
    """Choose one incident branch with the averaged four-point condition."""

    neighbors = sorted(adjacency[centroid])
    if len(neighbors) != 3:
        raise ValueError(
            f"Centroid {centroid} must have degree three in an unrooted binary tree, "
            f"found {len(neighbors)}"
        )
    representatives: dict[int, list[int]] = {}
    for neighbor in neighbors:
        leaves = branch_leaves(adjacency, neighbor, centroid, n_taxa)
        if not leaves:
            raise ValueError(f"Branch {centroid}-{neighbor} has no orienting leaf")
        take = min(len(leaves), max(1, sample_size))
        if take < len(leaves):
            sampled = sorted(int(x) for x in rng.choice(leaves, size=take, replace=False))
        else:
            sampled = leaves
        sampled.sort(key=lambda leaf: (distance(query, leaf), leaf))
        representatives[neighbor] = sampled[: min(orienting_count, len(sampled))]

    query_distance = {
        neighbor: float(np.mean([distance(query, leaf) for leaf in representatives[neighbor]]))
        for neighbor in neighbors
    }
    scores: dict[int, float] = {}
    for neighbor in neighbors:
        other = [value for value in neighbors if value != neighbor]
        scores[neighbor] = query_distance[neighbor] + _mean_between(
            representatives[other[0]], representatives[other[1]], distance
        )
    return min(neighbors, key=lambda neighbor: (scores[neighbor], neighbor)), scores


@dataclass
class SparseNJStats:
    seed_size: int
    inserted: int = 0
    navigation_steps: int = 0
    branch_decisions: int = 0
    constrained_fallbacks: int = 0
    max_candidate_edges: int = 0
    local_edges_scored: int = 0
    local_edge_changes: int = 0


def _stable_close_representatives(
    leaves: list[int],
    query: int,
    distance: Callable[[int, int], float],
    sample_size: int,
    count: int,
    salt: int,
) -> list[int]:
    """Deterministically sample a branch, then retain query-close leaves."""

    ranked = sorted(
        leaves,
        key=lambda leaf: (
            ((leaf + 1) * 2654435761 + (query + 1) * 2246822519 + salt) & 0xFFFFFFFF,
            leaf,
        ),
    )
    sampled = ranked[: min(len(ranked), max(1, sample_size))]
    sampled.sort(key=lambda leaf: (distance(query, leaf), leaf))
    return sampled[: min(len(sampled), max(1, count))]


def _incident_branch_representatives(
    endpoint: int,
    opposite: int,
    query: int,
    adjacency: dict[int, set[int]],
    distance: Callable[[int, int], float],
    n_taxa: int,
    sample_size: int,
    count: int,
) -> list[list[int]]:
    if endpoint < n_taxa:
        return [[endpoint]]
    groups: list[list[int]] = []
    for neighbor in sorted(adjacency[endpoint] - {opposite}):
        leaves = branch_leaves(adjacency, neighbor, endpoint, n_taxa)
        groups.append(
            _stable_close_representatives(
                leaves,
                query,
                distance,
                sample_size,
                count,
                salt=(endpoint + 1) * 1315423911 + neighbor,
            )
        )
    return groups


def local_edge_quartet_score(
    edge: Edge,
    query: int,
    adjacency: dict[int, set[int]],
    distance: Callable[[int, int], float],
    n_taxa: int,
    sample_size: int,
    orienting_count: int,
) -> float:
    """Margin supporting insertion on an edge from its incident subtrees."""

    left, right = edge
    left_groups = _incident_branch_representatives(
        left,
        right,
        query,
        adjacency,
        distance,
        n_taxa,
        sample_size,
        orienting_count,
    )
    right_groups = _incident_branch_representatives(
        right,
        left,
        query,
        adjacency,
        distance,
        n_taxa,
        sample_size,
        orienting_count,
    )
    margins: list[float] = []

    def add_constraint(same_side: list[list[int]], opposite_side: list[list[int]]) -> None:
        if len(same_side) < 2 or not opposite_side:
            return
        first, second = same_side[0], same_side[1]
        opposite = [leaf for group in opposite_side for leaf in group]
        desired = _mean_between(first, second, distance) + float(
            np.mean([distance(query, leaf) for leaf in opposite])
        )
        alternative_a = float(np.mean([distance(query, leaf) for leaf in first])) + _mean_between(
            second, opposite, distance
        )
        alternative_b = float(np.mean([distance(query, leaf) for leaf in second])) + _mean_between(
            first, opposite, distance
        )
        margins.append(min(alternative_a, alternative_b) - desired)

    add_constraint(left_groups, right_groups)
    add_constraint(right_groups, left_groups)
    # An internal candidate edge must satisfy the quartet constraint on both
    # endpoints.  Averaging lets one strongly supported side hide a violated
    # side and systematically favors central edges.
    return float(min(margins)) if margins else -math.inf


def nearby_edges(
    adjacency: dict[int, set[int]], edge: Edge, radius: int
) -> list[Edge]:
    if radius <= 0:
        return [canonical_edge(*edge)]
    distances = {edge[0]: 0, edge[1]: 0}
    queue = [edge[0], edge[1]]
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        if distances[node] >= radius:
            continue
        for neighbor in sorted(adjacency[node]):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    candidates = {
        canonical_edge(node, neighbor)
        for node in distances
        for neighbor in adjacency[node]
        if min(distances.get(node, radius + 1), distances.get(neighbor, radius + 1)) <= radius
    }
    return sorted(candidates)


def select_local_edge(
    adjacency: dict[int, set[int]],
    selected_edge: Edge,
    query: int,
    distance: Callable[[int, int], float],
    n_taxa: int,
    current_taxa: int,
    orienting_count: int,
    radius: int,
) -> tuple[Edge, int]:
    candidates = nearby_edges(adjacency, selected_edge, radius)
    sample_size = max(1, int(math.ceil(math.log2(max(4, current_taxa)))))
    scores = {
        edge: local_edge_quartet_score(
            edge,
            query,
            adjacency,
            distance,
            n_taxa,
            sample_size,
            orienting_count,
        )
        for edge in candidates
    }
    finite = [edge for edge in candidates if math.isfinite(scores[edge])]
    if not finite:
        return canonical_edge(*selected_edge), len(candidates)
    best = max(finite, key=lambda edge: (scores[edge], tuple(-value for value in edge)))
    return best, len(candidates)


def insert_taxon(
    adjacency: dict[int, set[int]],
    query: int,
    distance: Callable[[int, int], float],
    n_taxa: int,
    rng: np.random.Generator,
    stats: SparseNJStats,
    orienting_count: int = 3,
    local_radius: int = 0,
) -> None:
    candidate_edges = graph_edges(adjacency)
    stats.max_candidate_edges = max(stats.max_candidate_edges, len(candidate_edges))
    sample_size = max(1, int(math.ceil(math.log2(max(4, stats.seed_size + stats.inserted)))))

    while len(candidate_edges) > 1:
        centroid, groups = edge_centroid(candidate_edges, n_taxa)
        if len(groups) == 1:
            candidate_edges = next(iter(groups.values()))
            stats.navigation_steps += 1
            continue
        selected, scores = choose_branch(
            query,
            centroid,
            adjacency,
            distance,
            n_taxa,
            rng,
            sample_size,
            orienting_count,
        )
        stats.branch_decisions += 1
        if selected not in groups:
            selected = min(groups, key=lambda neighbor: (scores[neighbor], neighbor))
            stats.constrained_fallbacks += 1
        next_edges = groups[selected]
        if len(next_edges) >= len(candidate_edges):
            raise RuntimeError("Centroid navigation failed to reduce the candidate edge set")
        candidate_edges = next_edges
        stats.navigation_steps += 1

    selected_edge = next(iter(candidate_edges))
    if local_radius > 0:
        corrected_edge, scored = select_local_edge(
            adjacency,
            selected_edge,
            query,
            distance,
            n_taxa,
            stats.seed_size + stats.inserted,
            orienting_count,
            local_radius,
        )
        stats.local_edges_scored += scored
        if corrected_edge != canonical_edge(*selected_edge):
            stats.local_edge_changes += 1
        selected_edge = corrected_edge
    left, right = selected_edge
    adjacency[left].remove(right)
    adjacency[right].remove(left)
    internal = max(adjacency) + 1
    adjacency[internal] = {left, right, query}
    adjacency[left].add(internal)
    adjacency[right].add(internal)
    adjacency[query] = {internal}
    stats.inserted += 1


def build_sparse_nj(
    n_taxa: int,
    distance: Callable[[int, int], float],
    seed: int = 20260828,
    orienting_count: int = 3,
    seed_size: int | None = None,
    seed_factor: float = 1.0,
    local_radius: int = 0,
) -> tuple[dict[int, set[int]], SparseNJStats]:
    if n_taxa < 4:
        raise ValueError("SparseNJ requires at least four taxa")
    rng = np.random.default_rng(seed)
    order = [int(value) for value in rng.permutation(n_taxa)]
    if seed_size is None:
        seed_size = int(
            math.ceil(seed_factor * math.sqrt(n_taxa * math.log(max(n_taxa, 2))))
        )
    seed_size = min(n_taxa, max(4, int(seed_size)))
    seed_leaves = order[:seed_size]
    remaining = order[seed_size:]
    root = neighbor_joining(seed_leaves, distance)
    adjacency = tree_to_graph(root, n_taxa)
    stats = SparseNJStats(seed_size=seed_size)
    for query in remaining:
        insert_taxon(
            adjacency,
            query,
            distance,
            n_taxa,
            rng,
            stats,
            orienting_count=orienting_count,
            local_radius=local_radius,
        )
    validate_topology(adjacency, n_taxa)
    return adjacency, stats


def infer_snj_alignment(
    alignment_path: str | Path,
    max_sites: int = 8192,
    blocks: int = 32,
    orienting_count: int = 3,
    seed_size: int | None = None,
    seed_factor: float = 1.0,
    local_radius: int = 0,
    nni_passes: int = 2,
    representatives: int = 3,
    seed: int = 20260828,
) -> dict[str, object]:
    started = time.perf_counter()
    sketch: AlignmentSketch = load_phylip_sketch(
        alignment_path, max_sites=max_sites, blocks=blocks, seed=seed
    )
    loaded_at = time.perf_counter()
    distance = BoundedDistanceOracle(sketch)
    adjacency, stats = build_sparse_nj(
        sketch.n_taxa,
        distance,
        seed=seed,
        orienting_count=orienting_count,
        seed_size=seed_size,
        seed_factor=seed_factor,
        local_radius=local_radius,
    )
    scaffold_newick = graph_to_newick(adjacency, sketch.names, sketch.n_taxa)
    scaffold_at = time.perf_counter()
    repair = refine_nni(
        adjacency,
        sketch.n_taxa,
        distance,
        passes=nni_passes,
        representatives=representatives,
    )
    validate_topology(adjacency, sketch.n_taxa)
    refined_newick = graph_to_newick(adjacency, sketch.names, sketch.n_taxa)
    finished = time.perf_counter()
    return {
        "scaffold_newick": scaffold_newick,
        "refined_newick": refined_newick,
        "metadata": {
            "alignment": str(Path(alignment_path).resolve()),
            "n_taxa": sketch.n_taxa,
            "alignment_length": sketch.alignment_length,
            "variable_sketch_sites": sketch.n_sites,
            "max_sites": max_sites,
            "blocks": blocks,
            "orienting_count": orienting_count,
            "seed_size": stats.seed_size,
            "seed_factor": seed_factor,
            "local_radius": local_radius,
            "nni_passes": nni_passes,
            "representatives": representatives,
            "seed": seed,
            "sparse_nj": {
                "inserted": stats.inserted,
                "navigation_steps": stats.navigation_steps,
                "branch_decisions": stats.branch_decisions,
                "constrained_fallbacks": stats.constrained_fallbacks,
                "max_candidate_edges": stats.max_candidate_edges,
                "local_edges_scored": stats.local_edges_scored,
                "local_edge_changes": stats.local_edge_changes,
            },
            "repair": repair,
            "distance_calls": distance.calls,
            "distance_cache_hits": distance.cache_hits,
            "distance_cache_entries": distance.cache_entries,
            "load_seconds": loaded_at - started,
            "scaffold_seconds": scaffold_at - loaded_at,
            "repair_seconds": finished - scaffold_at,
            "total_seconds": finished - started,
        },
    }
