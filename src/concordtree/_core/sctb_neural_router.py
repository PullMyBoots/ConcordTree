"""Sparse neural centroid routing used by the SCTB initializer.

The router inserts one taxon by navigating a logarithmically shrinking set of
tree edges.  At each internal centroid, quartet probabilities decide which
incident branch contains the attachment.  This module never accesses a truth
tree and keeps the sequence scorer behind a small callable interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np

from concordtree._core.learned_nni import class_for_group_pairing
from concordtree._core.scaleqf import TreeNode, neighbor_joining, tree_to_graph, validate_topology
from concordtree._core.sparse_nj import (
    Edge,
    _edge_groups,
    _stable_close_representatives,
    branch_leaves,
    canonical_edge,
    edge_centroid,
    graph_edges,
)


Predictor = Callable[[np.ndarray], np.ndarray]
Distance = Callable[[int, int], float]


@dataclass
class NeuralRouterStats:
    seed_size: int
    inserted: int = 0
    navigation_steps: int = 0
    quartet_queries: int = 0
    restricted_choice_events: int = 0
    maximum_candidate_edges: int = 0
    margins_sum: float = 0.0
    margins_minimum: float = 1.0
    attachment_edges_scored: int = 0
    attachment_quartet_queries: int = 0
    aggregator_decisions: int = 0
    aggregator_changes: int = 0


def neural_branch_scores(
    query: int,
    centroid: int,
    adjacency: dict[int, set[int]],
    distance: Distance,
    predict_probabilities: Predictor,
    n_taxa: int,
    current_taxa: int,
    representatives: int,
) -> tuple[dict[int, float], int]:
    neighbors = sorted(adjacency[centroid])
    if centroid < n_taxa or len(neighbors) != 3:
        raise ValueError("neural routing requires a degree-three internal centroid")
    sample_size = max(representatives, int(np.ceil(np.log2(max(4, current_taxa)))))
    groups: dict[int, list[int]] = {}
    for neighbor in neighbors:
        leaves = branch_leaves(adjacency, neighbor, centroid, n_taxa)
        groups[neighbor] = _stable_close_representatives(
            leaves,
            query,
            distance,
            sample_size=sample_size,
            count=representatives,
            salt=(centroid + 1) * 1315423911 + neighbor,
        )
    records = list(product(*(groups[neighbor] for neighbor in neighbors)))
    quartets = np.asarray(
        [sorted((query, record[0], record[1], record[2])) for record in records],
        dtype=np.int64,
    )
    probabilities = np.asarray(predict_probabilities(quartets), dtype=np.float64)
    if probabilities.shape != (len(records), 3):
        raise ValueError(f"quartet probability shape mismatch: {probabilities.shape}")
    scores = {neighbor: 0.0 for neighbor in neighbors}
    for record, probs in zip(records, probabilities):
        sorted_quartet = tuple(sorted((query, *record)))
        for offset, neighbor in enumerate(neighbors):
            class_index = class_for_group_pairing(sorted_quartet, (query, record[offset]))
            scores[neighbor] += float(probs[class_index])
    for neighbor in scores:
        scores[neighbor] /= max(len(records), 1)
    return scores, len(records)


def _edge_side_groups(
    edge: Edge,
    endpoint: int,
    opposite: int,
    query: int,
    adjacency: dict[int, set[int]],
    distance: Distance,
    n_taxa: int,
    current_taxa: int,
    representatives: int,
) -> list[list[int]]:
    if endpoint < n_taxa:
        return [[endpoint]]
    sample_size = max(representatives, int(np.ceil(np.log2(max(4, current_taxa)))))
    groups: list[list[int]] = []
    for neighbor in sorted(adjacency[endpoint] - {opposite}):
        leaves = branch_leaves(adjacency, neighbor, endpoint, n_taxa)
        groups.append(
            _stable_close_representatives(
                leaves,
                query,
                distance,
                sample_size=sample_size,
                count=representatives,
                salt=(endpoint + 1) * 2246822519 + neighbor,
            )
        )
    return groups


def neural_attachment_scores(
    query: int,
    edges: list[Edge],
    adjacency: dict[int, set[int]],
    distance: Distance,
    predict_probabilities: Predictor,
    n_taxa: int,
    current_taxa: int,
    representatives: int,
) -> tuple[dict[Edge, float], int]:
    """Score candidate attachment edges by their local quartet invariants."""

    specifications = attachment_specifications(
        query,
        edges,
        adjacency,
        distance,
        n_taxa,
        current_taxa,
        representatives,
    )
    if not specifications:
        return {canonical_edge(*edge): float("-inf") for edge in edges}, 0
    quartets = np.asarray([sorted(item[1]) for item in specifications], dtype=np.int64)
    probabilities = np.asarray(predict_probabilities(quartets), dtype=np.float64)
    return attachment_scores_from_probabilities(edges, specifications, probabilities), len(specifications)


def attachment_specifications(
    query: int,
    edges: list[Edge],
    adjacency: dict[int, set[int]],
    distance: Distance,
    n_taxa: int,
    current_taxa: int,
    representatives: int,
) -> list[tuple[Edge, tuple[int, int, int, int], tuple[int, int]]]:
    """Build the observable quartet constraints for several attachment actions."""

    specifications: list[tuple[Edge, tuple[int, int, int, int], tuple[int, int]]] = []
    for raw_edge in edges:
        edge = canonical_edge(*raw_edge)
        left, right = edge
        left_groups = _edge_side_groups(
            edge, left, right, query, adjacency, distance, n_taxa, current_taxa, representatives
        )
        right_groups = _edge_side_groups(
            edge, right, left, query, adjacency, distance, n_taxa, current_taxa, representatives
        )
        if len(left_groups) == 1 and len(right_groups) >= 2:
            for leaf, first, second in product(
                left_groups[0], right_groups[0], right_groups[1]
            ):
                specifications.append((edge, (query, leaf, first, second), (query, leaf)))
        elif len(right_groups) == 1 and len(left_groups) >= 2:
            for leaf, first, second in product(
                right_groups[0], left_groups[0], left_groups[1]
            ):
                specifications.append((edge, (query, leaf, first, second), (query, leaf)))
        elif len(left_groups) >= 2 and len(right_groups) >= 2:
            for first, second, opposite_leaf in product(
                left_groups[0], left_groups[1], right_groups[0] + right_groups[1]
            ):
                specifications.append(
                    (edge, (query, first, second, opposite_leaf), (query, opposite_leaf))
                )
            for first, second, opposite_leaf in product(
                right_groups[0], right_groups[1], left_groups[0] + left_groups[1]
            ):
                specifications.append(
                    (edge, (query, first, second, opposite_leaf), (query, opposite_leaf))
                )
    return specifications


def attachment_scores_from_probabilities(
    edges: list[Edge],
    specifications: list[tuple[Edge, tuple[int, int, int, int], tuple[int, int]]],
    probabilities: np.ndarray,
) -> dict[Edge, float]:
    """Aggregate probabilities computed in one shared context into action scores."""

    if probabilities.shape != (len(specifications), 3):
        raise ValueError(f"quartet probability shape mismatch: {probabilities.shape}")
    values: dict[Edge, list[float]] = {canonical_edge(*edge): [] for edge in edges}
    for (edge, quartet, desired_pair), probs in zip(specifications, probabilities):
        sorted_quartet = tuple(sorted(quartet))
        desired = class_for_group_pairing(sorted_quartet, desired_pair)
        alternative = max(float(probs[index]) for index in range(3) if index != desired)
        values[edge].append(float(probs[desired]) - alternative)
    return {
        edge: float(np.mean(edge_values)) if edge_values else float("-inf")
        for edge, edge_values in values.items()
    }


def beam_route_edge_scores(
    query: int,
    adjacency: dict[int, set[int]],
    distance: Distance,
    predict_probabilities: Predictor,
    n_taxa: int,
    current_taxa: int,
    representatives: int,
    beam_width: int,
    stats: NeuralRouterStats,
) -> dict[Edge, float]:
    """Return terminal attachment edges with their best navigation log score."""

    initial = graph_edges(adjacency)
    states: list[tuple[float, set[Edge]]] = [(0.0, initial)]
    maximum_rounds = max(4, 4 * int(np.ceil(np.log2(max(len(initial), 2)))))
    for _ in range(maximum_rounds):
        if all(len(edges) == 1 for _, edges in states):
            break
        expanded: list[tuple[float, set[Edge]]] = []
        for path_score, candidate_edges in states:
            if len(candidate_edges) == 1:
                expanded.append((path_score, candidate_edges))
                continue
            centroid, groups = edge_centroid(candidate_edges, n_taxa)
            if len(groups) == 1:
                expanded.append((path_score, next(iter(groups.values()))))
                stats.navigation_steps += 1
                continue
            if centroid < n_taxa or len(adjacency[centroid]) != 3:
                chosen = min(groups.values(), key=lambda value: (len(value), sorted(value)))
                expanded.append((path_score, chosen))
                stats.restricted_choice_events += 1
                stats.navigation_steps += 1
                continue
            scores, queries = neural_branch_scores(
                query,
                centroid,
                adjacency,
                distance,
                predict_probabilities,
                n_taxa,
                current_taxa,
                representatives,
            )
            stats.quartet_queries += queries
            allowed = sorted(groups)
            raw = np.asarray([scores[neighbor] for neighbor in allowed], dtype=np.float64)
            raw -= raw.max()
            probabilities = np.exp(raw)
            probabilities /= probabilities.sum()
            for neighbor, probability in zip(allowed, probabilities):
                expanded.append(
                    (path_score + float(np.log(max(float(probability), 1e-12))), groups[neighbor])
                )
            stats.navigation_steps += 1
        deduplicated: dict[tuple[Edge, ...], tuple[float, set[Edge]]] = {}
        for item in expanded:
            key = tuple(sorted(item[1]))
            incumbent = deduplicated.get(key)
            if incumbent is None or item[0] > incumbent[0]:
                deduplicated[key] = item
        states = sorted(
            deduplicated.values(), key=lambda item: (-item[0], len(item[1]), sorted(item[1]))
        )[: max(1, beam_width)]
    candidates: dict[Edge, float] = {}
    for path_score, edges in states:
        if len(edges) == 1:
            edge = next(iter(edges))
        else:
            edge = min(edges)
        edge = canonical_edge(*edge)
        candidates[edge] = max(candidates.get(edge, float("-inf")), path_score)
    return dict(sorted(candidates.items()))


def beam_route_edges(
    query: int,
    adjacency: dict[int, set[int]],
    distance: Distance,
    predict_probabilities: Predictor,
    n_taxa: int,
    current_taxa: int,
    representatives: int,
    beam_width: int,
    stats: NeuralRouterStats,
) -> list[Edge]:
    """Compatibility wrapper returning only the scored beam's edge identities."""

    return list(
        beam_route_edge_scores(
            query,
            adjacency,
            distance,
            predict_probabilities,
            n_taxa,
            current_taxa,
            representatives,
            beam_width,
            stats,
        )
    )


def insert_taxon_neural(
    adjacency: dict[int, set[int]],
    query: int,
    distance: Distance,
    predict_probabilities: Predictor,
    n_taxa: int,
    current_taxa: int,
    representatives: int,
    stats: NeuralRouterStats,
) -> None:
    candidate_edges = graph_edges(adjacency)
    stats.maximum_candidate_edges = max(stats.maximum_candidate_edges, len(candidate_edges))
    while len(candidate_edges) > 1:
        centroid, groups = edge_centroid(candidate_edges, n_taxa)
        if len(groups) == 1:
            candidate_edges = next(iter(groups.values()))
            stats.navigation_steps += 1
            continue
        if centroid < n_taxa or len(adjacency[centroid]) != 3:
            # This is a deterministic progress fallback for a two-edge terminal
            # region; it is topology-only and does not inspect reference data.
            candidate_edges = min(
                groups.values(), key=lambda value: (len(value), sorted(value))
            )
            stats.restricted_choice_events += 1
            stats.navigation_steps += 1
            continue
        scores, queries = neural_branch_scores(
            query,
            centroid,
            adjacency,
            distance,
            predict_probabilities,
            n_taxa,
            current_taxa,
            representatives,
        )
        stats.quartet_queries += queries
        allowed = sorted(groups)
        ranked = sorted(allowed, key=lambda neighbor: (-scores[neighbor], neighbor))
        selected = ranked[0]
        full_best = max(scores, key=lambda neighbor: (scores[neighbor], -neighbor))
        if full_best not in groups:
            stats.restricted_choice_events += 1
        if len(ranked) > 1:
            margin = scores[ranked[0]] - scores[ranked[1]]
            stats.margins_sum += margin
            stats.margins_minimum = min(stats.margins_minimum, margin)
        next_edges = groups[selected]
        if len(next_edges) >= len(candidate_edges):
            raise RuntimeError("neural centroid navigation did not shrink the edge set")
        candidate_edges = next_edges
        stats.navigation_steps += 1

    left, right = next(iter(candidate_edges))
    adjacency[left].remove(right)
    adjacency[right].remove(left)
    internal = max(adjacency) + 1
    adjacency[internal] = {left, right, query}
    adjacency[left].add(internal)
    adjacency[right].add(internal)
    adjacency[query] = {internal}
    stats.inserted += 1


def insert_taxon_beam(
    adjacency: dict[int, set[int]],
    query: int,
    distance: Distance,
    predict_probabilities: Predictor,
    n_taxa: int,
    current_taxa: int,
    representatives: int,
    beam_width: int,
    attachment_representatives: int,
    stats: NeuralRouterStats,
) -> None:
    candidates = beam_route_edges(
        query,
        adjacency,
        distance,
        predict_probabilities,
        n_taxa,
        current_taxa,
        representatives,
        beam_width,
        stats,
    )
    scores, queries = neural_attachment_scores(
        query,
        candidates,
        adjacency,
        distance,
        predict_probabilities,
        n_taxa,
        current_taxa,
        attachment_representatives,
    )
    stats.attachment_edges_scored += len(candidates)
    stats.attachment_quartet_queries += queries
    selected = max(candidates, key=lambda edge: (scores[edge], tuple(-x for x in edge)))
    left, right = selected
    adjacency[left].remove(right)
    adjacency[right].remove(left)
    internal = max(adjacency) + 1
    adjacency[internal] = {left, right, query}
    adjacency[left].add(internal)
    adjacency[right].add(internal)
    adjacency[query] = {internal}
    stats.inserted += 1


def build_neural_router_tree(
    n_taxa: int,
    distance: Distance,
    predict_probabilities: Predictor,
    seed: int,
    seed_size: int = 24,
    representatives: int = 2,
    beam_width: int = 1,
    attachment_representatives: int = 2,
    insertion_order: list[int] | None = None,
) -> tuple[dict[int, set[int]], NeuralRouterStats]:
    if n_taxa < 4:
        raise ValueError("neural router requires at least four taxa")
    seed_size = min(n_taxa, max(4, int(seed_size)))
    if insertion_order is None:
        rng = np.random.default_rng(seed)
        order = [int(value) for value in rng.permutation(n_taxa)]
    else:
        order = [int(value) for value in insertion_order]
        if sorted(order) != list(range(n_taxa)):
            raise ValueError("insertion_order must be a permutation of all taxa")
    seed_leaves = order[:seed_size]
    root = neighbor_joining(seed_leaves, distance)
    adjacency = tree_to_graph(root, n_taxa)
    stats = NeuralRouterStats(seed_size=seed_size)
    for query in order[seed_size:]:
        if beam_width <= 1:
            insert_taxon_neural(
                adjacency,
                query,
                distance,
                predict_probabilities,
                n_taxa,
                seed_size + stats.inserted,
                representatives,
                stats,
            )
        else:
            insert_taxon_beam(
                adjacency,
                query,
                distance,
                predict_probabilities,
                n_taxa,
                seed_size + stats.inserted,
                representatives,
                beam_width,
                attachment_representatives,
                stats,
            )
    validate_topology(adjacency, n_taxa)
    return adjacency, stats
