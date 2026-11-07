"""Oracle reachability diagnostics for Sparse Contextual Tree Building.

This module never uses the reference topology as an inference feature.  It uses
the reference only to ask a deliberately narrower development question: if an
otherwise perfect action scorer were restricted to a sparse candidate graph,
could it still choose all of the true contractions?  Dense ranking is allowed
inside this diagnostic and is reported separately; a later ANN gate must show
that the same sparse candidates can be retrieved at scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ReachabilityResult:
    candidate_k: int
    complete: bool
    stalled: bool
    rounds: int
    initial_taxa: int
    final_clusters: int
    resolved_splits: int
    total_splits: int
    optimistic_rf_ceiling: float
    selected_directed_candidates: int
    dense_pairs_ranked: int
    maximum_candidate_degree: int
    required_k: tuple[int, ...]


@dataclass(frozen=True)
class ProjectionCandidateConfig:
    projections: int = 16
    window: int = 4
    candidate_cap: int = 32
    seed: int = 20260903


def initial_profile_counts(states: np.ndarray) -> dict[int, np.ndarray]:
    """Return per-taxon A/C/G/T counts over shared sampled alignment sites."""

    states = np.asarray(states, dtype=np.uint8)
    if states.ndim != 2:
        raise ValueError("states must have shape [taxa, sites]")
    counts: dict[int, np.ndarray] = {}
    for taxon, row in enumerate(states):
        value = np.zeros((len(row), 4), dtype=np.float32)
        valid = row < 4
        value[np.nonzero(valid)[0], row[valid]] = 1.0
        counts[taxon] = value
    return counts


def profile_distance_matrix(
    active: Iterable[int], counts: dict[int, np.ndarray]
) -> tuple[list[int], np.ndarray]:
    """Expected mismatch distance between mergeable cluster profiles.

    Each site contributes equally when both clusters have at least one observed
    state.  Missing-only overlaps receive infinite distance.  BLAS computes the
    match and shared-observation matrices, keeping the development diagnostic
    compact while preserving explicit dense-work accounting.
    """

    nodes = sorted(active)
    if not nodes:
        raise ValueError("no active clusters")
    raw = np.stack([counts[node] for node in nodes]).astype(np.float32, copy=False)
    totals = raw.sum(axis=2, keepdims=True)
    frequencies = np.divide(raw, totals, out=np.zeros_like(raw), where=totals > 0)
    observed = (totals[..., 0] > 0).astype(np.float32)
    flat = frequencies.reshape(len(nodes), -1)
    matches = flat @ flat.T
    shared = observed @ observed.T
    similarity = np.divide(
        matches,
        shared,
        out=np.full_like(matches, -np.inf),
        where=shared > 0,
    )
    distance = 1.0 - similarity
    distance[shared <= 0] = np.inf
    np.fill_diagonal(distance, np.inf)
    return nodes, distance


def _pair_profile_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_total = left.sum(axis=1, keepdims=True)
    right_total = right.sum(axis=1, keepdims=True)
    valid = (left_total[:, 0] > 0) & (right_total[:, 0] > 0)
    if not np.any(valid):
        return float("inf")
    left_frequency = left[valid] / left_total[valid]
    right_frequency = right[valid] / right_total[valid]
    return float(1.0 - np.sum(left_frequency * right_frequency) / valid.sum())


def projection_order_candidates(
    active: Iterable[int],
    counts: dict[int, np.ndarray],
    config: ProjectionCandidateConfig,
    tie_keys: dict[int, tuple[int, ...]] | None = None,
) -> dict[int, set[int]]:
    """Bounded candidate retrieval from random profile orderings.

    Random projections are over site/state coordinates rather than taxon ids,
    so relabeling taxa only permutes the rows.  Each projected ordering proposes
    a small window; exact profile mismatch reranks only that bounded union.
    """

    nodes = sorted(active)
    if config.projections <= 0 or config.window <= 0 or config.candidate_cap <= 0:
        raise ValueError("projection configuration must be positive")
    raw = np.stack([counts[node] for node in nodes]).astype(np.float32, copy=False)
    totals = raw.sum(axis=2, keepdims=True)
    frequencies = np.divide(raw, totals, out=np.zeros_like(raw), where=totals > 0)
    flat = frequencies.reshape(len(nodes), -1)
    rng = np.random.default_rng(config.seed)
    directions = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32),
        size=(flat.shape[1], config.projections),
    )
    projected = flat @ directions
    pools = {node: set() for node in nodes}
    for column in range(config.projections):
        order = np.argsort(projected[:, column], kind="stable")
        for rank, row_value in enumerate(order):
            row = int(row_value)
            node = nodes[row]
            low = max(0, rank - config.window)
            high = min(len(nodes), rank + config.window + 1)
            pools[node].update(nodes[int(other)] for other in order[low:high] if int(other) != row)

    if tie_keys is None:
        tie_keys = {node: (node,) for node in nodes}
    result: dict[int, set[int]] = {}
    for node in nodes:
        ranked = sorted(
            (
                (_pair_profile_distance(counts[node], counts[other]), other)
                for other in pools[node]
            ),
            key=lambda item: (item[0], tie_keys[item[1]]),
        )
        result[node] = {
            other for _distance, other in ranked[: config.candidate_cap]
        }
    return result


def tied_top_k_candidates(
    nodes: list[int], distance: np.ndarray, k: int, tolerance: float = 1e-7
) -> dict[int, set[int]]:
    """Return top-k candidates while including all boundary ties.

    Including ties avoids taxon-id-dependent scientific behavior.  The actual
    candidate degree, which may exceed ``k``, is retained in work counters.
    """

    if distance.shape != (len(nodes), len(nodes)):
        raise ValueError("distance shape does not match active nodes")
    if k <= 0:
        raise ValueError("k must be positive")
    result: dict[int, set[int]] = {}
    for row, node in enumerate(nodes):
        finite = distance[row][np.isfinite(distance[row])]
        if finite.size == 0:
            result[node] = set()
            continue
        position = min(k, finite.size) - 1
        threshold = float(np.partition(finite, position)[position])
        result[node] = {
            nodes[column]
            for column in np.nonzero(distance[row] <= threshold + tolerance)[0]
            if column != row
        }
    return result


def minimum_union_k(
    left: int,
    right: int,
    nodes: list[int],
    distance: np.ndarray,
    tolerance: float = 1e-7,
) -> int:
    """Smallest top-k whose symmetrized directed graph contains a pair."""

    position = {node: index for index, node in enumerate(nodes)}
    a, b = position[left], position[right]
    target = float(distance[a, b])
    if not np.isfinite(target):
        return len(nodes)
    rank_a = 1 + int(np.count_nonzero(distance[a] < target - tolerance))
    rank_b = 1 + int(np.count_nonzero(distance[b] < target - tolerance))
    return min(rank_a, rank_b)


def true_cherries(
    adjacency: dict[int, set[int]], active: set[int]
) -> list[tuple[int, int, int]]:
    """Return ``(left, right, parent)`` for current contracted true cherries."""

    cherries: list[tuple[int, int, int]] = []
    for parent in sorted(adjacency):
        if parent in active:
            continue
        leaves = sorted(neighbor for neighbor in adjacency[parent] if neighbor in active)
        if len(leaves) == 2:
            cherries.append((leaves[0], leaves[1], parent))
        elif len(leaves) > 2 and len(active) > 3:
            raise ValueError("non-binary reference encountered before terminal state")
    return cherries


def contract_true_cherry(
    adjacency: dict[int, set[int]],
    active: set[int],
    profiles: dict[int, np.ndarray],
    members: dict[int, frozenset[int]],
    left: int,
    right: int,
    parent: int,
) -> None:
    """Contract one verified reference cherry in place."""

    if left not in active or right not in active or parent in active:
        raise ValueError("invalid active-state contraction")
    if adjacency[left] != {parent} or adjacency[right] != {parent}:
        raise ValueError("requested pair is not a pendant true cherry")
    adjacency[parent].remove(left)
    adjacency[parent].remove(right)
    del adjacency[left]
    del adjacency[right]
    active.remove(left)
    active.remove(right)
    active.add(parent)
    profiles[parent] = profiles.pop(left) + profiles.pop(right)
    members[parent] = members.pop(left) | members.pop(right)


def _canonical_split(cluster: frozenset[int], n_taxa: int) -> frozenset[int] | None:
    other = frozenset(range(n_taxa)) - cluster
    if min(len(cluster), len(other)) < 2:
        return None
    if len(cluster) < len(other):
        return cluster
    if len(other) < len(cluster):
        return other
    return min(cluster, other, key=lambda value: tuple(sorted(value)))


def simulate_oracle_reachability(
    reference_adjacency: dict[int, set[int]],
    states: np.ndarray,
    candidate_k: int,
) -> ReachabilityResult:
    """Run perfect-score contractions restricted to a sparse profile graph."""

    states = np.asarray(states, dtype=np.uint8)
    n_taxa = len(states)
    if n_taxa < 4:
        raise ValueError("at least four taxa are required")
    adjacency = {node: set(neighbors) for node, neighbors in reference_adjacency.items()}
    active = set(range(n_taxa))
    profiles = initial_profile_counts(states)
    members = {taxon: frozenset((taxon,)) for taxon in range(n_taxa)}
    resolved: set[frozenset[int]] = set()
    required_k: list[int] = []
    rounds = 0
    selected_directed = 0
    dense_pairs = 0
    maximum_degree = 0
    stalled = False

    while len(active) > 3:
        cherries = true_cherries(adjacency, active)
        if not cherries:
            stalled = True
            break
        nodes, distance = profile_distance_matrix(active, profiles)
        dense_pairs += len(nodes) * (len(nodes) - 1) // 2
        candidates = tied_top_k_candidates(nodes, distance, candidate_k)
        selected_directed += sum(len(values) for values in candidates.values())
        maximum_degree = max(maximum_degree, max(map(len, candidates.values()), default=0))
        reachable: list[tuple[int, int, int]] = []
        for left, right, parent in cherries:
            required_k.append(minimum_union_k(left, right, nodes, distance))
            if right in candidates[left] or left in candidates[right]:
                reachable.append((left, right, parent))
        if not reachable:
            stalled = True
            break
        for left, right, parent in reachable:
            cluster = members[left] | members[right]
            split = _canonical_split(cluster, n_taxa)
            if split is not None:
                resolved.add(split)
            contract_true_cherry(
                adjacency, active, profiles, members, left, right, parent
            )
        rounds += 1

    total_splits = n_taxa - 3
    complete = len(active) <= 3
    resolved_count = min(len(resolved), total_splits)
    optimistic_rf = 0.0 if complete else 1.0 - resolved_count / total_splits
    return ReachabilityResult(
        candidate_k=int(candidate_k),
        complete=complete,
        stalled=stalled,
        rounds=rounds,
        initial_taxa=n_taxa,
        final_clusters=len(active),
        resolved_splits=resolved_count,
        total_splits=total_splits,
        optimistic_rf_ceiling=float(optimistic_rf),
        selected_directed_candidates=selected_directed,
        dense_pairs_ranked=dense_pairs,
        maximum_candidate_degree=maximum_degree,
        required_k=tuple(required_k),
    )


def simulate_projection_reachability(
    reference_adjacency: dict[int, set[int]],
    states: np.ndarray,
    config: ProjectionCandidateConfig,
) -> ReachabilityResult:
    """Perfect-score contraction using only bounded projection candidates."""

    states = np.asarray(states, dtype=np.uint8)
    n_taxa = len(states)
    if n_taxa < 4:
        raise ValueError("at least four taxa are required")
    adjacency = {node: set(neighbors) for node, neighbors in reference_adjacency.items()}
    active = set(range(n_taxa))
    profiles = initial_profile_counts(states)
    members = {taxon: frozenset((taxon,)) for taxon in range(n_taxa)}
    resolved: set[frozenset[int]] = set()
    rounds = 0
    selected_directed = 0
    maximum_degree = 0
    stalled = False

    while len(active) > 3:
        cherries = true_cherries(adjacency, active)
        if not cherries:
            stalled = True
            break
        round_config = ProjectionCandidateConfig(
            projections=config.projections,
            window=config.window,
            candidate_cap=config.candidate_cap,
            seed=config.seed + rounds * 1_000_003,
        )
        candidates = projection_order_candidates(
            active,
            profiles,
            round_config,
            tie_keys={node: tuple(sorted(members[node])) for node in active},
        )
        selected_directed += sum(len(values) for values in candidates.values())
        maximum_degree = max(maximum_degree, max(map(len, candidates.values()), default=0))
        reachable = [
            (left, right, parent)
            for left, right, parent in cherries
            if right in candidates[left] or left in candidates[right]
        ]
        if not reachable:
            stalled = True
            break
        for left, right, parent in reachable:
            split = _canonical_split(members[left] | members[right], n_taxa)
            if split is not None:
                resolved.add(split)
            contract_true_cherry(
                adjacency, active, profiles, members, left, right, parent
            )
        rounds += 1

    total_splits = n_taxa - 3
    complete = len(active) <= 3
    resolved_count = min(len(resolved), total_splits)
    optimistic_rf = 0.0 if complete else 1.0 - resolved_count / total_splits
    return ReachabilityResult(
        candidate_k=config.candidate_cap,
        complete=complete,
        stalled=stalled,
        rounds=rounds,
        initial_taxa=n_taxa,
        final_clusters=len(active),
        resolved_splits=resolved_count,
        total_splits=total_splits,
        optimistic_rf_ceiling=float(optimistic_rf),
        selected_directed_candidates=selected_directed,
        dense_pairs_ranked=0,
        maximum_candidate_degree=maximum_degree,
        required_k=tuple(),
    )
