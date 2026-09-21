"""Truth-used monotone NNI ceiling utilities for development diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from concordtree._core.eapc_reachability import split_bitmasks
from concordtree._core.graphrank_laminar import canonical_split
from concordtree._core.learned_nni import EdgeEvidence, apply_independent_nni


@dataclass(frozen=True)
class OracleNNIPass:
    opportunities: int
    moves: int
    normalized_rf: float


def directed_edge_leaf_masks(
    adjacency: dict[int, set[int]], n_taxa: int
) -> dict[tuple[int, int], int]:
    """Return the taxon mask on the first-node side of every directed edge."""

    if not adjacency:
        return {}
    root = min(adjacency)
    parent: dict[int, int | None] = {root: None}
    order = [root]
    for node in order:
        for neighbor in sorted(adjacency[node]):
            if neighbor == parent[node]:
                continue
            if neighbor in parent:
                raise ValueError("adjacency is not a tree")
            parent[neighbor] = node
            order.append(neighbor)
    if len(order) != len(adjacency):
        raise ValueError("adjacency is disconnected")
    subtree: dict[int, int] = {}
    for node in reversed(order):
        mask = (1 << node) if node < n_taxa else 0
        for child in adjacency[node]:
            if parent.get(child) == node:
                mask |= subtree[child]
        subtree[node] = mask
    total = subtree[root]
    messages: dict[tuple[int, int], int] = {}
    for node in order[1:]:
        ancestor = parent[node]
        assert ancestor is not None
        messages[(node, ancestor)] = subtree[node]
        messages[(ancestor, node)] = total ^ subtree[node]
    return messages


def normalized_split_distance(current: frozenset[int], reference: frozenset[int]) -> float:
    denominator = len(current) + len(reference)
    return 0.0 if denominator == 0 else len(current.symmetric_difference(reference)) / denominator


def monotone_oracle_nni_pass(
    adjacency: dict[int, set[int]], n_taxa: int, reference_splits: frozenset[int]
) -> OracleNNIPass:
    """Apply a maximal independent set of NNIs that insert a true split."""

    masks = directed_edge_leaf_masks(adjacency, n_taxa)
    proposals: list[EdgeEvidence] = []
    internal_edges = sorted(
        (u, v)
        for u in adjacency
        for v in adjacency[u]
        if u < v and u >= n_taxa and v >= n_taxa
    )
    for u, v in internal_edges:
        u_side = sorted(node for node in adjacency[u] if node != v)
        v_side = sorted(node for node in adjacency[v] if node != u)
        if len(u_side) != 2 or len(v_side) != 2:
            continue
        a, b, c, d = (*u_side, *v_side)
        group_masks = [masks[(node, owner)] for node, owner in zip((a, b, c, d), (u, u, v, v))]
        old = canonical_split(group_masks[0] | group_masks[1], n_taxa)
        if old is None or old in reference_splits:
            continue
        alternatives = (
            canonical_split(group_masks[0] | group_masks[2], n_taxa),
            canonical_split(group_masks[0] | group_masks[3], n_taxa),
        )
        best = next(
            (index + 1 for index, split in enumerate(alternatives) if split in reference_splits),
            None,
        )
        if best is None:
            continue
        scores = [0.0, 0.0, 0.0]
        scores[best] = 1.0
        proposals.append(
            EdgeEvidence(
                edge=(u, v),
                branch_nodes=(a, b, c, d),
                scores=tuple(scores),
                best=best,
                margin=1.0,
                quartet_count=0,
            )
        )
    selected = apply_independent_nni(adjacency, proposals, min_margin=0.5)
    current = split_bitmasks(adjacency, n_taxa)
    return OracleNNIPass(
        opportunities=len(proposals),
        moves=len(selected),
        normalized_rf=normalized_split_distance(current, reference_splits),
    )


def run_monotone_oracle_nni(
    adjacency: dict[int, set[int]],
    n_taxa: int,
    reference_splits: frozenset[int],
    max_passes: int,
) -> list[OracleNNIPass]:
    history: list[OracleNNIPass] = []
    for _ in range(max_passes):
        item = monotone_oracle_nni_pass(adjacency, n_taxa, reference_splits)
        history.append(item)
        if item.moves == 0:
            break
    return history
