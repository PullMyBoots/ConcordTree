"""Bounded learned NNI refinement for an existing unrooted binary scaffold."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
from ete3 import Tree


PAIRINGS = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)
PAIR_CLASS_BY_RANK = np.asarray(
    (
        (-1, 0, 1, 2),
        (0, -1, 2, 1),
        (1, 2, -1, 0),
        (2, 1, 0, -1),
    ),
    dtype=np.intp,
)


def aggregate_topology_probabilities(
    values: np.ndarray, aggregation: str = "arithmetic"
) -> np.ndarray:
    """Aggregate per-quartet probabilities for the three edge topologies.

    ``log_opinion`` is the normalized geometric mean (a log opinion pool), so
    the result remains on the probability simplex and retains the existing
    margin semantics.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("values must have nonempty shape (m, 3)")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    if aggregation == "arithmetic":
        return values.mean(axis=0)
    if aggregation == "log_opinion":
        log_scores = np.log(np.clip(values, 1e-8, 1.0)).mean(axis=0)
        shifted = np.exp(log_scores - log_scores.max())
        return shifted / shifted.sum()
    raise ValueError(f"unknown aggregation {aggregation}")


@dataclass(frozen=True)
class EdgeEvidence:
    edge: tuple[int, int]
    branch_nodes: tuple[int, int, int, int]
    scores: tuple[float, float, float]
    best: int
    margin: float
    quartet_count: int


@dataclass(frozen=True)
class EdgeProbabilityPanel:
    edge: tuple[int, int]
    branch_nodes: tuple[int, int, int, int]
    mapped_probabilities: np.ndarray


@dataclass(frozen=True)
class EdgeQuartetPlan:
    """Integer-only map from one tree to all learned-NNI quartet rows."""

    edges: np.ndarray
    branch_nodes: np.ndarray
    offsets: np.ndarray
    ordered_quartets: np.ndarray
    canonical_quartets: np.ndarray


def tree_path_to_graph(path: Path, names: list[str]) -> dict[int, set[int]]:
    """Load a Newick tree into the integer graph convention used by ScaleQF."""

    # ``quoted_node_names`` is required for empirical labels containing Newick
    # punctuation; without it ETE3 retains the quote marks as part of the name.
    tree = Tree(str(path), quoted_node_names=True)
    tree.unroot()
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    leaves = tree.get_leaf_names()
    if len(leaves) != len(set(leaves)):
        raise ValueError(f"Duplicate leaf labels in {path}")
    if set(leaves) != set(name_to_idx):
        missing = sorted(set(name_to_idx) - set(leaves))[:5]
        extra = sorted(set(leaves) - set(name_to_idx))[:5]
        raise ValueError(f"Leaf-set mismatch; missing={missing}, extra={extra}")

    adjacency: dict[int, set[int]] = {idx: set() for idx in range(len(names))}
    node_ids: dict[object, int] = {}
    next_internal = len(names)
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node_ids[node] = name_to_idx[node.name]
        else:
            node_ids[node] = next_internal
            adjacency[next_internal] = set()
            next_internal += 1
    for node in tree.traverse():
        if node.up is None:
            continue
        left, right = node_ids[node], node_ids[node.up]
        adjacency[left].add(right)
        adjacency[right].add(left)

    roots = [node for node in adjacency if node >= len(names) and len(adjacency[node]) == 2]
    for root in roots:
        left, right = sorted(adjacency[root])
        adjacency[left].remove(root)
        adjacency[right].remove(root)
        adjacency[left].add(right)
        adjacency[right].add(left)
        del adjacency[root]
    return adjacency


def near_edge_representatives(
    adjacency: dict[int, set[int]],
    start: int,
    blocked: int,
    n_taxa: int,
    limit: int,
) -> list[int]:
    """Return deterministic, boundary-near leaves from one directed edge side."""

    queue = deque([(start, blocked)])
    leaves: list[int] = []
    while queue and len(leaves) < limit:
        node, parent = queue.popleft()
        if node < n_taxa:
            leaves.append(node)
            continue
        for child in sorted(x for x in adjacency[node] if x != parent):
            queue.append((child, node))
    return leaves


def directed_edge_near_far_representatives(
    adjacency: dict[int, set[int]], n_taxa: int
) -> dict[tuple[int, int], list[int]]:
    """Find one nearest and one farthest leaf on every directed edge side.

    A postorder/downward pass followed by a preorder/rerooting pass computes
    all messages in linear time on the binary tree.  Ties are deterministic;
    choosing opposite taxon-id tie breaks keeps two representatives distinct
    whenever a side contains at least two leaves.
    """

    if not adjacency:
        return {}
    if any(len(neighbors) > 3 for neighbors in adjacency.values()):
        raise ValueError("near/far messages require a binary unrooted tree")

    def extrema(candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
        by_leaf: dict[int, int] = {}
        for distance, leaf in candidates:
            previous = by_leaf.get(leaf)
            if previous is None or distance < previous:
                by_leaf[leaf] = distance
        if not by_leaf:
            raise ValueError("directed edge side contains no taxon")
        values = [(distance, leaf) for leaf, distance in by_leaf.items()]
        near = min(values, key=lambda value: (value[0], value[1]))
        far = max(values, key=lambda value: (value[0], value[1]))
        return [near] if near[1] == far[1] else [near, far]

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
    if len(parent) != len(adjacency):
        raise ValueError("adjacency is disconnected")

    down: dict[int, list[tuple[int, int]]] = {}
    for node in reversed(order):
        candidates: list[tuple[int, int]] = []
        if node < n_taxa:
            candidates.append((0, node))
        for child in adjacency[node]:
            if parent.get(child) == node:
                candidates.extend((distance + 1, leaf) for distance, leaf in down[child])
        down[node] = extrema(candidates)

    outside: dict[int, list[tuple[int, int]]] = {root: []}
    messages: dict[tuple[int, int], list[int]] = {}
    for node in order:
        children = sorted(child for child in adjacency[node] if parent.get(child) == node)
        for child in children:
            candidates = list(outside[node])
            if node < n_taxa:
                candidates.append((0, node))
            for sibling in children:
                if sibling != child:
                    candidates.extend(
                        (distance + 1, leaf) for distance, leaf in down[sibling]
                    )
            side = extrema(candidates)
            messages[(node, child)] = [leaf for _distance, leaf in side]
            messages[(child, node)] = [leaf for _distance, leaf in down[child]]
            outside[child] = [(distance + 1, leaf) for distance, leaf in side]

    expected = sum(len(neighbors) for neighbors in adjacency.values())
    if len(messages) != expected:
        raise AssertionError(f"missing directed messages: {len(messages)} != {expected}")
    return messages


def class_for_group_pairing(sorted_quartet: tuple[int, int, int, int], groups: tuple[int, int]) -> int:
    """Map a pair of original group labels to the class of a sorted quartet."""

    group_positions = {group: position for position, group in enumerate(sorted_quartet)}
    target = frozenset(group_positions[group] for group in groups)
    for class_idx, (left, _right) in enumerate(PAIRINGS):
        if frozenset(left) == target or frozenset(_right) == target:
            return class_idx
    raise AssertionError((sorted_quartet, groups))


def remap_group_order_probabilities(
    ordered_quartets: np.ndarray,
    canonical_probabilities: np.ndarray,
) -> np.ndarray:
    """Vectorize the exact scalar perfect-matching class permutation."""

    ordered = np.asarray(ordered_quartets, dtype=np.int64)
    probabilities = np.asarray(canonical_probabilities)
    if ordered.ndim != 2 or ordered.shape[1] != 4:
        raise ValueError(f"ordered_quartets must have shape (n, 4): {ordered.shape}")
    if probabilities.shape != (len(ordered), 3):
        raise ValueError(
            f"canonical_probabilities must have shape ({len(ordered)}, 3): "
            f"{probabilities.shape}"
        )
    ranks = np.count_nonzero(
        ordered[:, :, np.newaxis] > ordered[:, np.newaxis, :], axis=2
    )
    classes = np.column_stack(
        (
            PAIR_CLASS_BY_RANK[ranks[:, 0], ranks[:, 1]],
            PAIR_CLASS_BY_RANK[ranks[:, 0], ranks[:, 2]],
            PAIR_CLASS_BY_RANK[ranks[:, 0], ranks[:, 3]],
        )
    )
    if np.any(classes < 0):
        raise ValueError("each ordered quartet must contain four distinct taxa")
    return np.take_along_axis(probabilities, classes, axis=1)


def _adjacency_neighbor_array(
    adjacency: dict[int, set[int]], n_taxa: int
) -> np.ndarray:
    if not adjacency or set(adjacency) != set(range(max(adjacency) + 1)):
        raise ValueError("learned-NNI plan requires contiguous tree node ids")
    neighbors = np.full((len(adjacency), 3), -1, dtype=np.int64)
    for node, values in adjacency.items():
        ordered = sorted(values)
        expected = 1 if node < n_taxa else 3
        if len(ordered) != expected:
            raise ValueError("learned-NNI plan requires an unrooted binary tree")
        neighbors[node, : len(ordered)] = ordered
    return neighbors


def compile_edge_quartet_plan(
    adjacency: dict[int, set[int]],
    n_taxa: int,
    representatives: int = 2,
    representative_strategy: str = "near",
    plan_backend: Any | None = None,
) -> EdgeQuartetPlan:
    """Compile the exact integer row plan before probability evaluation."""

    if representative_strategy not in ("near", "near_far"):
        raise ValueError(f"unknown representative strategy {representative_strategy}")
    if representative_strategy == "near_far" and representatives != 2:
        raise ValueError("near_far currently requires exactly two representatives")
    if plan_backend is not None:
        if representative_strategy != "near":
            raise ValueError("native learned-NNI planning currently supports near only")
        raw = plan_backend.compile_nni_plan(
            _adjacency_neighbor_array(adjacency, n_taxa),
            n_taxa,
            representatives,
        )
        if not isinstance(raw, tuple) or len(raw) != 5:
            raise ValueError("native learned-NNI plan has an invalid schema")
        edges, branch_nodes, offsets, ordered, canonical = (
            np.asarray(value, dtype=np.int64) for value in raw
        )
        if edges.ndim != 2 or edges.shape[1:] != (2,):
            raise ValueError("native learned-NNI edges have an invalid shape")
        if branch_nodes.shape != (len(edges), 4):
            raise ValueError("native learned-NNI branches have an invalid shape")
        if offsets.shape != (len(edges) + 1,) or offsets[0] != 0:
            raise ValueError("native learned-NNI offsets have an invalid shape")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("native learned-NNI offsets are not monotone")
        if ordered.ndim != 2 or ordered.shape[1:] != (4,):
            raise ValueError("native learned-NNI ordered rows have an invalid shape")
        if canonical.shape != ordered.shape or offsets[-1] != len(ordered):
            raise ValueError("native learned-NNI row cardinalities are inconsistent")
        return EdgeQuartetPlan(edges, branch_nodes, offsets, ordered, canonical)

    edge_specs: list[
        tuple[tuple[int, int], tuple[int, int, int, int], int]
    ] = []
    ordered_quartets: list[tuple[int, int, int, int]] = []
    canonical_quartets: list[tuple[int, int, int, int]] = []
    directed_representatives = (
        directed_edge_near_far_representatives(adjacency, n_taxa)
        if representative_strategy == "near_far"
        else None
    )
    internal_edges = sorted(
        (u, v)
        for u in adjacency
        for v in adjacency[u]
        if u < v and u >= n_taxa and v >= n_taxa
    )
    for u, v in internal_edges:
        if len(adjacency[u]) != 3 or len(adjacency[v]) != 3:
            continue
        u_side = sorted(node for node in adjacency[u] if node != v)
        v_side = sorted(node for node in adjacency[v] if node != u)
        if len(u_side) != 2 or len(v_side) != 2:
            continue
        branches = (u_side[0], u_side[1], v_side[0], v_side[1])
        owners = (u, u, v, v)
        groups = (
            [directed_representatives[(node, owner)] for node, owner in zip(branches, owners)]
            if directed_representatives is not None
            else [
                near_edge_representatives(
                    adjacency, node, owner, n_taxa, max(1, representatives)
                )
                for node, owner in zip(branches, owners)
            ]
        )
        if any(not group for group in groups):
            continue
        quartets = [tuple(values) for values in product(*groups)]
        edge_specs.append(((u, v), branches, len(quartets)))
        ordered_quartets.extend(quartets)
        canonical_quartets.extend(tuple(sorted(values)) for values in quartets)

    edges = np.asarray([item[0] for item in edge_specs], dtype=np.int64).reshape(-1, 2)
    branches = np.asarray([item[1] for item in edge_specs], dtype=np.int64).reshape(-1, 4)
    offsets = np.empty(len(edge_specs) + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum([item[2] for item in edge_specs], dtype=np.int64)
    return EdgeQuartetPlan(
        edges,
        branches,
        offsets,
        np.asarray(ordered_quartets, dtype=np.int64).reshape(-1, 4),
        np.asarray(canonical_quartets, dtype=np.int64).reshape(-1, 4),
    )


def collect_edge_probability_panels(
    adjacency: dict[int, set[int]],
    n_taxa: int,
    predict_probabilities: Callable[[np.ndarray], np.ndarray],
    representatives: int = 2,
    batch_size: int = 65_536,
    representative_strategy: str = "near",
    plan_backend: Any | None = None,
    timing: dict[str, float] | None = None,
) -> list[EdgeProbabilityPanel]:
    """Collect topology-aligned quartet probabilities for every internal edge."""
    phase = perf_counter()
    plan = compile_edge_quartet_plan(
        adjacency,
        n_taxa,
        representatives=representatives,
        representative_strategy=representative_strategy,
        plan_backend=plan_backend,
    )
    if timing is not None:
        timing["plan_seconds"] = timing.get("plan_seconds", 0.0) + perf_counter() - phase
    if not len(plan.canonical_quartets):
        return []
    phase = perf_counter()
    probabilities = np.empty((len(plan.canonical_quartets), 3), dtype=np.float64)
    for start in range(0, len(plan.canonical_quartets), batch_size):
        end = min(start + batch_size, len(plan.canonical_quartets))
        chunk = np.asarray(
            predict_probabilities(plan.canonical_quartets[start:end]), dtype=np.float64
        )
        if chunk.shape != (end - start, 3):
            raise ValueError(f"Probability shape mismatch: {chunk.shape}")
        probabilities[start:end] = chunk
    if timing is not None:
        timing["predict_seconds"] = timing.get("predict_seconds", 0.0) + perf_counter() - phase

    phase = perf_counter()
    mapped_all = remap_group_order_probabilities(
        plan.ordered_quartets, probabilities
    )
    if timing is not None:
        timing["remap_seconds"] = timing.get("remap_seconds", 0.0) + perf_counter() - phase
    phase = perf_counter()
    panels: list[EdgeProbabilityPanel] = []
    for index in range(len(plan.edges)):
        start, end = int(plan.offsets[index]), int(plan.offsets[index + 1])
        panels.append(
            EdgeProbabilityPanel(
                edge=tuple(int(value) for value in plan.edges[index]),
                branch_nodes=tuple(int(value) for value in plan.branch_nodes[index]),
                mapped_probabilities=mapped_all[start:end].copy(),
            )
        )
    if timing is not None:
        timing["panel_seconds"] = timing.get("panel_seconds", 0.0) + perf_counter() - phase
    return panels


def edge_evidence_from_panels(
    panels: list[EdgeProbabilityPanel], aggregation: str = "arithmetic"
) -> list[EdgeEvidence]:
    """Convert reusable probability panels into NNI edge evidence."""

    evidence: list[EdgeEvidence] = []
    for panel in panels:
        scores = aggregate_topology_probabilities(panel.mapped_probabilities, aggregation)
        order = np.argsort(scores)
        best = int(order[-1])
        margin = float(scores[best] - scores[0]) if best != 0 else 0.0
        evidence.append(
            EdgeEvidence(
                edge=panel.edge,
                branch_nodes=panel.branch_nodes,
                scores=tuple(float(value) for value in scores),
                best=best,
                margin=margin,
                quartet_count=len(panel.mapped_probabilities),
            )
        )
    return evidence


def collect_edge_evidence(
    adjacency: dict[int, set[int]],
    n_taxa: int,
    predict_probabilities: Callable[[np.ndarray], np.ndarray],
    representatives: int = 2,
    batch_size: int = 65_536,
    aggregation: str = "arithmetic",
    representative_strategy: str = "near",
    plan_backend: Any | None = None,
    timing: dict[str, float] | None = None,
    compact_reduction: bool = False,
    compact_min_margin: float | None = None,
) -> list[EdgeEvidence]:
    """Score all NNI alternatives with bounded representative quartets."""

    if compact_reduction:
        if aggregation != "arithmetic":
            raise ValueError("compact NNI reduction currently requires arithmetic aggregation")
        if plan_backend is None or not hasattr(plan_backend, "reduce_nni_arithmetic"):
            raise ValueError("compact NNI reduction requires its native plan backend")
        phase = perf_counter()
        plan = compile_edge_quartet_plan(
            adjacency,
            n_taxa,
            representatives=representatives,
            representative_strategy=representative_strategy,
            plan_backend=plan_backend,
        )
        if timing is not None:
            timing["plan_seconds"] = timing.get("plan_seconds", 0.0) + perf_counter() - phase
        if not len(plan.canonical_quartets):
            return []
        phase = perf_counter()
        probabilities = np.empty((len(plan.canonical_quartets), 3), dtype=np.float64)
        for start in range(0, len(plan.canonical_quartets), batch_size):
            end = min(start + batch_size, len(plan.canonical_quartets))
            chunk = np.asarray(
                predict_probabilities(plan.canonical_quartets[start:end]),
                dtype=np.float64,
            )
            if chunk.shape != (end - start, 3):
                raise ValueError(f"Probability shape mismatch: {chunk.shape}")
            probabilities[start:end] = chunk
        if timing is not None:
            timing["predict_seconds"] = timing.get("predict_seconds", 0.0) + perf_counter() - phase
        phase = perf_counter()
        scores = np.asarray(
            plan_backend.reduce_nni_arithmetic(
                plan.ordered_quartets, probabilities, plan.offsets
            ),
            dtype=np.float64,
        )
        if timing is not None:
            timing["compact_reduction_seconds"] = (
                timing.get("compact_reduction_seconds", 0.0) + perf_counter() - phase
            )
        phase = perf_counter()
        best_by_edge = np.argsort(scores, axis=1)[:, -1]
        margins = np.zeros(len(scores), dtype=np.float64)
        nonzero = best_by_edge != 0
        rows = np.nonzero(nonzero)[0]
        margins[rows] = scores[rows, best_by_edge[rows]] - scores[rows, 0]
        eligible = nonzero
        if compact_min_margin is not None:
            eligible = eligible & (margins >= compact_min_margin)
        evidence: list[EdgeEvidence] = []
        for index in np.nonzero(eligible)[0]:
            values = scores[index]
            best = int(best_by_edge[index])
            evidence.append(
                EdgeEvidence(
                    edge=tuple(int(value) for value in plan.edges[index]),
                    branch_nodes=tuple(
                        int(value) for value in plan.branch_nodes[index]
                    ),
                    scores=tuple(float(value) for value in values),
                    best=best,
                    margin=float(margins[index]),
                    quartet_count=int(plan.offsets[index + 1] - plan.offsets[index]),
                )
            )
        if timing is not None:
            timing["evidence_seconds"] = timing.get("evidence_seconds", 0.0) + perf_counter() - phase
            timing["_latest_edge_count"] = float(len(scores))
            timing["_latest_quartet_count"] = float(len(plan.canonical_quartets))
        return evidence

    panels = collect_edge_probability_panels(
        adjacency,
        n_taxa,
        predict_probabilities,
        representatives=representatives,
        batch_size=batch_size,
        representative_strategy=representative_strategy,
        plan_backend=plan_backend,
        timing=timing,
    )
    phase = perf_counter()
    result = edge_evidence_from_panels(panels, aggregation=aggregation)
    if timing is not None:
        timing["evidence_seconds"] = timing.get("evidence_seconds", 0.0) + perf_counter() - phase
    return result


def apply_independent_nni(
    adjacency: dict[int, set[int]],
    evidence: list[EdgeEvidence],
    min_margin: float,
) -> list[EdgeEvidence]:
    """Apply a maximum-greedy set of non-adjacent, confident NNI proposals."""

    selected: list[EdgeEvidence] = []
    occupied: set[int] = set()
    for item in sorted(evidence, key=lambda value: (-value.margin, value.edge)):
        u, v = item.edge
        if item.best == 0 or item.margin < min_margin or u in occupied or v in occupied:
            continue
        a_node, b_node, c_node, d_node = item.branch_nodes
        displaced_v = c_node if item.best == 1 else d_node
        if b_node not in adjacency[u] or displaced_v not in adjacency[v]:
            continue
        adjacency[u].remove(b_node)
        adjacency[b_node].remove(u)
        adjacency[v].remove(displaced_v)
        adjacency[displaced_v].remove(v)
        adjacency[u].add(displaced_v)
        adjacency[displaced_v].add(u)
        adjacency[v].add(b_node)
        adjacency[b_node].add(v)
        occupied.update((u, v))
        selected.append(item)
    return selected


def refine_learned_nni(
    adjacency: dict[int, set[int]],
    n_taxa: int,
    predict_probabilities: Callable[[np.ndarray], np.ndarray],
    passes: int = 3,
    representatives: int = 2,
    min_margin: float = 0.10,
    aggregation: str = "arithmetic",
    representative_strategy: str = "near",
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    stop_moves_at_most: int | None = None,
    plan_backend: Any | None = None,
    compact_reduction: bool = False,
) -> dict[str, object]:
    if stop_moves_at_most is not None and stop_moves_at_most < 0:
        raise ValueError("stop_moves_at_most must be nonnegative")
    history: list[dict[str, object]] = []
    timing: dict[str, float] = {}
    for pass_idx in range(passes):
        evidence = collect_edge_evidence(
            adjacency,
            n_taxa,
            predict_probabilities,
            representatives=representatives,
            aggregation=aggregation,
            representative_strategy=representative_strategy,
            plan_backend=plan_backend,
            timing=timing,
            compact_reduction=compact_reduction,
            compact_min_margin=min_margin if compact_reduction else None,
        )
        phase = perf_counter()
        selected = apply_independent_nni(adjacency, evidence, min_margin=min_margin)
        timing["apply_seconds"] = timing.get("apply_seconds", 0.0) + perf_counter() - phase
        history.append(
            {
                "pass": pass_idx + 1,
                "edges_scored": int(timing.get("_latest_edge_count", len(evidence))),
                "quartets_scored": int(
                    timing.get(
                        "_latest_quartet_count",
                        sum(item.quartet_count for item in evidence),
                    )
                ),
                "moves": len(selected),
                "selected": [
                    {
                        "edge": list(item.edge),
                        "scores": list(item.scores),
                        "best": item.best,
                        "margin": item.margin,
                    }
                    for item in selected
                ],
            }
        )
        if progress_callback is not None:
            progress_callback(history[-1])
        if not selected or (
            stop_moves_at_most is not None and len(selected) <= stop_moves_at_most
        ):
            break
    timing.pop("_latest_edge_count", None)
    timing.pop("_latest_quartet_count", None)
    return {
        "passes": history,
        "moves": sum(int(item["moves"]) for item in history),
        "timing": timing,
    }
