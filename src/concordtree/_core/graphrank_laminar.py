"""Taxon-keyed multiscale graph proposals for laminar split pursuit."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from concordtree._core.scaleqf import TreeNode
from concordtree._core.splitbank import split_set_to_tree_indexed


@dataclass(frozen=True)
class ProposalBank:
    names: tuple[str, ...]
    candidates: frozenset[int]
    graph_edges: tuple[tuple[int, int, float], ...]
    orders: tuple[tuple[int, ...], ...]
    landmark_count: int


def _priority(name: str, salt: str) -> tuple[bytes, str]:
    return hashlib.sha256(f"graphrank-laminar-v0\0{salt}\0{name}".encode()).digest(), name


def canonical_split(mask: int, n_taxa: int) -> int | None:
    total = (1 << n_taxa) - 1
    mask &= total
    other = total ^ mask
    if min(mask.bit_count(), other.bit_count()) < 2:
        return None
    if mask.bit_count() < other.bit_count():
        return mask
    if other.bit_count() < mask.bit_count():
        return other
    return min(mask, other)


def _pair_distance(states: np.ndarray, left: int, right: int) -> float:
    a, b = states[left], states[right]
    valid = (a < 4) & (b < 4)
    if not np.any(valid):
        return 1.0
    return float(np.count_nonzero(a[valid] != b[valid]) / int(valid.sum()))


def _landmark_embedding(states: np.ndarray, landmarks: list[int]) -> np.ndarray:
    result = np.empty((len(states), len(landmarks)), dtype=np.float32)
    for column, landmark in enumerate(landmarks):
        target = states[landmark]
        valid = (states < 4) & (target[None, :] < 4)
        denominator = valid.sum(axis=1)
        mismatch = ((states != target[None, :]) & valid).sum(axis=1)
        result[:, column] = np.divide(
            mismatch,
            denominator,
            out=np.ones(len(states), dtype=np.float64),
            where=denominator > 0,
        )
    return result


def _direction(landmark_names: list[str], index: int) -> np.ndarray:
    values = []
    for name in landmark_names:
        digest = hashlib.sha256(
            f"graphrank-laminar-v0\0direction\0{index}\0{name}".encode()
        ).digest()
        integer = int.from_bytes(digest[:8], "little")
        values.append((integer / 2**64) * 2.0 - 1.0)
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else np.ones_like(vector) / math.sqrt(len(vector))


def _order(values: np.ndarray, names: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(sorted(range(len(names)), key=lambda index: (float(values[index]), names[index])))


def _mst_splits(
    n_taxa: int, edges: list[tuple[int, int, float]], names: tuple[str, ...]
) -> set[int]:
    parent = list(range(n_taxa))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    tree_edges: list[tuple[int, int]] = []
    for left, right, weight in sorted(
        edges, key=lambda item: (item[2], names[item[0]], names[item[1]])
    ):
        a, b = find(left), find(right)
        if a == b:
            continue
        parent[b] = a
        tree_edges.append((left, right))
        if len(tree_edges) == n_taxa - 1:
            break
    if len(tree_edges) != n_taxa - 1:
        raise ValueError("proposal graph is disconnected")

    adjacency = {node: set() for node in range(n_taxa)}
    for left, right in tree_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    root = min(range(n_taxa), key=lambda index: names[index])
    rooted_parent = {root: None}
    traversal = [root]
    for node in traversal:
        for neighbor in sorted(adjacency[node], key=lambda index: names[index]):
            if neighbor == rooted_parent[node]:
                continue
            rooted_parent[neighbor] = node
            traversal.append(neighbor)
    subtree = {node: 1 << node for node in range(n_taxa)}
    for node in reversed(traversal[1:]):
        ancestor = rooted_parent[node]
        assert ancestor is not None
        subtree[ancestor] |= subtree[node]
    result = set()
    for node in traversal[1:]:
        split = canonical_split(subtree[node], n_taxa)
        if split is not None:
            result.add(split)
    return result


def propose_split_bank(
    names: list[str] | tuple[str, ...],
    states: np.ndarray,
    *,
    max_landmarks: int = 16,
    projection_count: int = 16,
    neighbor_window: int = 1,
) -> ProposalBank:
    """Create permutation-equivariant split proposals without topology access."""

    if len(names) != len(states) or len(names) < 4:
        raise ValueError("names/states mismatch or fewer than four taxa")
    if len(set(names)) != len(names):
        raise ValueError("taxon labels must be unique")
    if max_landmarks <= 0 or projection_count < 0 or neighbor_window <= 0:
        raise ValueError("invalid proposal configuration")

    input_names = tuple(str(name) for name in names)
    canonical_input = sorted(range(len(input_names)), key=lambda index: input_names[index])
    ordered_names = tuple(input_names[index] for index in canonical_input)
    ordered_states = np.asarray(states, dtype=np.uint8)[canonical_input]
    n_taxa = len(ordered_names)

    landmark_count = min(n_taxa, max_landmarks)
    landmarks = sorted(
        range(n_taxa), key=lambda index: _priority(ordered_names[index], "landmark")
    )[:landmark_count]
    embedding = _landmark_embedding(ordered_states, landmarks)
    orders: list[tuple[int, ...]] = [
        _order(embedding[:, column], ordered_names) for column in range(landmark_count)
    ]
    landmark_names = [ordered_names[index] for index in landmarks]
    for projection in range(projection_count):
        orders.append(_order(embedding @ _direction(landmark_names, projection), ordered_names))

    edge_pairs: set[tuple[int, int]] = set()
    for order in orders:
        for position, left in enumerate(order):
            for offset in range(1, neighbor_window + 1):
                if position + offset >= n_taxa:
                    break
                right = order[position + offset]
                edge_pairs.add((min(left, right), max(left, right)))
    for taxon in range(n_taxa):
        nearest = min(
            landmarks,
            key=lambda landmark: (
                float(embedding[taxon, landmarks.index(landmark)]),
                ordered_names[landmark],
            ),
        )
        if taxon != nearest:
            edge_pairs.add((min(taxon, nearest), max(taxon, nearest)))
    for left, right in zip(landmarks, landmarks[1:]):
        edge_pairs.add((min(left, right), max(left, right)))
    edges = [
        (left, right, _pair_distance(ordered_states, left, right))
        for left, right in sorted(edge_pairs)
    ]

    candidates: set[int] = set()
    for order in orders:
        mask = 0
        for position, taxon in enumerate(order[:-1], start=1):
            mask |= 1 << taxon
            if 2 <= position <= n_taxa - 2:
                split = canonical_split(mask, n_taxa)
                if split is not None:
                    candidates.add(split)
    adjacency = {node: [] for node in range(n_taxa)}
    for left, right, weight in edges:
        adjacency[left].append((weight, ordered_names[right], right))
        adjacency[right].append((weight, ordered_names[left], left))
    for taxon in range(n_taxa):
        ranked = [item[2] for item in sorted(adjacency[taxon])]
        mask = 1 << taxon
        for neighbor in ranked[: min(4, len(ranked))]:
            mask |= 1 << neighbor
            split = canonical_split(mask, n_taxa)
            if split is not None:
                candidates.add(split)
    candidates.update(_mst_splits(n_taxa, edges, ordered_names))
    return ProposalBank(
        names=ordered_names,
        candidates=frozenset(candidates),
        graph_edges=tuple(edges),
        orders=tuple(orders),
        landmark_count=landmark_count,
    )


def _leaf_set(node: TreeNode) -> frozenset[int]:
    if node.leaf is not None:
        return frozenset((node.leaf,))
    return frozenset().union(*(_leaf_set(child) for child in node.children))


def _binarize_with_mask(node: TreeNode) -> tuple[TreeNode, int]:
    """Binarize with cached leaf masks, preserving the historical order.

    Use an explicit postorder stack because valid high-taxon laminar families
    can form caterpillars much deeper than Python's recursion limit.
    """

    def key(item: tuple[TreeNode, int]) -> tuple[int, int]:
        mask = item[1]
        return ((mask & -mask).bit_length() - 1, mask.bit_count())

    completed: dict[int, tuple[TreeNode, int]] = {}
    stack: list[tuple[TreeNode, bool]] = [(node, False)]
    while stack:
        current, expanded = stack.pop()
        if current.leaf is not None:
            completed[id(current)] = (TreeNode(leaf=current.leaf), 1 << current.leaf)
            continue
        if not expanded:
            stack.append((current, True))
            stack.extend((child, False) for child in reversed(current.children))
            continue
        children = [completed[id(child)] for child in current.children]
        # Child masks are nonempty and pairwise disjoint, so their least-set
        # taxa are unique.  After the two least keys are merged, the result
        # retains the globally least taxon and is therefore still the least
        # key.  The historical "merge two, sort all again" loop is consequently
        # identical to one initial sort followed by a left fold.  This removes
        # an accidental O(k^2 log k) completion path without changing one
        # selected pair or one output split.
        children.sort(key=key)
        if len(children) > 2:
            first = children[0]
            for second in children[1:-1]:
                first = (
                    TreeNode(children=[first[0], second[0]]),
                    first[1] | second[1],
                )
            children = [first, children[-1]]
        mask = 0
        for _child, child_mask in children:
            mask |= child_mask
        completed[id(current)] = (
            TreeNode(children=[child for child, _mask in children]),
            mask,
        )
    return completed[id(node)]


def _binarize(node: TreeNode) -> TreeNode:
    return _binarize_with_mask(node)[0]


def complete_compatible_splits(
    splits: frozenset[int],
    n_taxa: int,
    *,
    backend: Any | None = None,
) -> frozenset[int]:
    """Deterministically binarize a compatible split subset without an anchor tree."""

    root = _binarize(
        split_set_to_tree_indexed(
            splits,
            n_taxa,
            anchor_leaf=0,
            backend=backend,
        )
    )
    total = (1 << n_taxa) - 1
    result: set[int] = set()

    masks: dict[int, int] = {}
    stack: list[tuple[TreeNode, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if node.leaf is not None:
            masks[id(node)] = 1 << node.leaf
            continue
        if not expanded:
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(node.children))
            continue
        mask = 0
        for child in node.children:
            child_mask = masks[id(child)]
            mask |= child_mask
            split = canonical_split(child_mask, n_taxa)
            if split is not None:
                result.add(split)
        masks[id(node)] = mask

    if masks[id(root)] != total:
        raise AssertionError("completed tree lost taxa")
    if len(result) != n_taxa - 3:
        raise AssertionError(f"completion has {len(result)} splits, expected {n_taxa - 3}")
    return frozenset(result)
