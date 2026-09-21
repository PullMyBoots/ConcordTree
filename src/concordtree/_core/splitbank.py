"""Compatible split-bank decoding and topology reconstruction."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np

from concordtree._core.scaleqf import TreeNode


def splits_compatible(left: int, right: int, total: int) -> bool:
    left_other, right_other = total ^ left, total ^ right
    return any(
        value == 0
        for value in (
            left & right,
            left & right_other,
            left_other & right,
            left_other & right_other,
        )
    )


def greedy_compatible(
    counts: Counter[int], total: int, min_votes: int, target: int
) -> frozenset[int]:
    selected: list[int] = []
    for split, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count < min_votes:
            continue
        if all(splits_compatible(split, existing, total) for existing in selected):
            selected.append(split)
            if len(selected) >= target:
                break
    return frozenset(selected)


def anchored_completion(
    counts: Counter[int],
    anchor: frozenset[int],
    total: int,
    min_votes: int,
    target: int,
) -> frozenset[int]:
    """Lock high-vote splits, then preserve compatible anchor structure."""

    selected = list(greedy_compatible(counts, total, min_votes, target))
    ranked_rest = sorted(
        (split for split in counts if split not in selected),
        key=lambda split: (-(split in anchor), -counts[split], split),
    )
    for split in ranked_rest:
        if all(splits_compatible(split, existing, total) for existing in selected):
            selected.append(split)
            if len(selected) >= target:
                break
    return frozenset(selected)


def split_set_to_tree(splits: frozenset[int], n_taxa: int, anchor_leaf: int = 0) -> TreeNode:
    """Reconstruct a possibly unresolved tree from compatible unrooted splits."""

    total = (1 << n_taxa) - 1
    anchor_bit = 1 << anchor_leaf
    clusters: set[int] = set()
    for split in splits:
        cluster = (total ^ split) if split & anchor_bit else split
        if cluster.bit_count() < 2 or cluster == total:
            continue
        clusters.add(cluster)
    ordered = sorted(clusters, key=lambda value: (value.bit_count(), value))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if left & right and left & right not in (left, right):
                raise ValueError("Oriented split clusters are not laminar")

    def build(cluster: int) -> TreeNode:
        contained = [value for value in clusters if value != cluster and value & cluster == value]
        maximal = [
            value
            for value in contained
            if not any(value != other and value & other == value for other in contained)
        ]
        covered = 0
        children: list[TreeNode] = []
        for child_cluster in sorted(maximal, key=lambda value: (value.bit_count(), value)):
            covered |= child_cluster
            children.append(build(child_cluster))
        for leaf in range(n_taxa):
            bit = 1 << leaf
            if cluster & bit and not covered & bit:
                children.append(TreeNode(leaf=leaf))
        if len(children) == 1:
            return children[0]
        return TreeNode(children=children)

    return build(total)


def greedy_ranked_compatible(
    ranked_splits: list[int], n_taxa: int, target: int | None = None,
) -> frozenset[int]:
    """Greedily select compatible splits with vectorized bitset checks.

    The input order is the priority order.  Orienting every split away from a
    common anchor reduces unrooted compatibility to rooted-cluster laminarity.
    This removes the Python pairwise loop, while preserving the exact greedy
    decision made by repeated :func:`splits_compatible` calls.
    """

    selector = LaminarSplitSelector(n_taxa, target)
    for split in ranked_splits:
        if selector.full:
            break
        selector.add(split)
    return selector.splits


class LaminarSplitSelector:
    """Incremental exact selector for one anchored laminar split family.

    Keeping the packed accepted-prefix state across ranking phases avoids
    rescanning unanimous/repeated splits during model-bank and anchor
    completion.  ``add`` implements the same disjoint-or-nested predicate as
    :func:`splits_compatible` after orienting every split away from leaf zero.
    """

    def __init__(
        self,
        n_taxa: int,
        target: int | None = None,
        *,
        backend: Any | None = None,
    ) -> None:
        if n_taxa < 4:
            raise ValueError("n_taxa must be at least four")
        if target is None:
            target = n_taxa - 3
        if target < 0 or target > n_taxa - 3:
            raise ValueError("invalid target split count")
        self.n_taxa = n_taxa
        self.target = target
        self.total = (1 << n_taxa) - 1
        self.width = (n_taxa + 63) // 64
        self._words = np.empty((target, self.width), dtype=np.uint64)
        self._original: list[int] = []
        self._seen: set[int] = set()
        self._backend = backend

    @property
    def full(self) -> bool:
        return len(self._original) >= self.target

    @property
    def splits(self) -> frozenset[int]:
        return frozenset(self._original)

    @property
    def ordered_splits(self) -> tuple[int, ...]:
        return tuple(self._original)

    def add(self, split: int) -> bool:
        if self.full or split in self._seen:
            return False
        self._seen.add(split)
        cluster = (self.total ^ split) if split & 1 else split
        if cluster.bit_count() < 2 or (self.total ^ cluster).bit_count() < 2:
            return False
        words = np.frombuffer(
            int(cluster).to_bytes(self.width * 8, "little"), dtype="<u8"
        )
        size = len(self._original)
        if size:
            old = self._words[:size]
            intersection = np.bitwise_and(old, words)
            compatible = (
                ~np.any(intersection, axis=1)
                | np.all(intersection == words, axis=1)
                | np.all(intersection == old, axis=1)
            )
            if not bool(np.all(compatible)):
                return False
        self._words[size] = words
        self._original.append(split)
        return True

    def extend(self, ranked_splits: Iterable[int]) -> tuple[int, ...]:
        """Consume one priority stream with the exact scalar greedy semantics.

        When the optional native backend is present, candidate clusters are
        packed once and the sequential disjoint-or-nested scan runs in C++.
        The dependency between acceptances remains ordered and exact.
        """

        if self._backend is None:
            accepted: list[int] = []
            for split in ranked_splits:
                if self.full:
                    break
                if self.add(split):
                    accepted.append(split)
            return tuple(accepted)

        eligible_splits: list[int] = []
        eligible_words: list[np.ndarray] = []
        for split in ranked_splits:
            if self.full:
                break
            if split in self._seen:
                continue
            self._seen.add(split)
            cluster = (self.total ^ split) if split & 1 else split
            if cluster.bit_count() < 2 or (self.total ^ cluster).bit_count() < 2:
                continue
            eligible_splits.append(split)
            eligible_words.append(
                np.frombuffer(
                    int(cluster).to_bytes(self.width * 8, "little"),
                    dtype="<u8",
                )
            )
        if not eligible_splits:
            return ()
        candidates = np.ascontiguousarray(np.stack(eligible_words), dtype=np.uint64)
        size = len(self._original)
        hierarchy_selector = getattr(
            self._backend, "greedy_laminar_accept_hierarchy", None
        )
        keep = np.asarray(
            hierarchy_selector(
                self._words[:size], candidates, self.target - size, self.n_taxa
            )
            if hierarchy_selector is not None
            else self._backend.greedy_laminar_accept(
                self._words[:size], candidates, self.target - size
            ),
            dtype=np.uint8,
        )
        if keep.shape != (len(eligible_splits),) or np.any(keep > 1):
            raise AssertionError("native laminar selector returned an invalid mask")
        selected_rows = np.flatnonzero(keep)
        if size + len(selected_rows) > self.target:
            raise AssertionError("native laminar selector exceeded the target")
        self._words[size : size + len(selected_rows)] = candidates[selected_rows]
        accepted = tuple(eligible_splits[int(row)] for row in selected_rows)
        self._original.extend(accepted)
        return accepted


def split_set_to_tree_indexed(
    splits: frozenset[int], n_taxa: int, anchor_leaf: int = 0,
    *, binary_complete: bool = False, backend: Any | None = None,
) -> TreeNode:
    """Reconstruct a tree without scanning every cluster for every parent.

    Compatible splits are oriented away from ``anchor_leaf``.  Processing the
    resulting laminar clusters from large to small lets a per-leaf owner table
    identify the immediate parent.  Runtime is proportional to the sum of
    oriented cluster cardinalities; with dense integer split masks this still
    has a quadratic caterpillar worst case, but avoids the older unconditional
    all-pairs cluster scan.
    """

    total = (1 << n_taxa) - 1
    anchor_bit = 1 << anchor_leaf
    clusters = {
        (total ^ split) if split & anchor_bit else split
        for split in splits
        if 2 <= min(split.bit_count(), (total ^ split).bit_count())
    }
    ordered = sorted(clusters, key=lambda value: (-value.bit_count(), value))
    hierarchy_builder = (
        getattr(backend, "laminar_parent_indices", None)
        if backend is not None
        else None
    )
    if hierarchy_builder is not None and ordered:
        width = (n_taxa + 63) // 64
        packed = np.empty((len(ordered), width), dtype=np.uint64)
        packed_bytes = packed.view(np.uint8).reshape(len(ordered), width * 8)
        for row, cluster in enumerate(ordered):
            packed_bytes[row] = np.frombuffer(
                int(cluster).to_bytes(width * 8, "little"), dtype=np.uint8
            )
        parent_rows, leaf_owners = hierarchy_builder(packed, n_taxa)
        parent_rows = np.asarray(parent_rows, dtype=np.int64)
        leaf_owners = np.asarray(leaf_owners, dtype=np.int64)
        root_row = len(ordered)
        if parent_rows.shape != (len(ordered),) or leaf_owners.shape != (n_taxa,):
            raise AssertionError("native laminar hierarchy returned invalid dimensions")
        if (
            np.any(parent_rows < 0)
            or np.any(parent_rows > root_row)
            or np.any(leaf_owners < 0)
            or np.any(leaf_owners > root_row)
        ):
            raise AssertionError("native laminar hierarchy returned invalid owners")
        child_rows: list[list[int]] = [[] for _ in range(root_row + 1)]
        child_leaves: list[list[int]] = [[] for _ in range(root_row + 1)]
        for row, parent in enumerate(parent_rows.tolist()):
            child_rows[parent].append(row)
        for leaf, owner in enumerate(leaf_owners.tolist()):
            child_leaves[owner].append(leaf)
        built_rows: list[TreeNode | None] = [None] * len(ordered)

        def materialize(row: int) -> TreeNode:
            rows = sorted(
                child_rows[row],
                key=lambda child: (
                    ordered[child].bit_count(),
                    ordered[child],
                ),
            )
            materialized = [
                *(built_rows[child] for child in rows),
                *(TreeNode(leaf=leaf) for leaf in child_leaves[row]),
            ]
            if any(child is None for child in materialized):
                raise AssertionError("native laminar hierarchy build order changed")
            if binary_complete:
                while len(materialized) > 2:
                    first, second = materialized[:2]
                    materialized = [
                        TreeNode(children=[first, second]),
                        *materialized[2:],
                    ]
            return TreeNode(children=materialized)  # type: ignore[arg-type]

        for row in range(len(ordered) - 1, -1, -1):
            built_rows[row] = materialize(row)
        return materialize(root_row)

    root = total
    owner: list[int] = [root] * n_taxa
    children: dict[int, list[int | TreeNode]] = {root: []}
    for cluster in ordered:
        bit = cluster & -cluster
        representative = bit.bit_length() - 1
        parent = owner[representative]
        if cluster & parent != cluster:
            raise ValueError("Oriented split clusters are not laminar")
        children.setdefault(parent, []).append(cluster)
        children[cluster] = []
        remaining = cluster
        while remaining:
            leaf_bit = remaining & -remaining
            leaf = leaf_bit.bit_length() - 1
            if owner[leaf] != parent:
                raise ValueError("Oriented split clusters are not laminar")
            owner[leaf] = cluster
            remaining ^= leaf_bit
    for leaf, parent in enumerate(owner):
        children[parent].append(TreeNode(leaf=leaf))

    built: dict[int, TreeNode] = {}
    for cluster in reversed([root, *ordered]):
        # Preserve the scalar decoder's observable child order exactly:
        # immediate cluster children first in (cardinality, mask) order, then
        # uncovered leaves in taxon order.  Internal node identifiers assigned
        # by tree_to_graph are consumed by deterministic panel construction, so
        # topology equivalence alone is not a sufficient execution invariant.
        child_clusters = sorted(
            (child for child in children[cluster] if isinstance(child, int)),
            key=lambda value: (value.bit_count(), value),
        )
        child_leaves = sorted(
            (child for child in children[cluster] if isinstance(child, TreeNode)),
            key=lambda node: int(node.leaf),
        )
        materialized = [*(built[child] for child in child_clusters), *child_leaves]
        if binary_complete:
            while len(materialized) > 2:
                first, second = materialized[:2]
                materialized = [TreeNode(children=[first, second]), *materialized[2:]]
        built[cluster] = TreeNode(children=materialized)
    return built[root]
