"""Shared graph validation and Newick utilities for the release runtime."""
from __future__ import annotations
import re


def _newick_label(name: str) -> str:
    # ETE3's default Newick reader treats single quotes as literal label
    # characters.  ``+`` is safe unquoted and occurs in empirical taxon names.
    if re.fullmatch(r"[A-Za-z0-9_.+\-]+", name):
        return name
    return "'" + name.replace("'", "''") + "'"


def graph_to_newick(
    adjacency: dict[int, set[int]], names: list[str], n_taxa: int
) -> str:
    degree_two = sorted(
        node for node in adjacency if node >= n_taxa and len(adjacency[node]) == 2
    )
    internal = sorted(node for node in adjacency if node >= n_taxa)
    if not internal:
        raise ValueError("Topology has no internal node")
    root = degree_two[0] if degree_two else internal[0]

    output: list[str] = []
    stack: list[tuple[str, int | str, int | None]] = [("node", root, None)]
    while stack:
        kind, value, parent = stack.pop()
        if kind == "text":
            output.append(str(value))
            continue
        node = int(value)
        if node < n_taxa:
            output.append(_newick_label(names[node]))
            continue
        children = sorted(x for x in adjacency[node] if x != parent)
        if not children:
            raise ValueError(f"Internal node {node} became empty")
        output.append("(")
        stack.append(("text", ")", None))
        for index in range(len(children) - 1, -1, -1):
            stack.append(("node", children[index], node))
            if index > 0:
                stack.append(("text", ",", None))
    return "".join(output) + ";"


def validate_topology(adjacency: dict[int, set[int]], n_taxa: int) -> None:
    if any(leaf not in adjacency or len(adjacency[leaf]) != 1 for leaf in range(n_taxa)):
        raise ValueError("Topology did not preserve all leaves at degree one")
    edge_count = sum(len(values) for values in adjacency.values()) // 2
    if edge_count != len(adjacency) - 1:
        raise ValueError("Topology is not a tree")
    visited: set[int] = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency[node] - visited)
    if len(visited) != len(adjacency):
        raise ValueError("Topology is disconnected")

