"""Read-only alignment sampling and leakage-safe dataset metadata utilities.

The project datasets are immutable inputs.  This module never writes beside an
alignment; callers choose an output directory outside ``data/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


STATE_CODE = np.full(256, 4, dtype=np.uint8)
for _code, _symbols in enumerate((b"Aa", b"Cc", b"Gg", b"TtUu")):
    for _symbol in _symbols:
        STATE_CODE[_symbol] = _code


@dataclass(frozen=True)
class AlignmentSketch:
    names: tuple[str, ...]
    states: torch.Tensor
    alignment_length: int
    selected_sites: tuple[int, ...]


def stratified_site_indices(length: int, max_sites: int, seed: int) -> np.ndarray:
    """Choose at most ``max_sites`` sites across the full alignment span.

    One jittered position is chosen from each equal-width stratum.  The method
    therefore sees the whole coordinate range without allocating an ``n x L``
    alignment tensor.
    """

    if length <= 0 or max_sites <= 0:
        raise ValueError("length and max_sites must be positive")
    count = min(length, max_sites)
    if count == length:
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(seed)
    starts = np.floor(np.arange(count) * length / count).astype(np.int64)
    ends = np.floor((np.arange(count) + 1) * length / count).astype(np.int64)
    widths = np.maximum(ends - starts, 1)
    return starts + (rng.random(count) * widths).astype(np.int64)


def _next_nonempty(handle) -> str:
    for line in handle:
        if line.strip():
            return line.rstrip("\n\r")
    raise ValueError("Unexpected end of PHYLIP alignment")


def load_sequential_phylip_sketch(
    path: Path | str, *, max_sites: int, seed: int
) -> AlignmentSketch:
    """Load sampled columns from a sequential PHYLIP alignment.

    Only one full taxon sequence is resident while parsing.  Persistent memory
    is ``O(n * max_sites)`` even when the alignment is extremely long.
    """

    path = Path(path)
    with path.open("r") as handle:
        header = _next_nonempty(handle).split()
        if len(header) < 2:
            raise ValueError(f"Invalid PHYLIP header in {path}")
        n_taxa, alignment_length = int(header[0]), int(header[1])
        indices = stratified_site_indices(alignment_length, max_sites, seed)
        names: list[str] = []
        sampled: list[np.ndarray] = []
        for _ in range(n_taxa):
            first = _next_nonempty(handle)
            fields = first.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"Expected taxon name and sequence in {path}")
            name, sequence_chunk = fields
            pieces = ["".join(sequence_chunk.split())]
            observed = len(pieces[0])
            while observed < alignment_length:
                continuation = _next_nonempty(handle)
                compact = "".join(continuation.split())
                pieces.append(compact)
                observed += len(compact)
            sequence = "".join(pieces)
            if len(sequence) != alignment_length:
                raise ValueError(
                    f"Taxon {name!r} has {len(sequence)} sites; expected "
                    f"{alignment_length}"
                )
            encoded = STATE_CODE[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
            names.append(name)
            sampled.append(encoded[indices])
    if len(set(names)) != n_taxa:
        raise ValueError(f"Duplicate taxon names in {path}")
    states = torch.from_numpy(np.stack(sampled).astype(np.int64, copy=False))
    return AlignmentSketch(
        names=tuple(names),
        states=states,
        alignment_length=alignment_length,
        selected_sites=tuple(int(value) for value in indices),
    )


def phylip_shape(path: Path | str) -> tuple[int, int]:
    with Path(path).open("r") as handle:
        header = _next_nonempty(handle).split()
    if len(header) < 2:
        raise ValueError(f"Invalid PHYLIP header in {path}")
    return int(header[0]), int(header[1])
