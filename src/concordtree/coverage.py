"""Reference-free coverage diagnostics for choosing an initial-tree model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SAMPLE_TAXA = 128
DEFAULT_SAMPLE_SITES = 65_536
STANDARD_CSI_MAX = 0.25
COVERAGE_AWARE_CSI_MIN = 0.35

_POPCOUNT = np.asarray(
    [int(value).bit_count() for value in range(256)], dtype=np.uint8
)
_OBSERVED = np.zeros(256, dtype=np.bool_)
for _base in b"ACGTUacgtu":
    _OBSERVED[_base] = True


def _sample_positions(size: int, maximum: int) -> np.ndarray:
    """Return deterministic, approximately equally spaced positions."""

    if size <= 0:
        raise ValueError("sampled dimension must be positive")
    if maximum <= 0 or size <= maximum:
        return np.arange(size, dtype=np.int64)
    # Midpoints of equal-width strata avoid a random seed and cover the full
    # alignment. Since maximum < size, integer midpoints are unique.
    index = np.arange(maximum, dtype=np.int64)
    return ((2 * index + 1) * size // (2 * maximum)).astype(np.int64)


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def csi_evidence_band(csi: float) -> tuple[str, str | None, str]:
    """Map CSI to conservative scaffold-evidence guidance.

    The grey zone is intentional: the current evidence supports two regimes,
    not a validated final-tree automatic switch.
    """

    if not np.isfinite(csi) or csi < 0.0 or csi > 1.0:
        raise ValueError("CSI must be finite and between zero and one")
    if csi <= STANDARD_CSI_MAX:
        return (
            "standard-supported",
            "standard",
            "Coverage is not coherently sparse; standard is the supported initial-tree model.",
        )
    if csi >= COVERAGE_AWARE_CSI_MIN:
        return (
            "coverage-aware-supported",
            "coverage-aware",
            "Coverage is coherently sparse; coverage-aware is the supported initial-tree model.",
        )
    return (
        "indeterminate",
        None,
        "Current evidence does not separate the models in this CSI range; compare both when accuracy matters.",
    )


def inspect_alignment_coverage(
    msa: Path,
    *,
    sample_taxa: int = DEFAULT_SAMPLE_TAXA,
    sample_sites: int = DEFAULT_SAMPLE_SITES,
) -> dict[str, Any]:
    """Measure coherent sparsity from a sequential PHYLIP nucleotide MSA.

    Missing fraction is exact. Pairwise coverage statistics use deterministic
    stratified samples bounded by ``sample_taxa`` and ``sample_sites``.
    No topology, checkpoint, GPU, or reference tree is accessed.
    """

    if sample_taxa < 2:
        raise ValueError("sample-taxa must be at least 2")
    if sample_sites < 1:
        raise ValueError("sample-sites must be positive")
    path = Path(msa).resolve(strict=True)
    with path.open("rb") as stream:
        header = stream.readline().split()
        if len(header) < 2:
            raise ValueError("invalid sequential PHYLIP header; expected '<taxa> <sites>'")
        n_taxa, n_sites = map(int, header[:2])
        if n_taxa < 2 or n_sites < 1:
            raise ValueError("coverage inspection requires at least 2 taxa and 1 site")
        taxon_positions = _sample_positions(n_taxa, sample_taxa)
        site_positions = _sample_positions(n_sites, sample_sites)
        selected_taxa = set(map(int, taxon_positions))
        packed_masks: list[np.ndarray] = []
        observed_cells = 0
        row = 0
        for raw in stream:
            fields = raw.split()
            if len(fields) < 2:
                continue
            sequence = b"".join(fields[1:])
            if len(sequence) != n_sites:
                raise ValueError(
                    f"{path}: row {row} has {len(sequence)} sites, expected {n_sites}; "
                    "input must be sequential PHYLIP with one taxon per line"
                )
            encoded = np.frombuffer(sequence, dtype=np.uint8)
            observed = _OBSERVED[encoded]
            observed_cells += int(observed.sum())
            if row in selected_taxa:
                packed_masks.append(
                    np.packbits(observed[site_positions], bitorder="little")
                )
            row += 1
    if row != n_taxa:
        raise ValueError(f"{path}: read {row} taxa, expected {n_taxa}")
    if len(packed_masks) != len(taxon_positions):
        raise AssertionError("failed to collect every sampled taxon")

    sampled_length = len(site_positions)
    overlaps: list[float] = []
    jaccards: list[float] = []
    for left in range(len(packed_masks)):
        for right in range(left + 1, len(packed_masks)):
            intersection = int(
                _POPCOUNT[
                    np.bitwise_and(packed_masks[left], packed_masks[right])
                ].sum()
            )
            union = int(
                _POPCOUNT[
                    np.bitwise_or(packed_masks[left], packed_masks[right])
                ].sum()
            )
            overlaps.append(intersection / sampled_length)
            jaccards.append(intersection / union if union else 1.0)

    overlap = np.asarray(overlaps, dtype=np.float64)
    jaccard = np.asarray(jaccards, dtype=np.float64)
    median_overlap = _quantile(overlap, 0.5)
    median_jaccard = _quantile(jaccard, 0.5)
    csi = float((1.0 - median_overlap) * median_jaccard)
    band, recommendation, explanation = csi_evidence_band(csi)
    return {
        "msa": str(path),
        "taxa": n_taxa,
        "sites": n_sites,
        "missing_fraction": float(1.0 - observed_cells / (n_taxa * n_sites)),
        "sampled_taxa": len(packed_masks),
        "sampled_sites": sampled_length,
        "sampled_pairs": len(overlaps),
        "median_pairwise_shared_coverage": median_overlap,
        "median_coverage_mask_jaccard": median_jaccard,
        "coherent_sparsity_index": csi,
        "evidence_band": band,
        "suggested_missing_data_model": recommendation,
        "guidance": explanation,
        "guidance_scope": (
            "Initial-tree scaffold evidence only; inference keeps the model choice explicit."
        ),
    }


def format_coverage_report(report: dict[str, Any]) -> str:
    recommendation = report["suggested_missing_data_model"] or "compare both"
    return "\n".join(
        [
            f"Alignment: {report['msa']}",
            f"Taxa: {report['taxa']}",
            f"Sites: {report['sites']}",
            f"Missing cells: {100.0 * report['missing_fraction']:.2f}%",
            (
                "Median pairwise shared coverage: "
                f"{100.0 * report['median_pairwise_shared_coverage']:.2f}%"
            ),
            (
                "Median coverage-mask Jaccard: "
                f"{report['median_coverage_mask_jaccard']:.4f}"
            ),
            f"Coherent sparsity index (CSI): {report['coherent_sparsity_index']:.4f}",
            f"Evidence band: {report['evidence_band']}",
            f"Suggested --missing-data-model: {recommendation}",
            f"Guidance: {report['guidance']}",
            f"Scope: {report['guidance_scope']}",
        ]
    )
