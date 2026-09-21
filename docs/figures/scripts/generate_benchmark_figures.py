#!/usr/bin/env python3
"""Generate the static, paper-style figures used by the local result site."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve()
DOCS = HERE.parents[2]
RELEASE = DOCS.parent
PROJECT = RELEASE.parents[1]
ASSETS = DOCS / "assets"
STYLE = HERE.parents[1] / "styles/deepscientist-academic.mplstyle"

V012 = PROJECT / "artifacts/experiment/v012-unified-transformer-all-test-20260919/records.json"
SMALL_BASELINES = PROJECT / "artifacts/baseline/small-test-fasttree-caster-v1-20260907/records.json"
SMALL_DIPPER = PROJECT / "artifacts/baseline/dipper-gpu-small-test-forcebinary-v0.1.5-20260907/records"
LARGE_BASELINES = PROJECT / "artifacts/baseline/large-test-vft-rapidnj-fastme-n512-n1024-20260910/records.json"
LARGE_DIPPER = PROJECT / "artifacts/experiment/v174-vs-dipper-large-le1k-20260910/comparison/records.json"
GENE_CPU = PROJECT / "artifacts/experiment/cpu-baselines-alisim-homogeneous-v3-20260918/records.json"
GENE_DIPPER = PROJECT / "artifacts/baseline/dipper-gpu-alisim-homogeneous-v3-full200-20260918/records.json"
CORRECTED_DIPPER_TIME = PROJECT / "artifacts/benchmark/dipper-gpu-site-correction-v1-20260919/timing/records.json"
CONTINUATION_ROOT = PROJECT / "artifacts/experiment/v012-test-convergence-dipper-gpu0-20260920/continuations"

COLORS = {
    "ConcordTree attention": "#173F70",
    "ConcordTree lightweight": "#7891AC",
    "DIPPER GPU": "#A34B43",
    "VeryFastTree": "#58676A",
    "FastTree": "#58676A",
    "CASTER-site": "#7E878A",
    "RapidNJ": "#8B7851",
    "FastME": "#7B657D",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def load_continuation_overlay() -> dict[str, dict]:
    overlay: dict[str, dict] = {}
    for path in CONTINUATION_ROOT.rglob("metrics.json"):
        record_id = "/".join(path.relative_to(CONTINUATION_ROOT).parts[:-1])
        overlay[record_id] = load_json(path)
    return overlay


CONTINUATIONS = load_continuation_overlay()


def released_value(row: dict, field: str) -> float:
    continuation = CONTINUATIONS.get(row["record_id"])
    if continuation is not None and field == "mlp_nrf":
        return float(continuation["final_nrf"])
    if continuation is not None and field == "mlp_internal_wall_seconds":
        return float(continuation["source_e2e_seconds"]) + float(
            continuation["added_saturation_seconds"]
        )
    return float(row[field])


def bootstrap_mean(values: Iterable[float], *, seed: int) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        raise ValueError("empty sample")
    generator = np.random.default_rng(seed)
    draws = generator.choice(array, size=(4000, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(ASSETS / f"{stem}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def mean_ci_bars(ax, rows, title: str, panel: str, *, ymax: float) -> None:
    labels = [row[0] for row in rows]
    stats = [bootstrap_mean(row[1], seed=1000 + index) for index, row in enumerate(rows)]
    means = np.array([item[0] for item in stats])
    lower = means - np.array([item[1] for item in stats])
    upper = np.array([item[2] for item in stats]) - means
    positions = np.arange(len(rows))
    colors = [COLORS[label] for label in labels]
    hatches = ["///" if label.startswith("ConcordTree") else "" for label in labels]
    bars = ax.barh(
        positions,
        means,
        xerr=np.vstack([lower, upper]),
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        height=0.66,
        capsize=2.2,
        error_kw={"elinewidth": 0.8, "ecolor": "#2D3338", "capthick": 0.8},
        zorder=3,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for y, mean, high in zip(positions, means, upper):
        ax.text(mean + high + ymax * 0.025, y, f"{mean:.3f}", ha="left", va="center", fontsize=7.3)
    ax.set_yticks(positions)
    ax.set_yticklabels([label.replace("ConcordTree ", "CT ") for label in labels])
    ax.invert_yaxis()
    ax.set_xlabel("Mean normalized RF distance")
    ax.set_xlim(0, ymax)
    ax.set_title(f"({panel})  {title}", loc="left", pad=7)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)


def accuracy_samples():
    release_rows = load_json(V012)

    def released(collection: str, field: str, taxa: set[int] | None = None):
        return [
            released_value(row, field)
            for row in release_rows
            if row["collection"] == collection and (taxa is None or row["n_taxa"] in taxa)
        ]

    small_baselines = load_json(SMALL_BASELINES)
    small_dipper_rows = [load_json(path) for path in SMALL_DIPPER.rglob("metrics.json")]
    small_dipper = [row["normalized_rf"] for row in small_dipper_rows if row.get("status") == "success"]
    small = [
        ("ConcordTree attention", released("small", "transformer_nrf")),
        ("ConcordTree lightweight", released("small", "mlp_nrf")),
        ("FastTree", [row["normalized_rf"] for row in small_baselines if row["method"] == "FastTree" and row["status"] == "success"]),
        ("CASTER-site", [row["normalized_rf"] for row in small_baselines if row["method"] == "CASTER-site" and row["status"] == "success"]),
        ("DIPPER GPU", small_dipper),
    ]

    large_baselines = load_json(LARGE_BASELINES)
    large_dipper = load_json(LARGE_DIPPER)
    large = [
        ("ConcordTree attention", released("large", "transformer_nrf", {512, 1024})),
        ("ConcordTree lightweight", released("large", "mlp_nrf", {512, 1024})),
        ("VeryFastTree", [row["normalized_rf"] for row in large_baselines if row["method"] == "veryfasttree" and row["status"] == "success"]),
        ("RapidNJ", [row["normalized_rf"] for row in large_baselines if row["method"] == "rapidnj" and row["status"] == "success"]),
        ("FastME", [row["normalized_rf"] for row in large_baselines if row["method"] == "fastme" and row["status"] == "success"]),
        ("DIPPER GPU", [row["dipper_rf"] for row in large_dipper]),
    ]

    gene_cpu = load_json(GENE_CPU)
    gene_dipper = load_json(GENE_DIPPER)
    gene = [
        ("ConcordTree attention", released("alisim-homogeneous-v3", "transformer_nrf")),
        ("ConcordTree lightweight", released("alisim-homogeneous-v3", "mlp_nrf")),
        ("VeryFastTree", [row["normalized_rf"] for row in gene_cpu if row["method"] == "veryfasttree" and row["status"] == "success"]),
        ("FastME", [row["normalized_rf"] for row in gene_cpu if row["method"] == "fastme" and row["status"] == "success"]),
        ("RapidNJ", [row["normalized_rf"] for row in gene_cpu if row["method"] == "rapidnj" and row["status"] == "success"]),
        ("DIPPER GPU", [row["normalized_rf"] for row in gene_dipper if row["status"] == "predicted"]),
    ]
    return small, large, gene


def make_accuracy_figures() -> None:
    small, large, gene = accuracy_samples()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), gridspec_kw={"width_ratios": [1, 1.08]})
    mean_ci_bars(axes[0], small, "Species trees: SimPhy Small", "a", ymax=0.225)
    mean_ci_bars(axes[1], large, "Species trees: SimPhy Large", "b", ymax=0.235)
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.17, top=0.91, wspace=0.42)
    save(fig, "species-tree-accuracy")

    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    mean_ci_bars(ax, gene, "Gene trees: AliSim single-history panel", "a", ymax=0.37)
    fig.subplots_adjust(left=0.205, right=0.985, bottom=0.17, top=0.91)
    save(fig, "gene-tree-accuracy")


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def make_runtime_figure() -> None:
    released = load_json(V012)
    cpu = load_json(GENE_CPU)
    dipper = load_json(CORRECTED_DIPPER_TIME)
    scales = [256, 512, 1024, 2048, 4096]

    def record_index(identifier: str) -> int:
        return int(identifier.rsplit("/", 1)[-1])

    def ct_values(scale: int, field: str) -> list[float]:
        return [
            released_value(row, field)
            for row in released
            if row["collection"] == "alisim-homogeneous-v3"
            and row["n_taxa"] == scale
            and record_index(row["record_id"]) < 10
        ]

    def cpu_values(scale: int, method: str) -> list[float]:
        return [
            row["runtime_seconds"]
            for row in cpu
            if row["method"] == method and row["scale"] == scale and row["index"] < 10 and row["status"] == "success"
        ]

    def dipper_values(scale: int) -> list[float]:
        return [row["wall_seconds"] for row in dipper if row["taxa"] == scale]

    series = {
        "ConcordTree lightweight": [ct_values(scale, "mlp_internal_wall_seconds") for scale in scales],
        "ConcordTree attention": [ct_values(scale, "transformer_internal_wall_seconds") for scale in scales],
        "DIPPER GPU": [dipper_values(scale) for scale in scales],
        "VeryFastTree": [cpu_values(scale, "veryfasttree") for scale in scales],
        "RapidNJ": [cpu_values(scale, "rapidnj") for scale in scales],
        "FastME": [cpu_values(scale, "fastme") for scale in scales],
    }

    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    x = np.arange(len(scales))
    order = ["ConcordTree lightweight", "ConcordTree attention", "DIPPER GPU", "VeryFastTree", "RapidNJ", "FastME"]
    markers = ["o", "s", "D", "^", "v", "P"]
    dashes = [None, None, (3, 2), None, None, None]
    for label, marker, dash in zip(order, markers, dashes):
        groups = series[label]
        medians = np.array([np.median(values) for values in groups], dtype=float)
        q1 = np.array([percentile(values, 0.25) for values in groups])
        q3 = np.array([percentile(values, 0.75) for values in groups])
        line, = ax.plot(
            x,
            medians,
            color=COLORS[label],
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=label,
            zorder=4,
        )
        if dash:
            line.set_dashes(dash)
        ax.fill_between(x, q1, q3, color=COLORS[label], alpha=0.09, linewidth=0, zorder=1)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{scale:,}" for scale in scales])
    ax.set_xlabel("Number of taxa")
    ax.set_ylabel("Inference time per alignment (s, log scale)")
    ax.set_title("Runtime by number of taxa", loc="left", pad=8)
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.0), columnspacing=1.3, labelspacing=0.4)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", which="major", visible=True)
    ax.text(0.99, 0.02, "line: median; band: interquartile range", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color="#5C6268")
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.17, top=0.92)
    save(fig, "runtime-scaling")


def main() -> int:
    plt.style.use(STYLE)
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_accuracy_figures()
    make_runtime_figure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
