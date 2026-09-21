"""Command-line interface for ConcordTree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from concordtree import __version__

DEFAULT_VIEW_COUNT = 4
MAX_VIEW_COUNT = 8
MAX_CONCURRENT_VIEW_WORKERS = 4
DEFAULT_VIEW_STOP_RATIO = 0.01
DEFAULT_VIEW_MAX_ROUNDS = 24
DEFAULT_COORDINATE_STOP_RATIO = 0.005
DEFAULT_COORDINATE_MAX_ROUNDS = 4
DEFAULT_SATURATION_STOP_RATIO = 0.005
DEFAULT_SATURATION_MAX_ROUNDS = 5
TREE_TYPE_TO_PREDICTOR = {
    "gene-tree": "homogeneous",
    "species-tree": "heterogeneous",
}


class _TreeTypeAction(argparse.Action):
    """Translate the public scientific term to the checkpoint-family key."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, TREE_TYPE_TO_PREDICTOR[values])


def _optional_stop_ratio(value: str) -> float | None:
    """Parse a normalized move threshold or the literal ``none``."""

    if value.lower() == "none":
        return None
    try:
        ratio = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a ratio in [0, 1] or 'none'") from error
    if not 0.0 <= ratio <= 1.0:
        raise argparse.ArgumentTypeError("expected a ratio in [0, 1] or 'none'")
    return ratio


def _optional_max_rounds(value: str) -> int | None:
    """Parse a positive round budget or the literal ``none``."""

    if value.lower() == "none":
        return None
    try:
        rounds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'") from error
    if rounds < 1:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'")
    return rounds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="concordtree",
        description="Infer a complete phylogenetic tree from a sequential PHYLIP MSA.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="{infer,inspect,doctor,validate}"
    )

    infer_parser = subparsers.add_parser(
        "infer", help="infer a tree with ConcordTree"
    )
    infer_parser.add_argument("--msa", type=Path, required=True)
    infer_parser.add_argument("--work-dir", type=Path, required=True)
    infer_parser.add_argument(
        "--output",
        type=Path,
        default=Path("tree.nwk"),
        help="final Newick path within --work-dir (default: tree.nwk)",
    )
    inspect_parser = subparsers.add_parser(
        "inspect", help="measure missing-data structure without a reference tree"
    )
    inspect_parser.add_argument("--msa", type=Path, required=True)
    inspect_parser.add_argument(
        "--sample-taxa",
        type=int,
        default=128,
        help="maximum deterministic taxon sample (default: 128)",
    )
    inspect_parser.add_argument(
        "--sample-sites",
        type=int,
        default=65_536,
        help="maximum deterministic site sample (default: 65536)",
    )
    inspect_parser.add_argument(
        "--json", action="store_true", help="print machine-readable JSON"
    )
    infer_parser.add_argument("--device", default="cuda:0")
    infer_parser.add_argument(
        "--parallelism",
        type=int,
        default=0,
        help=(
            "total CPU thread budget (0: auto); affects runtime only, not tree "
            "decisions"
        ),
    )
    infer_parser.add_argument(
        "--view-workers",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    infer_parser.add_argument(
        "--view-count",
        type=int,
        choices=range(2, MAX_VIEW_COUNT + 1),
        default=DEFAULT_VIEW_COUNT,
        help=(
            "number of deterministic alignment views (2-8; default: 4); larger "
            "values use more computation and provide more tree evidence"
        ),
    )
    infer_parser.add_argument(
        "--blas-threads",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    infer_parser.add_argument(
        "--quartet-model",
        choices=("mlp", "transformer"),
        default="mlp",
        help=(
            "neural model for local four-taxon comparisons (default: mlp)"
        ),
    )
    infer_parser.add_argument(
        "--missing-data-model",
        choices=("standard", "coverage-aware"),
        default="standard",
        help=(
            "missing-data assumption used only for initial-tree distances: "
            "standard (default) for ordinary alignments; coverage-aware for "
            "sparse alignments with limited shared site coverage"
        ),
    )
    infer_parser.add_argument(
        "--tree-type",
        dest="quartet_predictor",
        choices=tuple(TREE_TYPE_TO_PREDICTOR),
        action=_TreeTypeAction,
        default="heterogeneous",
        help=(
            "gene-tree for one locus or one shared genealogy; species-tree "
            "(default) for concatenated loci that may have discordant gene "
            "histories; changes only the matched model weights"
        ),
    )
    infer_parser.add_argument(
        "--family",
        "--data-type",
        "--quartet-predictor",
        dest="quartet_predictor",
        choices=("heterogeneous", "homogeneous"),
        help=argparse.SUPPRESS,
    )
    infer_parser.add_argument(
        "--trace-performance",
        action="store_true",
        help="record detailed scorer host/CUDA timings (adds synchronization overhead)",
    )
    refinement = infer_parser.add_argument_group("refinement stopping controls")
    refinement.add_argument(
        "--view-stop-ratio",
        type=_optional_stop_ratio,
        default=DEFAULT_VIEW_STOP_RATIO,
        metavar="RATIO|none",
        help="stop each View after a pass changes at most this fraction of internal edges (default: 0.01)",
    )
    refinement.add_argument(
        "--view-max-rounds",
        type=_optional_max_rounds,
        default=DEFAULT_VIEW_MAX_ROUNDS,
        metavar="ROUNDS|none",
        help="maximum View NNI passes; none removes the normal round budget (default: 24)",
    )
    refinement.add_argument(
        "--coordinate-stop-ratio",
        type=_optional_stop_ratio,
        default=DEFAULT_COORDINATE_STOP_RATIO,
        metavar="RATIO|none",
        help="stop Coordinate refinement at this accepted-move fraction (default: 0.005)",
    )
    refinement.add_argument(
        "--coordinate-max-rounds",
        type=_optional_max_rounds,
        default=DEFAULT_COORDINATE_MAX_ROUNDS,
        metavar="ROUNDS|none",
        help="maximum Coordinate rounds; none makes the threshold authoritative (default: 4)",
    )
    refinement.add_argument(
        "--saturation-stop-ratio",
        type=_optional_stop_ratio,
        default=DEFAULT_SATURATION_STOP_RATIO,
        metavar="RATIO|none",
        help="stop Saturation refinement at this accepted-move fraction (default: 0.005)",
    )
    refinement.add_argument(
        "--saturation-max-rounds",
        type=_optional_max_rounds,
        default=DEFAULT_SATURATION_MAX_ROUNDS,
        metavar="ROUNDS|none",
        help="maximum Saturation rounds; none makes the threshold authoritative (default: 5)",
    )
    doctor_parser = subparsers.add_parser(
        "doctor", help="check Python, CUDA, binary backends, and asset hashes"
    )
    doctor_parser.add_argument("--device", default="cuda:0")
    doctor_parser.add_argument(
        "--skip-hashes", action="store_true", help="skip the packaged asset checksum pass"
    )

    validate_parser = subparsers.add_parser(
        "validate", help="compare two trees by their canonical split sets"
    )
    validate_parser.add_argument("--msa", type=Path, required=True)
    validate_parser.add_argument("--expected", type=Path, required=True)
    validate_parser.add_argument("--actual", type=Path, required=True)

    hidden = subparsers.add_parser("_view", help=argparse.SUPPRESS)
    subparsers._choices_actions = [
        action for action in subparsers._choices_actions if action.dest != "_view"
    ]
    hidden.add_argument("--msa", type=Path, required=True)
    hidden.add_argument("--target", type=Path, required=True)
    hidden.add_argument("--view", type=int, choices=range(MAX_VIEW_COUNT), required=True)
    hidden.add_argument("--device", required=True)
    hidden.add_argument(
        "--candidate-distance-backend",
        choices=("eager", "compiled", "native"),
        default="eager",
    )
    hidden.add_argument("--input-sha256")
    hidden.add_argument(
        "--row-sum-backend",
        choices=("cpu", "cpu-reuse", "native", "gpu32", "gpu64"),
    )
    hidden.add_argument(
        "--nni-reduction-backend", choices=("python", "native")
    )
    hidden.add_argument(
        "--quartet-predictor",
        choices=("heterogeneous", "homogeneous"),
        default="heterogeneous",
    )
    hidden.add_argument(
        "--missing-distance-model",
        choices=("imputed", "coverage"),
        default="imputed",
    )
    hidden.add_argument(
        "--view-stop-ratio",
        type=_optional_stop_ratio,
        default=DEFAULT_VIEW_STOP_RATIO,
    )
    hidden.add_argument(
        "--view-max-rounds",
        type=_optional_max_rounds,
        default=DEFAULT_VIEW_MAX_ROUNDS,
    )
    return parser


def _resolve_output(work_dir: Path, output: Path) -> Path:
    resolved_work = work_dir.resolve()
    return output.resolve() if output.is_absolute() else (resolved_work / output).resolve()


def _read_n_taxa(msa: Path) -> int:
    with msa.open("rt", encoding="utf-8") as stream:
        fields = stream.readline().split()
    if len(fields) < 2:
        raise ValueError("invalid sequential PHYLIP header")
    return int(fields[0])


def _resolve_parallelism(
    parallelism: int,
    view_workers: int,
    blas_threads: int,
    n_taxa: int,
    available_cpus: int,
    view_count: int = DEFAULT_VIEW_COUNT,
) -> tuple[int, int, int]:
    """Map one public CPU budget to deterministic execution resources."""

    if parallelism < 0:
        raise ValueError("parallelism must be nonnegative")
    if view_count < 2 or view_count > MAX_VIEW_COUNT:
        raise ValueError("view-count must be between 2 and 8")
    if parallelism == 0:
        concurrency = view_workers or min(2, view_count)
        return (
            view_workers,
            blas_threads or max(1, min(16, available_cpus // concurrency)),
            int(os.environ.get("CONCORDTREE_CANDIDATE_CPU_WORKERS", "8")),
        )
    if view_workers != 0 or blas_threads != 0:
        raise ValueError(
            "--parallelism cannot be combined with --view-workers or --blas-threads"
        )
    if parallelism > available_cpus:
        raise ValueError(
            f"parallelism {parallelism} exceeds CPU affinity budget {available_cpus}"
        )
    memory_safe_views = (
        1 if n_taxa >= 32768 else min(2, view_count)
    )
    workers = min(memory_safe_views, parallelism)
    threads_per_view = max(1, parallelism // workers)
    return workers, min(16, threads_per_view), min(8, threads_per_view)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "_view":
        from concordtree.inference import build_view

        build_view(
            args.msa.resolve(strict=True),
            args.target.resolve(),
            args.view,
            args.device,
            args.candidate_distance_backend,
            input_sha256=args.input_sha256,
            row_sum_backend=args.row_sum_backend,
            nni_reduction_backend=args.nni_reduction_backend,
            quartet_predictor=args.quartet_predictor,
            missing_distance_model=args.missing_distance_model,
            view_stop_ratio=args.view_stop_ratio,
            view_max_rounds=args.view_max_rounds,
        )
        return
    if args.command == "doctor":
        from concordtree.inference import doctor

        report = doctor(args.device, verify_hashes=not args.skip_hashes)
        print(json.dumps(report, indent=2, sort_keys=True))
        if report["status"] != "ok":
            raise SystemExit(2)
        return
    if args.command == "inspect":
        from concordtree.coverage import (
            format_coverage_report,
            inspect_alignment_coverage,
        )

        try:
            report = inspect_alignment_coverage(
                args.msa,
                sample_taxa=args.sample_taxa,
                sample_sites=args.sample_sites,
            )
        except (OSError, ValueError) as error:
            print(f"concordtree: {type(error).__name__}: {error}", file=sys.stderr)
            raise SystemExit(1) from error
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(format_coverage_report(report))
        return
    if args.command == "validate":
        from concordtree.validation import compare_topologies

        report = compare_topologies(args.msa, args.expected, args.actual)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["equivalent"]:
            raise SystemExit(1)
        return
    if args.command == "infer":
        if args.blas_threads < 0:
            raise SystemExit("--blas-threads must be nonnegative")
        available_cpus = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )
        try:
            view_workers, blas_threads, candidate_workers = _resolve_parallelism(
                args.parallelism,
                args.view_workers,
                args.blas_threads,
                _read_n_taxa(args.msa),
                available_cpus,
                args.view_count,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        # These libraries read their thread budgets while inference imports
        # NumPy/Torch.  Configure them before that import, not inside forked
        # workers after their global pools already exist.
        os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
        os.environ["MKL_NUM_THREADS"] = str(blas_threads)
        os.environ["CONCORDTREE_CANDIDATE_CPU_WORKERS"] = str(candidate_workers)
        os.environ["OMP_NUM_THREADS"] = str(candidate_workers)
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        from concordtree.inference import atomic_json, infer, utc_now

        work_dir = args.work_dir.resolve()
        output = _resolve_output(work_dir, args.output)
        try:
            row = infer(
                args.msa,
                output,
                work_dir,
                device_name=args.device,
                view_workers=view_workers,
                view_count=args.view_count,
                quartet_model=args.quartet_model,
                trace_performance=args.trace_performance,
                quartet_predictor=args.quartet_predictor,
                parallelism=args.parallelism,
                missing_data_model=args.missing_data_model,
                view_stop_ratio=args.view_stop_ratio,
                view_max_rounds=args.view_max_rounds,
                coordinate_stop_ratio=args.coordinate_stop_ratio,
                coordinate_max_rounds=args.coordinate_max_rounds,
                saturation_stop_ratio=args.saturation_stop_ratio,
                saturation_max_rounds=args.saturation_max_rounds,
            )
        except Exception as error:
            work_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(
                work_dir / "failure.json",
                {
                    "status": "failure",
                    "failed_at": utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"concordtree: {type(error).__name__}: {error}", file=sys.stderr)
            raise SystemExit(1) from error
        print(row["final_prediction"])


if __name__ == "__main__":
    main()
