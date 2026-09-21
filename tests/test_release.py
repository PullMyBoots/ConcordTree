from __future__ import annotations

from collections import OrderedDict
from itertools import combinations, permutations
import tempfile
from pathlib import Path
import random
import sys
import unittest

import numpy as np
from concordtree import __version__
from concordtree.cli import _parser
from concordtree.coverage import csi_evidence_band, inspect_alignment_coverage
from concordtree.assets import (
    ASSET_SHA256,
    QUARTET_PREDICTORS,
    load_backends,
    load_candidate_graph_backend,
    load_learned_nni_plan_backend,
    load_panel_score_backend,
    load_split_compat_backend,
    model_asset,
    verify_assets,
)
from concordtree.inference import (
    LOCAL_TEMPLATE,
    MAX_SITES,
    PostViewContext,
    QFExecutionTrace,
    SEEDS,
    _edge_splits,
    _panel_probabilities,
    _prepare_fork_shared_view_input,
    _postorder_graph_handoff,
    _run_refinement_stage,
    _saturating_pass,
    coordinate_stop_moves,
    infer,
    read_dimensions,
    resolve_candidate_distance_backend,
    resolve_missing_distance_model,
    resolve_nni_reduction_backend,
    resolve_scaffold_row_sum_backend,
    resolve_view_executor,
    resolve_view_workers,
    validate_refinement_control,
)
from concordtree._core.eapc_reachability import split_bitmasks
from concordtree._core.graphrank_laminar import (
    canonical_split,
    complete_compatible_splits,
)
from concordtree._core.learned_nni import (
    class_for_group_pairing,
    compile_edge_quartet_plan,
    remap_group_order_probabilities,
    tree_path_to_graph,
)
from concordtree._core.scaleqf import (
    TreeNode,
    graph_to_newick,
    load_phylip_sketch,
    tree_to_graph,
    validate_topology,
)
from concordtree._core.sctb_aggregate_nj import (
    aggregate_coverage_calibrated_row_sums,
    aggregate_marginalized_row_sums,
    aggregate_mismatch_row_sums,
    coverage_calibrated_profile_distance,
    finalize_mismatch_row_sums,
    imputed_site_frequencies,
    imputed_profile_slab,
    initial_imputed_profiles,
    marginalized_profile_distance,
    observed_profile_slab,
    site_mismatch_prior,
    site_observation_fraction,
)
from concordtree._core.sctb_contextual_patch import (
    ContextualPanel,
    _distinct_four_ranks,
    _indexed_displayed_quartet_classes,
    build_contextual_tree_index,
    build_contextual_panel_cover,
    build_contextual_panel_covers,
    directed_edge_nearest_representatives,
    compile_sparse_contextual_panel_plan,
    score_contextual_panel,
    score_contextual_panel_edges,
    score_sparse_contextual_panel_plan,
)
from concordtree._core.sctb_gpu_candidate import (
    _active_profile_slab,
    canonical_node_pair,
    exact_profile_pair_distances_gpu,
    projection_candidate_batch_gpu,
    projection_pool_batch_gpu,
)
from concordtree._core.sctb_reachability import ProjectionCandidateConfig
from concordtree._core.sctb_oracle_nni import directed_edge_leaf_masks
from concordtree._core.splitbank import (
    LaminarSplitSelector,
    greedy_ranked_compatible,
    split_set_to_tree,
    split_set_to_tree_indexed,
    splits_compatible,
)
from concordtree.models import MLP, QuartetMLP, QuartFormer
from concordtree.sparse_attention import (
    COMPILED_ATTENTION_NUM_WARPS,
    compiled_mask_attention_forward,
    compile_historical_pair_masks,
    compress_block_layout,
    define_historical_quartet_tail,
)
import torch


RELEASE_ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_csi_evidence_bands_keep_an_explicit_grey_zone(self) -> None:
        self.assertEqual(csi_evidence_band(0.25)[1], "standard")
        self.assertIsNone(csi_evidence_band(0.30)[1])
        self.assertEqual(csi_evidence_band(0.35)[1], "coverage-aware")

    def test_coverage_inspection_separates_complete_and_coherent_sparse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = root / "complete.phy"
            sparse = root / "sparse.phy"
            complete.write_text(
                "4 8\n"
                "a AACCGGTT\n"
                "b AATCGGTA\n"
                "c TACCGGTT\n"
                "d AACCGGTA\n"
            )
            sparse.write_text(
                "4 8\n"
                "a AACC----\n"
                "b AATC----\n"
                "c TACC----\n"
                "d AACC----\n"
            )
            complete_report = inspect_alignment_coverage(complete)
            sparse_report = inspect_alignment_coverage(sparse)
            self.assertEqual(complete_report["coherent_sparsity_index"], 0.0)
            self.assertEqual(
                complete_report["suggested_missing_data_model"], "standard"
            )
            self.assertAlmostEqual(
                sparse_report["coherent_sparsity_index"], 0.5
            )
            self.assertEqual(
                sparse_report["suggested_missing_data_model"], "coverage-aware"
            )

    def test_inspect_cli_contract(self) -> None:
        args = _parser().parse_args(
            ["inspect", "--msa", "input.phy", "--json"]
        )
        self.assertEqual(args.command, "inspect")
        self.assertEqual(args.sample_taxa, 128)
        self.assertEqual(args.sample_sites, 65_536)
        self.assertTrue(args.json)

    def test_dual_projection_pool_is_the_union_of_both_evidence_views(self) -> None:
        missing = np.uint8(255)
        states = np.asarray(
            [
                [0, 1, missing, 2, 3],
                [1, missing, 3, 2, 0],
                [0, 2, 3, missing, 1],
                [3, 2, 1, 0, missing],
                [missing, 1, 2, 3, 0],
                [2, 0, missing, 1, 3],
            ],
            dtype=np.uint8,
        )
        imputed_slab = imputed_profile_slab(states)
        observed_slab = observed_profile_slab(states)
        nodes = list(range(len(states)))
        imputed = {node: imputed_slab[node] for node in nodes}
        observed = {node: observed_slab[node] for node in nodes}
        config = ProjectionCandidateConfig(projections=4, window=1, candidate_cap=4)

        def pairs(bank: dict[int, np.ndarray] | tuple[dict[int, np.ndarray], ...]):
            pool = projection_pool_batch_gpu(
                nodes,
                imputed,
                config,
                device="cpu",
                distance_backend="eager",
                projection_profiles=bank,
            )
            return {tuple(map(int, pair)) for pair in pool.pairs}

        self.assertEqual(
            pairs((imputed, observed)), pairs(imputed) | pairs(observed)
        )

    def test_coverage_calibration_interpolates_without_a_threshold(self) -> None:
        missing = np.uint8(255)
        states = np.asarray(
            [[0, 1, missing, 2], [1, missing, 3, 2], [0, 2, 3, missing]],
            dtype=np.uint8,
        )
        slab = observed_profile_slab(states)
        prior = site_mismatch_prior(states)
        frequency = imputed_site_frequencies(states).astype(np.float64)
        weight = site_observation_fraction(states)
        profiles = {node: slab[node] for node in range(len(states))}
        offsets = {node: 0.0 for node in profiles}
        rows = aggregate_coverage_calibrated_row_sums(
            list(profiles), profiles, offsets, prior, frequency, weight
        )
        for left in profiles:
            expected = sum(
                coverage_calibrated_profile_distance(
                    profiles[left], profiles[right], prior, frequency, weight
                )
                for right in profiles
                if right != left
            )
            self.assertAlmostEqual(rows[left], expected, places=12)

        # Full coverage gives weight one and the ordinary mismatch limit.
        complete = np.asarray([[0, 1, 2], [1, 1, 3]], dtype=np.uint8)
        complete_slab = observed_profile_slab(complete)
        complete_prior = site_mismatch_prior(complete)
        complete_frequency = imputed_site_frequencies(complete).astype(np.float64)
        complete_weight = site_observation_fraction(complete)
        self.assertTrue(np.all(complete_weight == 1.0))
        self.assertAlmostEqual(
            coverage_calibrated_profile_distance(
                complete_slab[0], complete_slab[1], complete_prior,
                complete_frequency, complete_weight,
            ),
            float(np.mean(complete[0] != complete[1])),
            places=12,
        )

    def test_marginalized_missing_distance_has_clean_complete_limit(self) -> None:
        states = np.asarray(
            [[0, 1, 2, 3, 0], [0, 2, 2, 1, 0], [3, 1, 0, 3, 2]],
            dtype=np.uint8,
        )
        slab = observed_profile_slab(states)
        prior = site_mismatch_prior(states)
        for left, right in combinations(range(len(states)), 2):
            expected = float(np.mean(states[left] != states[right]))
            actual = marginalized_profile_distance(slab[left], slab[right], prior)
            self.assertAlmostEqual(actual, expected, places=12)

        nodes = list(range(len(states)))
        profiles = {node: slab[node] for node in nodes}
        offsets = {node: 0.0 for node in nodes}
        marginalized = aggregate_marginalized_row_sums(
            nodes, profiles, offsets, prior
        )
        ordinary = aggregate_mismatch_row_sums(
            nodes, profiles, offsets, validate_profiles=True
        )
        for node in nodes:
            self.assertAlmostEqual(marginalized[node], ordinary[node], places=12)

    def test_marginalized_missing_distance_uses_site_prior_for_absence(self) -> None:
        missing = np.uint8(255)
        states = np.asarray(
            [[0, 1, missing], [1, missing, missing], [0, 3, 2]],
            dtype=np.uint8,
        )
        slab = observed_profile_slab(states)
        prior = site_mismatch_prior(states)
        # Site 0 is observed in both profiles and mismatches; sites 1 and 2
        # retain their population prior because at least one side is absent.
        expected = float((1.0 + prior[1] + prior[2]) / 3.0)
        self.assertAlmostEqual(
            marginalized_profile_distance(slab[0], slab[1], prior),
            expected,
            places=12,
        )

    def test_release_graph_operators_are_bound(self) -> None:
        self.assertTrue(callable(_saturating_pass))

    def test_version(self) -> None:
        self.assertEqual(__version__, "0.1.2")


    def test_parallelism_maps_only_to_execution_resources(self) -> None:
        from concordtree.cli import _resolve_parallelism

        self.assertEqual(_resolve_parallelism(1, 0, 0, 256, 64), (1, 1, 1))
        self.assertEqual(_resolve_parallelism(8, 0, 0, 256, 64), (2, 4, 4))
        self.assertEqual(_resolve_parallelism(64, 0, 0, 256, 64), (2, 16, 8))
        self.assertEqual(_resolve_parallelism(32, 0, 0, 50000, 64), (1, 16, 8))
        self.assertEqual(_resolve_parallelism(8, 0, 0, 256, 64, 2), (2, 4, 4))
        self.assertEqual(_resolve_parallelism(8, 0, 0, 256, 64, 8), (2, 4, 4))
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            _resolve_parallelism(8, 2, 0, 256, 64)
        with self.assertRaisesRegex(ValueError, "exceeds CPU affinity"):
            _resolve_parallelism(65, 0, 0, 256, 64)

    def test_elastic_view_budget_preserves_default_and_accepts_eight(self) -> None:
        from concordtree.cli import _parser
        from concordtree.inference import SEEDS

        parser = _parser()
        common = ["infer", "--msa", "input.phy", "--work-dir", "run"]
        self.assertEqual(parser.parse_args(common).view_count, 4)
        self.assertEqual(
            parser.parse_args([*common, "--view-count", "8"]).view_count, 8
        )
        self.assertEqual(len(SEEDS), 8)
        self.assertEqual(
            SEEDS[:4],
            (
                (20260903, 20260934),
                (20261912, 20261947),
                (20262921, 20262960),
                (20263930, 20263973),
            ),
        )

    def test_six_refinement_controls_parse_numbers_and_none(self) -> None:
        parser = _parser()
        common = ["infer", "--msa", "input.phy", "--work-dir", "run"]
        defaults = parser.parse_args(common)
        self.assertEqual(defaults.view_stop_ratio, 0.01)
        self.assertEqual(defaults.view_max_rounds, 24)
        self.assertEqual(defaults.coordinate_stop_ratio, 0.005)
        self.assertEqual(defaults.coordinate_max_rounds, 4)
        self.assertEqual(defaults.saturation_stop_ratio, 0.005)
        self.assertEqual(defaults.saturation_max_rounds, 5)
        threshold_only = parser.parse_args(
            [
                *common,
                "--view-stop-ratio",
                "0.002",
                "--view-max-rounds",
                "none",
                "--coordinate-stop-ratio",
                "none",
                "--coordinate-max-rounds",
                "9",
                "--saturation-stop-ratio",
                "0",
                "--saturation-max-rounds",
                "none",
            ]
        )
        self.assertEqual(threshold_only.view_stop_ratio, 0.002)
        self.assertIsNone(threshold_only.view_max_rounds)
        self.assertIsNone(threshold_only.coordinate_stop_ratio)
        self.assertEqual(threshold_only.coordinate_max_rounds, 9)
        self.assertEqual(threshold_only.saturation_stop_ratio, 0.0)
        self.assertIsNone(threshold_only.saturation_max_rounds)

    def test_refinement_control_requires_at_least_one_condition(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a stop ratio"):
            validate_refinement_control("coordinate", None, None)
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_refinement_control("coordinate", 0.005, 0)
        with self.assertRaisesRegex(ValueError, "\[0, 1\]"):
            validate_refinement_control("coordinate", 1.1, 4)

    def test_refinement_stage_supports_threshold_budget_and_combined_modes(self) -> None:
        def run(moves: list[int], ratio, rounds):
            remaining = iter(moves)

            def operator(parent, context, n_taxa, work_dir, label, trace):
                return {
                    "moves": next(remaining),
                    "prediction": str(work_dir / label / "tree.nwk"),
                }, parent

            with tempfile.TemporaryDirectory() as directory:
                history, _graph, _path = _run_refinement_stage(
                    operator,
                    "test-stage",
                    {0: set()},
                    None,
                    103,
                    Path(directory),
                    ratio,
                    rounds,
                    False,
                )
            return history

        threshold_only = run([5, 1], 0.01, None)
        self.assertEqual(len(threshold_only), 2)
        self.assertEqual(threshold_only[-1]["stop_reasons"], ["threshold"])
        budget_only = run([5, 4], None, 2)
        self.assertEqual(len(budget_only), 2)
        self.assertEqual(budget_only[-1]["stop_reasons"], ["max_rounds"])
        combined = run([5, 1], 0.01, 2)
        self.assertEqual(
            combined[-1]["stop_reasons"], ["threshold", "max_rounds"]
        )
        fixed_point = run([0], None, 8)
        self.assertEqual(fixed_point[-1]["stop_reasons"], ["no_moves"])

    def test_rejects_unknown_quartet_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "run"
            with self.assertRaisesRegex(ValueError, "quartet_model"):
                infer(
                    RELEASE_ROOT / "examples/minimal24.phy",
                    work / "tree.nwk",
                    work,
                    quartet_model="unknown",
                )

    def test_explicit_missing_data_models_have_a_safe_default(self) -> None:
        parser = _parser()
        common = ["infer", "--msa", "input.phy", "--work-dir", "run"]
        standard = parser.parse_args(common)
        coverage = parser.parse_args(
            [*common, "--missing-data-model", "coverage-aware"]
        )
        self.assertEqual(standard.missing_data_model, "standard")
        self.assertEqual(coverage.missing_data_model, "coverage-aware")
        self.assertEqual(resolve_missing_distance_model("standard"), "imputed")
        self.assertEqual(
            resolve_missing_distance_model("coverage-aware"), "coverage"
        )
        with self.assertRaisesRegex(ValueError, "missing_data_model"):
            resolve_missing_distance_model("automatic")


    def test_candidate_distance_backend_is_scale_qualified(self) -> None:
        with unittest.mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("CONCORDTREE_CANDIDATE_DISTANCE_BACKEND", None)
            self.assertEqual(resolve_candidate_distance_backend(4095, "fast"), "eager")
            self.assertEqual(resolve_candidate_distance_backend(4096, "fast"), "native")
            self.assertEqual(resolve_candidate_distance_backend(4096, "transformer"), "eager")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_z_complete_state_first_round_quotient_matches_onehot(self) -> None:
        rng = np.random.default_rng(20260915)
        _sequence_backend, backend = load_backends()
        device = torch.device("cuda")
        for sites in (1, 31, 257, 701):
            states = rng.integers(0, 4, size=(37, sites), dtype=np.uint8)
            directions = rng.choice(
                np.asarray([-1, 1], dtype=np.int8),
                size=(sites, 4, 7),
            )
            state_tensor = torch.tensor(states, dtype=torch.uint8, device=device)
            direction_tensor = torch.tensor(
                directions, dtype=torch.int8, device=device
            )
            observed_projection = (
                backend.complete_state_projections_cuda(
                    state_tensor, direction_tensor
                )
                .cpu()
                .numpy()
            )
            onehot = np.eye(4, dtype=np.float32)[states]
            expected_projection = onehot.reshape(len(states), -1) @ (
                directions.reshape(sites * 4, 7).astype(np.float32)
            )
            np.testing.assert_array_equal(
                observed_projection, expected_projection
            )

            counts = np.ascontiguousarray(
                np.column_stack(
                    [np.count_nonzero(states == state, axis=0) for state in range(4)]
                ),
                dtype=np.int64,
            )
            observed_rows = (
                backend.complete_state_row_sums_cuda(
                    state_tensor,
                    torch.tensor(counts, dtype=torch.int64, device=device),
                    True,
                )
                .cpu()
                .numpy()
            )
            expected_rows = np.asarray(
                [
                    sum(
                        np.count_nonzero(states[row] != states[other]) / sites
                        for other in range(len(states))
                        if other != row
                    )
                    for row in range(len(states))
                ],
                dtype=np.float64,
            )
            np.testing.assert_allclose(observed_rows, expected_rows, atol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_z_compact_imputed_and_streamed_profile_statistics(self) -> None:
        rng = np.random.default_rng(20260919)
        _sequence_backend, backend = load_backends()
        device = torch.device("cuda")
        states = rng.integers(0, 6, size=(41, 193), dtype=np.uint8)
        frequency = np.zeros((states.shape[1], 4), dtype=np.float32)
        for base in range(4):
            frequency[:, base] = np.count_nonzero(states == base, axis=0)
        totals = frequency.sum(axis=1, keepdims=True)
        frequency = np.divide(
            frequency,
            totals,
            out=np.full_like(frequency, 0.25),
            where=totals > 0,
        )
        profiles = np.empty((*states.shape, 4), dtype=np.float32)
        profiles[:] = frequency
        for row in range(len(states)):
            valid = states[row] < 4
            profiles[row, valid] = 0.0
            profiles[row, np.nonzero(valid)[0], states[row, valid]] = 1.0
        pairs = np.column_stack(
            (
                rng.integers(0, len(states), 2048),
                rng.integers(0, len(states), 2048),
            )
        ).astype(np.int64)
        state_tensor = torch.tensor(states, dtype=torch.uint8, device=device)
        frequency_tensor = torch.tensor(
            frequency, dtype=torch.float32, device=device
        )
        profile_tensor = torch.tensor(
            profiles, dtype=torch.float32, device=device
        )
        pair_tensor = torch.tensor(pairs, dtype=torch.int64, device=device)
        expanded = backend.sparse_profile_distances_cuda(
            profile_tensor, pair_tensor
        )
        compact = backend.sparse_imputed_state_distances_cuda(
            state_tensor, frequency_tensor, pair_tensor
        )
        self.assertTrue(torch.equal(expanded, compact))

        streamed_results = []
        for chunk_sites in (17, 64):
            matches = torch.zeros(
                len(pairs), dtype=torch.float64, device=device
            )
            for start in range(0, states.shape[1], chunk_sites):
                chunk = torch.tensor(
                    np.ascontiguousarray(
                        profiles[:, start : start + chunk_sites, :]
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                backend.accumulate_sparse_profile_matches_cuda(
                    chunk, pair_tensor, matches
                )
            streamed_results.append(1.0 - matches / states.shape[1])
        torch.testing.assert_close(
            streamed_results[0], streamed_results[1], rtol=0.0, atol=1e-12
        )
        torch.testing.assert_close(
            streamed_results[0], expanded.to(torch.float64), rtol=0.0, atol=1e-6
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_z_streamed_coverage_pool_matches_resident_statistics(self) -> None:
        rng = np.random.default_rng(20260920)
        states = rng.integers(0, 6, size=(43, 197), dtype=np.uint8)
        slab = observed_profile_slab(states)
        slab.flags.writeable = False
        profiles = {node: slab[node] for node in range(len(states))}
        prior = site_mismatch_prior(states)
        frequency = imputed_site_frequencies(states).astype(np.float64)
        weight = site_observation_fraction(states)
        config = ProjectionCandidateConfig(
            projections=5, window=3, candidate_cap=7, seed=20260920
        )

        common = dict(
            active=list(profiles),
            profiles=profiles,
            config=config,
            device="cuda",
            distance_backend="native",
            row_sum_backend="native",
            validate_profiles=False,
            prestacked_profiles=slab,
            site_mismatch_prior=prior,
            site_frequency=frequency,
            site_imputation_weight=weight,
        )
        resident = projection_pool_batch_gpu(
            **common, profile_stream_sites=0
        )
        streamed = projection_pool_batch_gpu(
            **common, profile_stream_sites=17
        )
        np.testing.assert_array_equal(streamed.pairs, resident.pairs)
        np.testing.assert_allclose(
            streamed.distances, resident.distances, rtol=0.0, atol=2e-7
        )
        np.testing.assert_allclose(
            streamed.base_row_sums,
            resident.base_row_sums,
            rtol=0.0,
            atol=2e-11,
        )
        self.assertEqual(
            streamed.timing.distance_backend, "coverage-native-streamed64"
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_z_compiled_attention_warp_schedule_is_exact(self) -> None:
        import concordtree.sparse_attention as sparse_attention

        generator = torch.Generator(device="cuda").manual_seed(20260918)
        active_indices = torch.tensor(
            [[0, 1], [0, 1]], dtype=torch.int32, device="cuda"
        )
        active_counts = torch.tensor([2, 2], dtype=torch.int32, device="cuda")
        pair_masks = torch.full(
            (2, 2, 16), 65535, dtype=torch.int32, device="cuda"
        )
        original = sparse_attention.COMPILED_ATTENTION_NUM_WARPS
        try:
            for dtype in (torch.float32, torch.float16):
                q = torch.randn((1, 2, 32, 16), generator=generator, device="cuda", dtype=dtype)
                k = torch.randn_like(q)
                v = torch.randn_like(q)
                with torch.no_grad():
                    sparse_attention.COMPILED_ATTENTION_NUM_WARPS = 4
                    expected = compiled_mask_attention_forward(
                        q, k, v, active_indices, active_counts, pair_masks
                    )
                    sparse_attention.COMPILED_ATTENTION_NUM_WARPS = 1
                    observed = compiled_mask_attention_forward(
                        q, k, v, active_indices, active_counts, pair_masks
                    )
                self.assertTrue(torch.equal(observed, expected))
        finally:
            sparse_attention.COMPILED_ATTENTION_NUM_WARPS = original

    def test_interleaved_native_filter_matches_explicit_block_msas(self) -> None:
        rng = np.random.default_rng(20260918)
        sequence_backend, _pattern_backend = load_backends()
        alphabet = np.asarray(list("ACGT"))
        length = 257
        states = rng.integers(0, 4, size=(12, length), dtype=np.uint8)
        states[:, [1, 6, 11, 192]] = 0
        sequences = ["".join(alphabet[row]) for row in states]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            msa = root / "input.phy"
            msa.write_text(
                f"{len(sequences)} {length}\n"
                + "".join(
                    f"sample_{index} {sequence}\n"
                    for index, sequence in enumerate(sequences)
                )
            )
            raw, names, loaded_length = (
                sequence_backend.load_phy_to_packed_tensor(str(msa), False)
            )
            observed, observed_lengths = (
                sequence_backend.filter_interleaved_conserved_packed(
                    raw, loaded_length, 4, 4
                )
            )
            for block in range(4):
                block_sequences = [sequence[block::4] for sequence in sequences]
                block_path = root / f"block-{block}.phy"
                block_path.write_text(
                    f"{len(sequences)} {len(block_sequences[0])}\n"
                    + "".join(
                        f"sample_{index} {sequence}\n"
                        for index, sequence in enumerate(block_sequences)
                    )
                )
                expected, expected_names, expected_length = (
                    sequence_backend.load_phy_to_packed_tensor(
                        str(block_path), True
                    )
                )
                self.assertEqual(list(names), list(expected_names))
                self.assertEqual(int(observed_lengths[block]), expected_length)
                np.testing.assert_array_equal(observed[block], expected)

    def test_exact_native_view_reductions_are_scale_qualified(self) -> None:
        with unittest.mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("CONCORDTREE_SCAFFOLD_ROW_SUM_BACKEND", None)
            os.environ.pop("CONCORDTREE_NNI_REDUCTION_BACKEND", None)
            self.assertEqual(resolve_scaffold_row_sum_backend(4096, "fast"), "gpu32")
            self.assertEqual(resolve_scaffold_row_sum_backend(4096, "transformer"), "gpu64")
            self.assertEqual(resolve_nni_reduction_backend(4096, "fast"), "native")
            self.assertEqual(resolve_scaffold_row_sum_backend(4095, "fast"), "cpu")
            self.assertEqual(resolve_nni_reduction_backend(4096, "transformer"), "python")
            self.assertEqual(resolve_scaffold_row_sum_backend(32768, "fast"), "native")
            self.assertEqual(resolve_scaffold_row_sum_backend(32768, "transformer"), "native")

    def test_view_executor_is_scale_and_platform_qualified(self) -> None:
        with unittest.mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("CONCORDTREE_VIEW_EXECUTOR", None)
            self.assertEqual(resolve_view_executor(4096, "fast"), "thread")
            self.assertEqual(resolve_view_executor(4095, "fast"), "thread")
            self.assertEqual(resolve_view_executor(4096, "transformer"), "thread")
        with unittest.mock.patch.dict(
            "os.environ", {"CONCORDTREE_VIEW_EXECUTOR": "subprocess"}
        ):
            self.assertEqual(resolve_view_executor(4096, "fast"), "subprocess")
        with unittest.mock.patch.dict(
            "os.environ", {"CONCORDTREE_VIEW_EXECUTOR": "thread"}
        ):
            self.assertEqual(resolve_view_executor(4096, "fast"), "thread")

    def test_auto_view_workers_bound_large_single_gpu_memory(self) -> None:
        self.assertEqual(resolve_view_workers(4096, 0), 2)
        self.assertEqual(resolve_view_workers(20000, 0), 2)
        self.assertEqual(resolve_view_workers(50000, 0), 1)
        self.assertEqual(resolve_view_workers(50000, 2), 2)
        self.assertEqual(resolve_view_workers(4096, 0, 8), 2)
        self.assertEqual(resolve_view_workers(4096, 4, 2), 2)
        with self.assertRaisesRegex(ValueError, "between 0"):
            resolve_view_workers(50000, 5)

    def test_postorder_graph_handoff_matches_newick_roundtrip(self) -> None:
        names = [f"t{index}" for index in range(6)]
        graph = {
            0: {9},
            1: {9},
            2: {6},
            3: {8},
            4: {7},
            5: {7},
            6: {9, 2, 8},
            7: {8, 4, 5},
            8: {6, 3, 7},
            9: {0, 1, 6},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tree.nwk"
            path.write_text(graph_to_newick(graph, names, len(names)) + "\n")
            expected = tree_path_to_graph(path, names)
        self.assertEqual(_postorder_graph_handoff(graph, len(names)), expected)

    def test_native_candidate_graph_matches_numpy_with_ties(self) -> None:
        rng = np.random.default_rng(20260914)
        projected = rng.integers(-3, 4, size=(37, 5)).astype(np.float32)
        window = 4
        blocks = []
        for column in range(projected.shape[1]):
            order = np.argsort(projected[:, column], kind="stable")
            for offset in range(1, window + 1):
                left, right = order[:-offset], order[offset:]
                blocks.append(
                    np.column_stack(
                        (np.minimum(left, right), np.maximum(left, right))
                    )
                )
        expected_pairs = np.unique(np.concatenate(blocks), axis=0)
        backend = load_candidate_graph_backend()
        observed_pairs = np.asarray(
            backend.projection_pairs(projected, window, 4), dtype=np.int64
        )
        np.testing.assert_array_equal(observed_pairs, expected_pairs)

        distances = rng.integers(0, 9, size=len(expected_pairs)).astype(np.float64)
        tie_order = rng.permutation(len(projected))
        tie_rank = np.empty(len(projected), dtype=np.int64)
        tie_rank[tie_order] = np.arange(len(projected))
        owners = np.concatenate((expected_pairs[:, 0], expected_pairs[:, 1]))
        others = np.concatenate((expected_pairs[:, 1], expected_pairs[:, 0]))
        directed_distances = np.concatenate((distances, distances))
        order = np.lexsort((tie_rank[others], directed_distances, owners))
        ranked_owners = owners[order]
        _rows, starts, counts = np.unique(
            ranked_owners, return_index=True, return_counts=True
        )
        within = np.arange(len(order)) - np.repeat(starts, counts)
        selected = order[within < 7]
        expected_selected = np.column_stack((owners[selected], others[selected]))
        observed_selected = np.asarray(
            backend.select_directed_pairs(
                observed_pairs, distances, tie_rank, 7, 4
            ),
            dtype=np.int64,
        )
        np.testing.assert_array_equal(observed_selected, expected_selected)

    def test_native_ordered_nj_fold_matches_python_with_ties(self) -> None:
        rng = np.random.default_rng(1742026)
        backend = load_candidate_graph_backend()
        for n_taxa in (4, 11, 37, 65):
            all_pairs = np.asarray(
                list(combinations(range(n_taxa), 2)), dtype=np.int64
            )
            take = min(len(all_pairs), max(n_taxa, n_taxa * 12))
            pair_rows = np.sort(rng.choice(len(all_pairs), take, replace=False))
            pairs = all_pairs[pair_rows]
            distances = rng.integers(0, 13, size=take).astype(np.float64) / 8.0
            order = rng.permutation(n_taxa)
            tie_rank = np.empty(n_taxa, dtype=np.int64)
            tie_rank[order] = np.arange(n_taxa)
            row_sums = (
                rng.integers(-30, 31, size=n_taxa).astype(np.float64) / 4.0
            )
            offsets = (
                rng.integers(-12, 13, size=n_taxa).astype(np.float64) / 16.0
            )
            cap = min(7, n_taxa - 1)

            incident = {node: [] for node in range(n_taxa)}
            for pool_row, (left, right) in enumerate(pairs):
                incident[int(left)].append(
                    (distances[pool_row], tie_rank[right], int(right), pool_row)
                )
                incident[int(right)].append(
                    (distances[pool_row], tie_rank[left], int(left), pool_row)
                )
            kept = {
                pool_row
                for node in range(n_taxa)
                for *_rank, pool_row in sorted(incident[node])[:cap]
            }
            candidates = []
            for pool_row in kept:
                left, right = map(int, pairs[pool_row])
                if tie_rank[right] < tie_rank[left]:
                    left, right = right, left
                adjusted = (
                    float(distances[pool_row])
                    + float(offsets[left])
                    + float(offsets[right])
                )
                score = (
                    (n_taxa - 2) * adjusted
                    - float(row_sums[left])
                    - float(row_sums[right])
                )
                candidates.append((left, right, pool_row, score))
            candidates.sort(
                key=lambda row: (
                    tie_rank[row[0]], tie_rank[row[1]], row[0], row[1]
                )
            )
            best = {}
            for index, (left, right, _pool_row, score) in enumerate(candidates):
                for owner, other in ((left, right), (right, left)):
                    rank = (score, tie_rank[other], other)
                    if owner not in best or rank < best[owner][0]:
                        best[owner] = (rank, index)
            ranked = sorted(
                range(len(candidates)),
                key=lambda index: (
                    candidates[index][3],
                    tie_rank[candidates[index][0]],
                    tie_rank[candidates[index][1]],
                    candidates[index][0],
                    candidates[index][1],
                ),
            )
            consumed = set()
            expected = []
            for index in ranked:
                left, right, pool_row, _score = candidates[index]
                if len(expected) >= n_taxa - 3:
                    break
                if left in consumed or right in consumed:
                    continue
                if best[left][1] == index and best[right][1] == index:
                    expected.append((left, right, pool_row))
                    consumed.update((left, right))
            if not expected:
                left, right, pool_row, _score = candidates[ranked[0]]
                expected = [(left, right, pool_row)]

            observed, candidate_count, maximum_degree = backend.select_nj_merges(
                pairs,
                distances,
                tie_rank,
                row_sums,
                offsets,
                cap,
                n_taxa - 3,
                4,
            )
            np.testing.assert_array_equal(
                np.asarray(observed, dtype=np.int64),
                np.asarray(expected, dtype=np.int64),
            )
            self.assertEqual(int(candidate_count), len(candidates))
            self.assertEqual(
                int(maximum_degree),
                max(min(cap, len(rows)) for rows in incident.values()),
            )

    def test_cli_exposes_one_pipeline_with_two_model_backends(self) -> None:
        mlp = _parser().parse_args(
            ["infer", "--msa", "input.phy", "--work-dir", "run"]
        )
        transformer = _parser().parse_args(
            [
                "infer", "--msa", "input.phy", "--work-dir", "run-transformer",
                "--quartet-model", "transformer",
            ]
        )
        self.assertEqual(mlp.quartet_model, "mlp")
        self.assertEqual(mlp.missing_data_model, "standard")
        self.assertEqual(transformer.quartet_model, "transformer")
        self.assertEqual(mlp.view_workers, 0)
        self.assertEqual(mlp.view_count, 4)
        self.assertEqual(mlp.blas_threads, 0)
        self.assertEqual(mlp.quartet_predictor, "heterogeneous")
        gene_tree = _parser().parse_args(
            [
                "infer", "--msa", "input.phy", "--work-dir", "run-gene-tree",
                "--quartet-model", "transformer",
                "--tree-type", "gene-tree",
            ]
        )
        self.assertEqual(gene_tree.quartet_model, "transformer")
        self.assertEqual(gene_tree.quartet_predictor, "homogeneous")
        species_tree = _parser().parse_args(
            [
                "infer", "--msa", "input.phy", "--work-dir", "run-species-tree",
                "--tree-type", "species-tree",
            ]
        )
        self.assertEqual(species_tree.quartet_predictor, "heterogeneous")
        legacy_name = _parser().parse_args(
            [
                "infer", "--msa", "input.phy", "--work-dir", "run-legacy",
                "--family", "homogeneous",
            ]
        )
        self.assertEqual(legacy_name.quartet_predictor, "homogeneous")
        help_text = _parser()._subparsers._group_actions[0].choices["infer"].format_help()
        self.assertIn("--tree-type {gene-tree,species-tree}", help_text)
        self.assertIn(
            "--missing-data-model {standard,coverage-aware}", help_text
        )
        self.assertNotIn("--family", help_text)

    def test_assets_are_complete_and_immutable(self) -> None:
        self.assertEqual(verify_assets(), ASSET_SHA256)




    def test_quartet_predictor_checkpoint_schemas(self) -> None:
        self.assertEqual(QUARTET_PREDICTORS, ("heterogeneous", "homogeneous"))
        for predictor in QUARTET_PREDICTORS:
            mlp_state = torch.load(
                model_asset("mlp", predictor),
                map_location="cpu",
                weights_only=True,
            )
            mlp = QuartetMLP()
            mlp.load_state_dict(mlp_state)
            self.assertEqual(mlp.fc3.out_features, 3)

            qf_state = torch.load(
                model_asset("qf", predictor),
                map_location="cpu",
                weights_only=True,
            )
            qf = QuartFormer(species_num=24)
            qf.load_state_dict(qf_state)
            self.assertEqual(qf.classifier.out_features, 3)

    def test_native_nni_plan_matches_python(self) -> None:
        graph = {
            0: {8}, 1: {8}, 2: {9}, 3: {10},
            4: {11}, 5: {12}, 6: {13}, 7: {13},
            8: {0, 1, 9}, 9: {2, 8, 10}, 10: {3, 9, 11},
            11: {4, 10, 12}, 12: {5, 11, 13}, 13: {6, 7, 12},
        }
        expected = compile_edge_quartet_plan(graph, 8, representatives=4)
        observed = compile_edge_quartet_plan(
            graph,
            8,
            representatives=4,
            plan_backend=load_learned_nni_plan_backend(),
        )
        for field in (
            "edges", "branch_nodes", "offsets",
            "ordered_quartets", "canonical_quartets",
        ):
            np.testing.assert_array_equal(getattr(observed, field), getattr(expected, field))

    def test_native_nni_reduction_matches_vectorized_mapping(self) -> None:
        rng = np.random.default_rng(20260915)
        ordered = np.asarray(
            [rng.permutation(4) + 100 * row for row in range(37)],
            dtype=np.int64,
        )
        probabilities = rng.random((len(ordered), 3))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        offsets = np.asarray([0, 5, 13, 24, 37], dtype=np.int64)
        mapped = remap_group_order_probabilities(ordered, probabilities)
        expected = np.asarray(
            [mapped[offsets[i] : offsets[i + 1]].mean(axis=0) for i in range(4)]
        )
        observed = load_learned_nni_plan_backend().reduce_nni_arithmetic(
            ordered, probabilities, offsets
        )
        np.testing.assert_allclose(observed, expected, rtol=2e-15, atol=2e-15)

    def test_frozen_layer_shapes(self) -> None:
        self.assertEqual(MLP().fc1.in_features, 256)
        self.assertEqual(QuartetMLP().fc3.out_features, 3)
        model = QuartFormer(species_num=24)
        self.assertEqual(len(model.layers), 3)
        self.assertEqual(model.mlp_layer.fc1.in_features, 256)
        self.assertEqual(model.input_layer[0].out_features, 256)
        self.assertEqual(model.classifier.out_features, 3)

    def test_example_dimensions(self) -> None:
        path = RELEASE_ROOT / "examples/minimal24.phy"
        self.assertEqual(read_dimensions(path), (24, 128))
        rows = path.read_text().splitlines()[1:]
        self.assertEqual(len(rows), 24)
        for row in rows:
            name, sequence = row.split()
            self.assertTrue(name.startswith("Taxon"))
            self.assertEqual(len(sequence), 128)
            self.assertLessEqual(set(sequence), set("ACGT?-"))

    def test_rejects_too_few_taxa(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "small.phy"
            path.write_text("4 1\na A\nb C\nc G\nd T\n")
            with self.assertRaisesRegex(ValueError, "at least 24"):
                read_dimensions(path)

    def test_single_scale_normalized_stopping_rule(self) -> None:
        self.assertEqual(coordinate_stop_moves(32), 0)
        self.assertEqual(coordinate_stop_moves(256), 1)
        self.assertEqual(coordinate_stop_moves(4096), 20)

    def test_transformer_large_taxa_uses_compatible_float64_row_sum(self) -> None:
        self.assertEqual(resolve_scaffold_row_sum_backend(4096, "transformer"), "gpu64")
        self.assertEqual(resolve_scaffold_row_sum_backend(4096, "fast"), "gpu32")
        self.assertEqual(resolve_scaffold_row_sum_backend(32768, "transformer"), "native")

    def test_execution_trace_schema(self) -> None:
        self.assertEqual(
            QFExecutionTrace().as_dict(),
            {
                "batches": 0,
                "requested_panels": 0,
                "panels": 0,
                "cache_hits": 0,
                "quartets": 0,
                "quartet_index_seconds": 0.0,
                "pattern_cuda_ms": 0.0,
                "model_cuda_ms": 0.0,
                "output_to_host_seconds": 0.0,
                "plan_compile_seconds": 0.0,
                "probability_seconds": 0.0,
                "score_reduce_seconds": 0.0,
            },
        )

    def test_panel_probability_cache_coalesces_exact_ordered_taxa(self) -> None:
        class PatternBackend:
            calls = 0

            @classmethod
            def compute_pattern_frequencies_cuda_packed(
                cls, sequences, indices, effective_length
            ):
                cls.calls += 1
                return torch.zeros((len(indices), 256), dtype=torch.float32)

        model = MLP().eval()
        context = PostViewContext(
            inference_mode="fast",
            sequence_backend=None,
            pattern_backend=PatternBackend,
            model=model,
            coeff=None,
            active_block_indices=None,
            active_block_counts=None,
            attention_pair_masks=None,
            quartet_matrix=None,
            species=None,
            sequences=torch.zeros((24, 1), dtype=torch.uint8),
            names=tuple(f"t{index}" for index in range(24)),
            effective_length=1,
            view_tree_paths=(),
            view_graphs=(),
            view_splits=(),
            view_counts={},
            view_count=4,
            medoid_index=0,
            device=torch.device("cpu"),
            local_template_device=torch.from_numpy(LOCAL_TEMPLATE),
            probability_cache=OrderedDict(),
        )
        taxa = tuple(range(24))
        panels = [
            ContextualPanel(0, 0, taxa, ()),
            ContextualPanel(1, 1, taxa, ()),
        ]
        first = _panel_probabilities(panels, context)
        second = _panel_probabilities(panels, context)
        self.assertEqual(PatternBackend.calls, 1)
        self.assertEqual(len(context.probability_cache), 1)
        np.testing.assert_array_equal(first[0], first[1])
        np.testing.assert_array_equal(first[0], second[0])

    def test_vectorized_group_order_remap_matches_all_scalar_permutations(self) -> None:
        ordered = np.asarray(list(permutations((3, 11, 19, 27))), dtype=np.int64)
        probabilities = np.arange(len(ordered) * 3, dtype=np.float64).reshape(-1, 3)
        vectorized = remap_group_order_probabilities(ordered, probabilities)
        scalar = np.empty_like(vectorized)
        for row, quartet in enumerate(ordered):
            canonical = tuple(sorted(int(value) for value in quartet))
            for topology, pair in enumerate(((0, 1), (0, 2), (0, 3))):
                taxa = (int(quartet[pair[0]]), int(quartet[pair[1]]))
                scalar[row, topology] = probabilities[
                    row, class_for_group_pairing(canonical, taxa)
                ]
        np.testing.assert_array_equal(vectorized, scalar)

    def test_projection_batch_reuses_bit_exact_selected_distances(self) -> None:
        states = np.random.default_rng(20260914).integers(
            0, 4, size=(12, 37), dtype=np.uint8
        )
        profiles = initial_imputed_profiles(states)
        batch = projection_candidate_batch_gpu(
            range(len(states)),
            profiles,
            ProjectionCandidateConfig(
                projections=5,
                window=3,
                candidate_cap=4,
                seed=20260914,
            ),
            device="cpu",
            pair_batch_size=7,
        )
        selected = sorted(
            {
                tuple(sorted((left, right)))
                for left, neighbors in batch.candidates.items()
                for right in neighbors
                if left != right
            }
        )
        legacy = exact_profile_pair_distances_gpu(
            selected, profiles, device="cpu", pair_batch_size=7
        )
        self.assertTrue(selected)
        for pair in selected:
            self.assertEqual(
                batch.exact_distances[canonical_node_pair(*pair)], legacy[pair]
            )

    def test_complete_profile_encoding_is_exact_and_slab_backed(self) -> None:
        states = np.random.default_rng(174).integers(
            0, 4, size=(9, 31), dtype=np.uint8
        )
        profiles = initial_imputed_profiles(states)
        expected = np.eye(4, dtype=np.float32)[states]
        np.testing.assert_array_equal(np.stack(list(profiles.values())), expected)
        self.assertIs(profiles[0].base, profiles[1].base)

        missing = states.copy()
        missing[2, 7] = 4
        imputed = initial_imputed_profiles(missing)
        self.assertFalse(np.shares_memory(imputed[0], imputed[1]))
        self.assertAlmostEqual(float(imputed[2][7].sum()), 1.0)

    def test_immutable_complete_profile_slab_matches_local_construction(self) -> None:
        states = np.asarray(
            [[0, 1, 2, 3, 0], [3, 2, 1, 0, 3], [1, 1, 2, 2, 0]],
            dtype=np.uint8,
        )
        slab = np.eye(4, dtype=np.float32)[states]
        slab.flags.writeable = False
        historical = initial_imputed_profiles(states)
        shared = initial_imputed_profiles(
            states, immutable_complete_slab=slab
        )
        for taxon in range(len(states)):
            np.testing.assert_array_equal(shared[taxon], historical[taxon])
            self.assertFalse(shared[taxon].flags.writeable)
            self.assertTrue(np.shares_memory(shared[taxon], slab))

        writable = np.array(slab, copy=True)
        with self.assertRaisesRegex(ValueError, "read-only"):
            initial_imputed_profiles(
                states, immutable_complete_slab=writable
            )
        missing = states.copy()
        missing[0, 0] = 4
        missing_slab = imputed_profile_slab(missing)
        historical_missing = initial_imputed_profiles(missing)
        shared_missing = initial_imputed_profiles(
            missing, immutable_complete_slab=missing_slab
        )
        for taxon in range(len(missing)):
            np.testing.assert_array_equal(
                shared_missing[taxon], historical_missing[taxon]
            )
            self.assertTrue(
                np.shares_memory(shared_missing[taxon], missing_slab)
            )

    def test_gap_heavy_fork_shared_input_matches_historical_sketch(self) -> None:
        records = {
            "taxon_c": "A-CGNTTA",
            "taxon_a": "ATCG-TTA",
            "taxon_d": "GTCGNT-A",
            "taxon_b": "GT-G-TTA",
        }
        with tempfile.TemporaryDirectory() as temporary:
            msa = Path(temporary) / "missing.phy"
            msa.write_text(
                "4 8\n"
                + "".join(
                    f"{name} {sequence}\n" for name, sequence in records.items()
                )
            )
            shared = _prepare_fork_shared_view_input(
                msa.resolve(), "fixture-sha", 4, 8
            )
            self.assertIsNotNone(shared)
            assert shared is not None
            sketch = load_phylip_sketch(msa, max_sites=16_384, seed=174)
            order = sorted(range(sketch.n_taxa), key=lambda row: sketch.names[row])
            expected_states = np.ascontiguousarray(sketch.states[order])
            expected_names = tuple(sketch.names[row] for row in order)
            self.assertEqual(shared.scaffold_names, expected_names)
            np.testing.assert_array_equal(shared.scaffold_states, expected_states)
            expected_profiles = initial_imputed_profiles(expected_states)
            for taxon in range(len(expected_states)):
                np.testing.assert_array_equal(
                    shared.leaf_profile_slab[taxon], expected_profiles[taxon]
                )
            self.assertFalse(shared.scaffold_states.flags.writeable)
            self.assertFalse(shared.leaf_profile_slab.flags.writeable)

    def test_long_fork_shared_multiview_matches_historical_sketches(self) -> None:
        length = MAX_SITES + 17
        motifs = (
            "ACGTU-",
            "CGTUA-",
            "GTUAC-",
            "TUACG-",
        )
        with tempfile.TemporaryDirectory() as directory:
            msa = Path(directory) / "long.phy"
            rows = [
                (f"taxon{index}", (motif * (length // len(motif) + 1))[:length])
                for index, motif in enumerate(motifs)
            ]
            msa.write_text(
                f"{len(rows)} {length}\n"
                + "".join(f"{name} {sequence}\n" for name, sequence in rows)
            )
            shared = _prepare_fork_shared_view_input(
                msa.resolve(), "fixture-sha", len(rows), length, 2
            )
            self.assertIsNotNone(shared)
            assert shared is not None
            self.assertIsNone(shared.leaf_profile_slab)
            self.assertEqual(len(shared.view_scaffold_states), 2)
            for view, observed in enumerate(shared.view_scaffold_states):
                sketch = load_phylip_sketch(
                    msa,
                    max_sites=MAX_SITES,
                    seed=SEEDS[view][0],
                )
                order = sorted(
                    range(sketch.n_taxa),
                    key=lambda row: sketch.names[row],
                )
                self.assertEqual(
                    shared.scaffold_names,
                    tuple(sketch.names[row] for row in order),
                )
                np.testing.assert_array_equal(
                    observed,
                    np.ascontiguousarray(sketch.states[order]),
                )
                self.assertFalse(observed.flags.writeable)

    def test_first_round_profile_slab_is_the_historical_stack(self) -> None:
        states = np.random.default_rng(20260915).integers(
            0, 4, size=(11, 37), dtype=np.uint8
        )
        slab = np.eye(4, dtype=np.float32)[states]
        slab.flags.writeable = False
        profiles = initial_imputed_profiles(
            states, immutable_complete_slab=slab
        )
        nodes = list(range(len(states)))
        historical = _active_profile_slab(nodes, profiles)
        expected = np.stack([profiles[node] for node in nodes]).astype(
            np.float32, copy=False
        )
        shared = _active_profile_slab(nodes, profiles, slab)
        np.testing.assert_array_equal(historical, expected)
        np.testing.assert_array_equal(shared, historical)
        self.assertTrue(np.shares_memory(shared, slab))

        writable = np.array(slab, copy=True)
        with self.assertRaisesRegex(ValueError, "immutable"):
            _active_profile_slab(nodes, profiles, writable)
        unrelated = np.array(slab, copy=True)
        unrelated.flags.writeable = False
        with self.assertRaisesRegex(ValueError, "back the active leaf profiles"):
            _active_profile_slab(nodes, profiles, unrelated)
        with self.assertRaisesRegex(ValueError, "numeric leaf order"):
            _active_profile_slab(nodes[::-1], profiles, slab)

    def test_profile_pair_distance_batch_partition_is_bit_exact(self) -> None:
        states = np.random.default_rng(1024).integers(
            0, 4, size=(14, 41), dtype=np.uint8
        )
        profiles = initial_imputed_profiles(states)
        pairs = list(combinations(range(len(states)), 2))
        baseline = exact_profile_pair_distances_gpu(
            pairs, profiles, device="cpu", pair_batch_size=1
        )
        candidate = exact_profile_pair_distances_gpu(
            pairs, profiles, device="cpu", pair_batch_size=37
        )
        self.assertEqual(candidate, baseline)

    def test_unit_mass_validation_is_hoisted_only_inside_builder(self) -> None:
        states = np.random.default_rng(174).integers(
            0, 4, size=(8, 19), dtype=np.uint8
        )
        profiles = initial_imputed_profiles(states)
        nodes = list(profiles)
        offsets = {node: 0.0 for node in nodes}
        checked_rows = aggregate_mismatch_row_sums(nodes, profiles, offsets)
        unchecked_rows = aggregate_mismatch_row_sums(
            nodes, profiles, offsets, validate_profiles=False
        )
        self.assertEqual(checked_rows, unchecked_rows)

        config = ProjectionCandidateConfig(
            projections=3, window=2, candidate_cap=3, seed=174
        )
        checked_batch = projection_candidate_batch_gpu(
            nodes, profiles, config, device="cpu", pair_batch_size=5
        )
        unchecked_batch = projection_candidate_batch_gpu(
            nodes,
            profiles,
            config,
            device="cpu",
            pair_batch_size=5,
            validate_profiles=False,
        )
        self.assertEqual(checked_batch.candidates, unchecked_batch.candidates)
        self.assertEqual(checked_batch.exact_distances, unchecked_batch.exact_distances)

        malformed = {node: profile.copy() for node, profile in profiles.items()}
        malformed[0][0, 0] += 0.25
        with self.assertRaisesRegex(ValueError, "unit-mass|complete sampled sites"):
            aggregate_mismatch_row_sums(nodes, malformed, offsets)
        with self.assertRaisesRegex(ValueError, "unit-mass"):
            projection_candidate_batch_gpu(
                nodes, malformed, config, device="cpu", pair_batch_size=5
            )

    def test_reused_profile_slab_row_sums_match_reference(self) -> None:
        rng = np.random.default_rng(20260915)
        for missing in (False, True):
            states = rng.integers(0, 4, size=(17, 73), dtype=np.uint8)
            if missing:
                states[rng.random(states.shape) < 0.13] = 4
            profiles = initial_imputed_profiles(states)
            nodes = list(reversed(profiles))
            offsets = {node: float(rng.normal(scale=0.01)) for node in nodes}
            expected = aggregate_mismatch_row_sums(nodes, profiles, offsets)
            config = ProjectionCandidateConfig(
                projections=3, window=2, candidate_cap=3, seed=174
            )
            exact = projection_pool_batch_gpu(
                nodes,
                profiles,
                config,
                device="cpu",
                row_sum_backend="cpu-reuse",
            )
            observed = finalize_mismatch_row_sums(
                list(exact.nodes), exact.base_row_sums, offsets
            )
            self.assertEqual(observed, expected)

            reduced = projection_pool_batch_gpu(
                nodes,
                profiles,
                config,
                device="cpu",
                row_sum_backend="gpu32",
            )
            reduced_rows = finalize_mismatch_row_sums(
                list(reduced.nodes), reduced.base_row_sums, offsets
            )
            np.testing.assert_allclose(
                [reduced_rows[node] for node in nodes],
                [expected[node] for node in nodes],
                rtol=2e-6,
                atol=2e-3,
            )

            native_base = load_candidate_graph_backend().aggregate_mismatch_base_rows(
                np.stack([profiles[node] for node in sorted(nodes)]), 3
            )
            native_rows = finalize_mismatch_row_sums(
                sorted(nodes), native_base, offsets
            )
            np.testing.assert_allclose(
                [native_rows[node] for node in nodes],
                [expected[node] for node in nodes],
                rtol=2e-12,
                atol=2e-9,
            )

    def test_compressed_attention_layout_is_sorted_and_exact(self) -> None:
        dense = torch.tensor(
            [
                [False, True, False, True, False],
                [True, False, False, False, False],
            ],
            dtype=torch.bool,
        )
        indices, counts = compress_block_layout(dense)
        self.assertEqual(indices.shape, (2, 16))
        self.assertEqual(counts.tolist(), [2, 1])
        self.assertEqual(indices[0, :2].tolist(), [1, 3])
        self.assertEqual(indices[1, :1].tolist(), [0])
        rebuilt = torch.zeros_like(dense)
        for row, count in enumerate(counts.tolist()):
            rebuilt[row, indices[row, :count].long()] = True
        self.assertTrue(torch.equal(rebuilt, dense))

    def test_historical_pair_mask_compilation_is_exact_on_cpu(self) -> None:
        indices, counts = compress_block_layout(torch.ones((1, 1), dtype=torch.bool))
        quartets = torch.zeros((16, 48), dtype=torch.float32)
        for row in range(16):
            quartets[row, row % 12 : row % 12 + 4] = 1
        quartets = define_historical_quartet_tail(quartets)
        self.assertEqual(quartets.shape, (16, 48))
        self.assertEqual(quartets.stride(), (48, 1))
        self.assertGreater(
            quartets.untyped_storage().nbytes(), quartets.numel() * quartets.element_size()
        )

        compiled = compile_historical_pair_masks(indices, counts, quartets)
        storage = torch.cat([quartets.contiguous().view(-1), torch.zeros(64)])
        effective = torch.as_strided(storage, (16, 64), (48, 1))
        expected = (effective @ effective.T) >= 3
        weights = 1 << torch.arange(16, dtype=torch.int32)
        expected_rows = torch.sum(
            expected.to(torch.int32) * weights[None, :], dim=1, dtype=torch.int32
        )
        self.assertTrue(torch.equal(compiled[0, 0], expected_rows))
        self.assertEqual(int(torch.count_nonzero(compiled[0, 1:])), 0)

    def test_packed_greedy_compatibility_matches_scalar_order(self) -> None:
        n_taxa = 12
        total = (1 << n_taxa) - 1
        rng = torch.Generator().manual_seed(20260913)
        ranked: list[int] = []
        for _ in range(300):
            side = int(torch.randint(1, total, (), generator=rng).item())
            other = total ^ side
            if min(side.bit_count(), other.bit_count()) >= 2:
                ranked.append(min(side, other))
        ranked = list(dict.fromkeys(ranked))
        scalar: list[int] = []
        for split in ranked:
            if all(splits_compatible(split, old, total) for old in scalar):
                scalar.append(split)
                if len(scalar) == n_taxa - 3:
                    break
        self.assertEqual(
            greedy_ranked_compatible(ranked, n_taxa), frozenset(scalar)
        )

    def test_native_laminar_selector_matches_scalar_across_phases(self) -> None:
        backend = load_split_compat_backend()
        for n_taxa in (12, 65, 130):
            rng = random.Random(20260914 + n_taxa)
            ranked = [rng.getrandbits(n_taxa) for _ in range(600)]
            ranked.extend((ranked[3], ranked[11], 1, (1 << n_taxa) - 2))
            scalar = LaminarSplitSelector(n_taxa)
            native = LaminarSplitSelector(n_taxa, backend=backend)
            boundaries = (0, 137, 401, len(ranked))
            for left, right in zip(boundaries, boundaries[1:]):
                scalar.extend(ranked[left:right])
                native.extend(ranked[left:right])
                self.assertEqual(native.ordered_splits, scalar.ordered_splits)

    def test_indexed_tree_reconstruction_preserves_splits(self) -> None:
        n_taxa = 12
        # Nested and disjoint anchored clusters form one nontrivial compatible
        # partial split family; compare topology, not child/Newick order.
        splits = frozenset((0b000000001110, 0b000000011110, 0b000111100000))
        scalar = tree_to_graph(split_set_to_tree(splits, n_taxa), n_taxa)
        indexed = tree_to_graph(split_set_to_tree_indexed(splits, n_taxa), n_taxa)
        self.assertEqual(split_bitmasks(scalar, n_taxa), split_bitmasks(indexed, n_taxa))
        self.assertEqual(scalar, indexed)

    def test_native_tree_reconstruction_preserves_exact_order(self) -> None:
        n_taxa = 8
        splits = frozenset(
            {
                0b00000011,
                0b00000111,
                0b00001111,
                0b00110000,
                0b11000000,
            }
        )
        names = [f"t{index}" for index in range(n_taxa)]
        python_graph = tree_to_graph(split_set_to_tree_indexed(splits, n_taxa), n_taxa)
        native_graph = tree_to_graph(
            split_set_to_tree_indexed(
                splits,
                n_taxa,
                backend=load_split_compat_backend(),
            ),
            n_taxa,
        )
        self.assertEqual(native_graph, python_graph)
        self.assertEqual(
            graph_to_newick(native_graph, names, n_taxa),
            graph_to_newick(python_graph, names, n_taxa),
        )

    def test_laminar_completion_handles_deep_caterpillar_iteratively(self) -> None:
        n_taxa = 1_500
        splits = frozenset(
            split
            for width in range(2, n_taxa - 1)
            if (split := canonical_split((1 << width) - 1, n_taxa)) is not None
        )
        completed = complete_compatible_splits(splits, n_taxa)
        self.assertTrue(splits <= completed)
        self.assertEqual(len(completed), n_taxa - 3)
        graph = tree_to_graph(split_set_to_tree_indexed(completed, n_taxa), n_taxa)
        validate_topology(graph, n_taxa)
        rendered = graph_to_newick(
            graph, [f"t{index}" for index in range(n_taxa)], n_taxa
        )
        self.assertTrue(rendered.startswith("("))
        self.assertTrue(rendered.endswith(");"))

    def test_iterative_tree_conversion_preserves_recursive_observable_order(self) -> None:
        root = TreeNode(
            children=[
                TreeNode(children=[TreeNode(leaf=0), TreeNode(leaf=1)]),
                TreeNode(
                    children=[
                        TreeNode(leaf=2),
                        TreeNode(children=[TreeNode(leaf=3), TreeNode(leaf=4)]),
                    ]
                ),
            ]
        )
        graph = tree_to_graph(root, 5)
        self.assertEqual(
            graph,
            {
                0: {6}, 1: {6}, 2: {7}, 3: {8}, 4: {8},
                6: {0, 1, 7}, 7: {2, 6, 8}, 8: {3, 4, 7},
            },
        )
        self.assertEqual(graph_to_newick(graph, list("abcde"), 5), "(a,b,(c,(d,e)));" )

    def test_dense_panel_score_is_bit_exact_to_scalar_lookup(self) -> None:
        graph = {node: set() for node in range(14)}
        for left, right in (
            (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
            (0, 8), (1, 8), (2, 9), (3, 10), (4, 11), (5, 12),
            (6, 13), (7, 13),
        ):
            graph[left].add(right)
            graph[right].add(left)
        panel = ContextualPanel(0, 0, tuple(range(8)), ((10, 11),))
        template = np.asarray(list(combinations(range(8), 4)), dtype=np.int16)
        probabilities = np.random.default_rng(20260913).dirichlet(
            np.ones(3), size=len(template)
        )
        scalar_lookup = {
            tuple(int(value) for value in quartet): row
            for row, quartet in enumerate(template)
        }
        dense_lookup = np.full((8, 8, 8, 8), -1, dtype=np.int32)
        for row, quartet in enumerate(template):
            for order in permutations(int(value) for value in quartet):
                dense_lookup[order] = row
        tree_index = build_contextual_tree_index(graph, 8)
        scalar = score_contextual_panel(
            graph,
            panel,
            probabilities,
            template,
            tree_index=tree_index,
            row_lookup=scalar_lookup,
        )
        dense = score_contextual_panel(
            graph,
            panel,
            probabilities,
            template,
            tree_index=tree_index,
            row_lookup=dense_lookup,
        )
        self.assertEqual(dense.edges, scalar.edges)
        self.assertEqual(dense.base_score, scalar.base_score)
        self.assertEqual(dense.discriminating_rows, scalar.discriminating_rows)
        np.testing.assert_array_equal(dense.predicted_classes, scalar.predicted_classes)
        np.testing.assert_array_equal(dense.discriminating_mask, scalar.discriminating_mask)

    def test_native_paired_medians_are_bit_exact_to_numpy(self) -> None:
        backend = load_panel_score_backend()
        rng = np.random.default_rng(20260914)
        cases = [
            np.asarray([1.0], dtype=np.float64),
            np.asarray([3.0, 1.0], dtype=np.float64),
            np.asarray([2.0, 2.0, 2.0, 1.0], dtype=np.float64),
        ]
        cases.extend(
            rng.integers(-40, 41, size=length).astype(np.float64) / 7.0
            for length in range(1, 65)
        )
        for position, first in enumerate(cases):
            second = cases[-position - 1]
            expected = np.asarray(
                [np.median(first), np.median(second)], dtype=np.float64
            )
            actual = np.asarray(backend.median_pair(first, second), dtype=np.float64)
            np.testing.assert_array_equal(
                actual.view(np.uint64), expected.view(np.uint64)
            )

    def test_native_directed_nearest_messages_are_exact(self) -> None:
        backend = load_panel_score_backend()
        for n_taxa in (4, 8, 19, 37):
            graph = {node: set() for node in range(2 * n_taxa - 2)}
            active = list(range(n_taxa))
            rng = random.Random(20260915 + n_taxa)
            next_node = n_taxa
            while len(active) > 2:
                left = active.pop(rng.randrange(len(active)))
                right = active.pop(rng.randrange(len(active)))
                graph[next_node].update((left, right))
                graph[left].add(next_node)
                graph[right].add(next_node)
                active.append(next_node)
                next_node += 1
            graph[active[0]].add(active[1])
            graph[active[1]].add(active[0])
            for limit in (1, 4, 24):
                scalar = directed_edge_nearest_representatives(
                    graph, n_taxa, limit=limit
                )
                native = directed_edge_nearest_representatives(
                    graph, n_taxa, limit=limit, plan_backend=backend
                )
                self.assertEqual(native, scalar)

    def test_shared_panel_cover_workspace_preserves_order(self) -> None:
        graph = {node: set() for node in range(14)}
        for left, right in (
            (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
            (0, 8), (1, 8), (2, 9), (3, 10), (4, 11), (5, 12),
            (6, 13), (7, 13),
        ):
            graph[left].add(right)
            graph[right].add(left)
        index = build_contextual_tree_index(graph, 8)
        independent = [
            panel
            for cover in (0, 1)
            for panel in build_contextual_panel_cover(
                graph,
                8,
                cover=cover,
                panel_size=8,
                maximum_target_edges=3,
                required_taxa_cap=8,
                tree_index=index,
            )
        ]
        shared = build_contextual_panel_covers(
            graph,
            8,
            covers=(0, 1),
            panel_size=8,
            maximum_target_edges=3,
            required_taxa_cap=8,
            tree_index=index,
        )
        self.assertEqual(shared, independent)

    def test_native_contextual_panel_covers_are_exact(self) -> None:
        backend = load_panel_score_backend()
        for n_taxa in (24, 37, 64):
            graph = {node: set() for node in range(2 * n_taxa - 2)}
            active = list(range(n_taxa))
            rng = random.Random(174000 + n_taxa)
            next_node = n_taxa
            while len(active) > 2:
                left = active.pop(rng.randrange(len(active)))
                right = active.pop(rng.randrange(len(active)))
                graph[next_node].update((left, right))
                graph[left].add(next_node)
                graph[right].add(next_node)
                active.append(next_node)
                next_node += 1
            graph[active[0]].add(active[1])
            graph[active[1]].add(active[0])
            index = build_contextual_tree_index(
                graph, n_taxa, representative_limit=4, plan_backend=backend
            )
            for covers in ((0, 1), (0, 1, 2, 3)):
                scalar = build_contextual_panel_covers(
                    graph,
                    n_taxa,
                    covers=covers,
                    panel_size=24,
                    maximum_target_edges=12,
                    required_taxa_cap=20,
                    tree_index=index,
                )
                native = build_contextual_panel_covers(
                    graph,
                    n_taxa,
                    covers=covers,
                    panel_size=24,
                    maximum_target_edges=12,
                    required_taxa_cap=20,
                    tree_index=index,
                    plan_backend=backend,
                )
                self.assertEqual(native, scalar)

    def test_compact_panel_kernel_is_exact(self) -> None:
        expected = np.arange(4, dtype=np.int8)
        for order in permutations(range(4)):
            values = np.asarray([order], dtype=np.int16)
            ranks = _distinct_four_ranks(values)[0]
            np.testing.assert_array_equal(np.sort(ranks), expected)
            for position, value in enumerate(order):
                self.assertEqual(int(ranks[position]), value)

        graph = {node: set() for node in range(14)}
        for left, right in (
            (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
            (0, 8), (1, 8), (2, 9), (3, 10), (4, 11), (5, 12),
            (6, 13), (7, 13),
        ):
            graph[left].add(right)
            graph[right].add(left)
        panel = ContextualPanel(0, 0, tuple(range(8)), ((10, 11),))
        template = np.asarray(list(combinations(range(8), 4)), dtype=np.int16)
        probabilities = np.random.default_rng(20260914).dirichlet(
            np.ones(3), size=len(template)
        )
        lookup = np.full((8, 8, 8, 8), -1, dtype=np.int32)
        for row, quartet in enumerate(template):
            for order in permutations(int(value) for value in quartet):
                lookup[order] = row
        tree_index = build_contextual_tree_index(graph, 8)
        complete = score_contextual_panel(
            graph,
            panel,
            probabilities,
            template,
            tree_index=tree_index,
            row_lookup=lookup,
        )
        compact = score_contextual_panel_edges(
            graph,
            panel,
            probabilities,
            template,
            tree_index=tree_index,
            row_lookup=lookup,
        )
        self.assertEqual(compact, complete.edges)
        native = score_contextual_panel_edges(
            graph,
            panel,
            probabilities,
            template,
            tree_index=tree_index,
            row_lookup=lookup,
            plan_backend=load_panel_score_backend(),
        )
        self.assertEqual(native, complete.edges)
        positions = np.asarray(
            [tree_index.position[taxon] for taxon in panel.taxa], dtype=np.int32
        )
        native_classes = load_panel_score_backend().compile_panel_current_classes(
            positions, tree_index.depth, tree_index.ancestors, template
        )
        np.testing.assert_array_equal(
            native_classes,
            _indexed_displayed_quartet_classes(tree_index, panel, template),
        )

    def test_sparse_panel_score_removes_only_common_likelihood(self) -> None:
        graph = {node: set() for node in range(14)}
        for left, right in (
            (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
            (0, 8), (1, 8), (2, 9), (3, 10), (4, 11), (5, 12),
            (6, 13), (7, 13),
        ):
            graph[left].add(right)
            graph[right].add(left)
        panel = ContextualPanel(0, 0, tuple(range(8)), ((10, 11),))
        template = np.asarray(list(combinations(range(8), 4)), dtype=np.int16)
        probabilities = np.random.default_rng(17420260915).dirichlet(
            np.ones(3), size=len(template)
        ).astype(np.float32)
        lookup = np.full((8, 8, 8, 8), -1, dtype=np.int32)
        for row, quartet in enumerate(template):
            for order in permutations(int(value) for value in quartet):
                lookup[order] = row
        backend = load_panel_score_backend()
        tree_index = build_contextual_tree_index(graph, 8)
        dense = score_contextual_panel_edges(
            graph,
            panel,
            probabilities,
            template,
            tree_index=tree_index,
            row_lookup=lookup,
            plan_backend=backend,
        )
        plan = compile_sparse_contextual_panel_plan(
            graph,
            panel,
            template,
            tree_index=tree_index,
            row_lookup=lookup,
            plan_backend=backend,
        )
        sparse = score_sparse_contextual_panel_plan(
            plan, probabilities[plan.row_indices], plan_backend=backend
        )
        splitbank_sparse = score_sparse_contextual_panel_plan(
            plan,
            probabilities[plan.row_indices],
            plan_backend=backend,
            compute_medians=False,
        )
        self.assertLess(len(plan.row_indices), len(template))
        self.assertEqual(len(sparse), len(dense))
        for observed, expected in zip(sparse, dense):
            self.assertEqual(observed.edge, expected.edge)
            self.assertEqual(observed.branch_nodes, expected.branch_nodes)
            self.assertEqual(observed.best, expected.best)
            self.assertEqual(
                observed.alternative_median_gains,
                expected.alternative_median_gains,
            )
            relative = np.asarray(expected.scores) - expected.scores[0]
            np.testing.assert_allclose(observed.scores, relative, rtol=0, atol=1e-14)
            self.assertAlmostEqual(observed.gain, expected.gain, places=14)
        for observed, without_median in zip(sparse, splitbank_sparse):
            self.assertEqual(observed.edge, without_median.edge)
            self.assertEqual(observed.scores, without_median.scores)
            self.assertEqual(observed.best, without_median.best)
            self.assertEqual(observed.gain, without_median.gain)
            self.assertEqual(without_median.alternative_median_gains, (0.0, 0.0))

    def test_single_orientation_edge_splits_match_directed_messages(self) -> None:
        graph = {node: set() for node in range(14)}
        for left, right in (
            (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
            (0, 8), (1, 8), (2, 9), (3, 10), (4, 11), (5, 12),
            (6, 13), (7, 13),
        ):
            graph[left].add(right)
            graph[right].add(left)
        directed = directed_edge_leaf_masks(graph, 8)
        expected = {
            (left, right): canonical_split(directed[(left, right)], 8)
            for left, right in (
                (left, right)
                for left in graph
                for right in graph[left]
                if left < right and left >= 8 and right >= 8
            )
        }
        self.assertEqual(_edge_splits(graph, 8), expected)



if __name__ == "__main__":
    unittest.main()
