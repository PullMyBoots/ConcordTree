# V1.0.1 integration validation

Delivery: shared optimized AP-NJ / MLP-NNI, homogeneous checkpoint for gene trees and heterogeneous checkpoint for species trees, followed by the respective frozen posterior algorithms. All final checks below passed. This is implementation and execution regression on exposed development data, not a new accuracy or generalization study.

## Frontend execution

Hardware: physical GPU0 NVIDIA L40S, CPU16, NUMA0; serial workers. Each arm loads its checkpoint before timing, discards one warmup, then measures three repeats. Timing includes original input read/sample, AP-NJ and MLP-NNI; it excludes first model load, process startup, serialization/progress writing and posterior refinement. Reference is the pre-optimization gene v0.5.3 frontend; optimized is V1.0.1's shared v0.5.9-derived frontend. Both use exactly the same observations, seeds, sampling, missing-data mode, checkpoint and scientific stopping rules.

| Checkpoint | Taxa | Sites | Reference median (s) | Shared optimized median (s) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| homogeneous | 256 | 2048 | 0.077243 | 0.062137 | 1.243× |
| homogeneous | 1024 | 2048 | 0.315923 | 0.242184 | 1.304× |
| homogeneous | 4096 | 2048 | 1.902740 | 1.079599 | 1.762× |
| heterogeneous | 1024 | 2048 | 0.317952 | 0.263243 | 1.208× |
| heterogeneous | 4096 | 2048 | 1.940333 | 1.088684 | 1.782× |

For every arm and repeat, AP-NJ tree, final frontend tree, complete MLP action history and checkpoint SHA256 match exactly. All five cases use the original exposed homogeneous gene simulations (`alisim-homogeneous-v3/n{N}/000`); the heterogeneous-weight cases isolate implementation with species weights and are **not species accuracy tests**. One record per size/checkpoint does not establish a broad performance distribution or whole-pipeline latency. The compact species release contains only a posterior refiner, so this table is not a comparison against a nonexistent full constructor in that release.

Evidence: `BENCHMARK_RESULTS.json`, `BENCHMARK_PLAN.json`, `BENCHMARK_RAW.json`. The raw results include warmups and all measured repeats, not just selected best times.

## Independent posterior / public CLI regression

Six paired runs completed in separate processes. New runs use the public `run.py` CLI, default stop ratio `.005` and the fixed task-specific weights.

| Case | Input | Comparison | Outcome |
| --- | --- | --- | --- |
| Gene fixture | 32 taxa × 257 sites | Full v0.5.9 pipeline with the same homogeneous checkpoint | Exact prefix, final Newick including branch lengths, scientific model fit and search history |
| Exposed gene simulation | 256 taxa × 2048 sites | Same independent frozen gene reference | Same exact checks passed |
| Species fixture | 32 taxa × 257 sites, compressed FASTA | Frozen compact MSA-only posterior, supplied identical MLP tree | Exact final Newick and all scientific posterior history fields |
| Species fixture + loci | 32 taxa × 257 sites, loci128+129 | Frozen compact observed-loci posterior, identical MLP tree | Same exact checks passed |
| Exposed ILS species simulation | 32 taxa × 135900 sites | Frozen compact MSA-only posterior, identical MLP tree | Same exact checks passed |
| Same ILS simulation + observed loci | Original observed manifest | Frozen compact observed-loci posterior, identical MLP tree | Same exact checks passed |

The last two artifact IDs are `species-real32` / `species-real32-loci`; these are exposed **simulated ILS data**, not empirical biological data. Their original source paths and hashes are in the regression plan.

The old gene default was heterogeneous. The independent gene reference explicitly loads the requested homogeneous checkpoint in memory; it does not claim equality with the old heterogeneous scientific configuration. Its only posterior repair is the same narrow four-ULP parameter-boundary normalization, applied in memory without writing old source files. Species comparison starts from the same new frontend tree, because historical species frontend sampling/filtering is not claimed equivalent to this new shared frontend.

Timing and allocation high-water counters are excluded from scientific comparisons. The original strict comparison noticed two differing `resident_setup.peak_live_site_id_*` memory diagnostics (44 versus45 rows on gene256); all numerical model fields, fit trajectory, serialized lengths, topology and search history were identical. The failed diagnostic-only comparison is retained as `REGRESSION_ATTEMPT1.json` in the original run directory.

Evidence: `REGRESSION_RESULTS.json`, `REGRESSION_PLAN.json`, `INPUT_INTEGRITY.json`, plus the full result files indexed in `RUN_ARTIFACTS.json`.

## Contracts, input errors and mixed tasks

Seven CPU contract tests passed: immutable output/cache path rejection (including linked data targets and research memory), input preservation and FASTA conversion, malformed PHYLIP/FASTA rejection, exact checkpoint hashes and task routing, numerical-boundary acceptance/rejection, API task/option validation and fresh outputs, and foreign-runtime detection.

Additional GPU/API checks passed:

- Both checkpoint adapters reproduce independently counted quartet pattern frequencies over odd-length, missing-data inputs; maximum absolute error `3.944425022717191e-9` over16 query rows per adapter.
- A gene run with `JC69` (canonicalized to `JC`), two views, `.75` initial sampling, coverage-aware AP-NJ and one posterior round completes.
- The same engine then completes an observed-loci species request with its heterogeneous checkpoint.
- Too-small species input, constant-only gene input and an entirely missing gene taxon produce explicit failures, failed progress/metadata and no final prediction. Input hashes remain unchanged.

Evidence: `CONTRACT_TESTS.log`, `EDGE_CASES.json`. Initial smoke also covered both species input modes through the API.

## Fixes made during integration

1. The inherited gene optimizer can return `exp(log(100)) = 100.00000000000004`, which its strict parameter constructor rejected. V1.0.1 normalizes only values up to4 ULP outside the original alpha lower/upper or pinv upper boundaries. Meaningful out-of-range values and NaN/Inf remain rejected; regression tests cover both paths. The original source is untouched.
2. Species progress rows contain `elapsed_seconds`. The wrapper now stores this as `stage_elapsed_seconds` and reports a separate outer job elapsed time, avoiding a duplicate-key failure.
3. PHYLIP validation now rejects malformed complete rows before the inherited native parser can silently pad/truncate them. FASTA conversion closes its streaming handles and leaves the source untouched.
4. A validation assertion initially expected the alias `JC69` in output; the inherited parser correctly reports its canonical name `JC`. The assertion was corrected, without changing the algorithm.

All original failed attempts and logs remain under project `output/v1.0.1-validation/`. The final summary is based on successful subsequent checks, rather than treating those earlier attempts as successes.

## Integrity and limitations

All251 source/asset provenance entries still match their original hashes in the old release locations. Of the imported entries,168 are retained unchanged,78 unused historical files are excluded, and5 files have documented integration modifications. New entry points, serializer extraction and source provenance are recorded separately. No `data/` input or research-memory file was modified. Native source additions for planner/panel accompany their frozen binaries.

The runtime is independently copied into this release; it does not import code or weights from old releases. The existing CPython3.10 scientific environment is reused. Bundled native binaries have not been rebuilt or certified on a second host. Dataset accuracy, general MSC consistency, global search optimality, large-taxa full-pipeline throughput and web deployment are outside this integration validation. The homogeneous gene-weight switch requires its own accuracy evaluation before inheriting old accuracy claims.
