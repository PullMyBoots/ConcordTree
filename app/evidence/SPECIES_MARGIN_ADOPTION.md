# Species support: user-authorized AUC-prioritized adoption

On 2026-11-05 the user explicitly chose the higher-AUC support strategy after reviewing the fixed-coverage error tradeoff. V1.0.1 now defaults to the already frozen development winner, **mean_worst_margin**, with public method ID `mean_worst_quartet_margin`. Gene support, inference/scoring models and tree search are unchanged. This adoption changes support ranking, not tree accuracy.

## Decision and evidence

The experiment selected exactly one of four predeclared margin candidates using 40 development species trees, before evaluating 40 record/seed-disjoint confirmation trees. We retain that winner; we do not reselect a different candidate using confirmation results.

| Panel | Legacy agreement AUC | Selected margin AUC |
| --- | ---: | ---: |
| Development,40 trees /48,008 branches | 0.8967301284 | 0.9204170750 |
| Confirmation,40 trees /48,008 branches | 0.9170828469 | 0.9228206745 |

Confirmation paired stratified tree-bootstrap 95% interval for AUC difference: **[-0.0012807421,0.0146396468]**. It crosses zero. At the same 73.4544% retained coverage (35,264 branches), incorrect retained branches increase 355→446. The original joint confirmation gate **failed**. The historical report and gate results are preserved in `SPECIES_MARGIN_EXPERIMENT_REPORT.md`, `SPECIES_MARGIN_EXPERIMENT_CONTRACT.json` and `SPECIES_MARGIN_EXPERIMENT_CONFIRMATION.json`; this user-directed AUC-priority decision does not rewrite them as passed.

This is same-pool simulation evidence with disjoint records/seeds, not a fresh cross-generator project holdout or a calibrated probability claim. It does not demonstrate better performance at every cutoff or on every dataset.

## Production definition and compatibility

For each final edge, reuse the same informative cached unique quartets spanning its four incident subtrees. Raw score:

`m = mean_q[current(q) - max(alternative 1(q), alternative 2(q))]`

Fully flat scores are excluded using exactly the legacy floating-point tolerance. Two-way ties remain informative. The sidecar records `mean_worst_margin`; this raw field is authoritative for ROC-AUC. For existing numeric tree labels, `value` is a fixed monotone display index `50 + 50*m/(1+abs(m))`, using unit scale 1 without fitting to correctness labels. This is not a percentage or a probability, and no legacy 95/99 threshold carries over. The old 0–100 agreement is retained as `sampled_quartet_agreement`, with counts and three-way vote fractions.

Species Newick labels use 17 significant digits, preserving their float64 display values. All 96,016 validation branches retain exactly the raw rank ordering after display conversion and decimal roundtrip. At extreme finite-precision saturation, raw scores remain the appropriate ranking field. Missing evidence stays null and receives no numeric label. Branches are never removed by support scoring.

The new native `reduce_margins` adds one signed-margin sum per edge in the existing O(Q+N) aggregation pass, with O(N) state. The legacy `reduce` entry remains compatible. No extra DNA scoring, quartet model predictions, fits, inference rounds or bootstrap trees are introduced.

## Integration verification

- 16 unit tests passed, including an independent scalar raw-margin reference, topology orientations, ties, missing evidence, annotation precision and 10,000-tip skew traversal.
- All 80 saved prototype panels: raw production display inputs match saved selected-candidate moments; AUC and ranks remain exact after display/serialization (`SPECIES_MARGIN_PANEL_EQUIVALENCE.json`). This recomputes summaries, not 80 inference runs.
- Actual 4096-taxon complex confirmation case rerun: all 4,093 raw branch margins exactly match both the independent prototype kernel applied to live cached scores and the saved experiment features. Unannotated tree is identical (`SPECIES_MARGIN_END_TO_END.json`).
- Four paired support-enabled/disabled runs cover gene default, species default, species early stopping/remapping and observed loci. Trees, search histories and fit/quartet-prediction counts match; gene annotated output and support values also match the previous release (`SPECIES_MARGIN_REGRESSION.json`). `branch_support=False` API behavior remains available.
- Exhaustive 70 quartets of an 8-tip tree: independent CPU four-component traversal agrees on raw margin and old vote diagnostics (`SPECIES_MARGIN_QUARTET_REFERENCE.json`).
- 251 original source hashes verified; all 27 pre-existing inference binaries unchanged. Only the CPU support reducer was explicitly rebuilt (`SPECIES_MARGIN_SOURCE_AUDIT.json`). Input hashes remained unchanged.

## Incremental runtime

GPU 0/L40S, NUMA 0,16 CPU threads, one process with predictors preloaded;4096 taxa×2048 sites. One warmup then three timed runs per arm, alternating order. Both arms calculate support: legacy vote strategy versus new margin strategy, using the backwards-compatible rebuilt reducer and the current serializer. All unannotated trees and scientific histories match exactly.

| Complete API call | Median seconds |
| --- | ---: |
| Legacy agreement | 2.590558 |
| Margin default | 2.643418 |
| Difference | +0.052859 (+2.04%) |

This is incremental overhead on one regression input, not a universal timing bound. Raw paired runs and summary are in `SPECIES_MARGIN_BENCHMARK_ADOPTION2_SPECIES_RAW.json` and `SPECIES_MARGIN_BENCHMARK_ADOPTION2_SUMMARY.json`.

Local reproduction scripts and detailed outputs: `output/species-margin-adoption-20261105/` (`validate_adoption.py`, `check_end_to_end.py`, `check_panel.py`, `benchmark_adoption.py`). The untouched pre-adoption snapshot is under its `baseline/`; the original experiment remains in `output/species-support-opt-20261105/`. One preliminary benchmark harness failed because legacy metadata lacked the new ranking field; it was corrected only in the harness, then rerun as `adoption 2`. The failed timing is excluded. Long-term research memory was not modified.
