2026-11-07 update: The tables below retain their historical scientific freeze. Current uniform design repair measurements, individual losses, ordinary timing overhead and scope are in APNJ_DESIGN_REPAIR.md; current delivered file hashes are in MANIFEST.json and APNJ_DESIGN_SOURCE_FREEZE.json.

# Bounded full-site AP-NJ initialization

Integrated into existing V1.0.2 on 2026-11-06T21:50:58.680292+00:00. No new release version.

The shared initializer uses all original columns, sparse Q-aware merging, a fixed 512-subtree exact NJ terminal solve and four fixed distance-only NNI cleanup passes. Gene distances use K2P-gamma with a bounded triplet shape estimate; species distances use JC. Pair distances use jointly observed sites. No locus truth or learned model is used by initialization. Existing task MLP weights, later refinement, support definitions and final output names remain unchanged.

| Task | Initial AP-NJ nRF | Fixed-K10 DIPPER nRF | AP-NJ warm seconds | DIPPER warm seconds | Speedup |
|---|---:|---:|---:|---:|---:|
| gene | 0.113404 | 0.241516 | 0.139171 | 0.222942 | 1.602x |
| species | 0.137289 | 0.188583 | 0.025782 | 0.052383 | 2.032x |

Primary data: 200 gene and 700 species records, all previously exposed development data. Zero failed initial trees or complete-pipeline arms. Accuracy and strict warm-speed paired intervals pass for both initializers. Individual reversals remain in the summaries; no universal per-record dominance.

Constructor timings include public validation, upload, gamma/projection, construction, terminal solve, cleanup, canonical Tree handoff and native buffer release. They exclude shared alignment parsing, Newick serialization, MLP and later inference. One warmup and four timed calls per arm use alternating ABBA order on the same packed input. Ratios use means of per-record medians, with record-level stratified bootstrap intervals.

| Task | Complete-pipeline AP-NJ nRF | DIPPER nRF | Paired difference 95% CI |
|---|---:|---:|---|
| gene | 0.033448 | 0.037021 | [-0.008785, 0.000963] |
| species | 0.066835 | 0.070653 | [-0.005075, -0.002600] |

Gene final-tree superiority remains unresolved because its paired interval crosses zero. Species final trees improve on these development records. Post-freeze new-seed checks are from the same simulator family. The DIPPER A10 10K stress case loses initial accuracy and is excluded from the project simulation mean; stress and row-permutation diagnostics are in the work report.

With fixed budgets and fixed-word arithmetic, numeric construction work is O(NL + N log N + B²R²L + B³), space O(NL + N + B²), B=512 and R=2. The preserved projection path plus minimum matching completion guarantees at least ceil(m/514) merges per outer round. This proves N log N growth at fixed L, not a bound on downstream convergence or an all-model accuracy theorem. Label-character work and ordinary Python integer-container assumptions are stated in [the proof](APNJ_COMPLEXITY.md).

Sources and the explicit L40S/SM89 build are under `native/apnj/`. No runtime compilation, FASTA intermediate or subprocess is used. The previous DIPPER adapter remains available for historical comparison but is not selected by UnifiedEngine. Initial debug trees are now `initial/apnj.nwk`.

The historical website benchmark plots retain their original scientific freeze. They have not been relabeled with these new results. Full evidence and reproduction scripts are in project `output/apnj-ninehour-20261107/`.
