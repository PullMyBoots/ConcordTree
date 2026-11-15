# Direct taxon placement validation

Frozen configuration: gene homogeneous MLP, two representative triples and mean log probabilities; species observed-locus/MSA-only statistical kernel, up to four triples and raw means. Missing taxa are processed once in alignment row order. No AP-NJ or whole-tree refinement is used.

## Exact recovery

For each alignment, prune 1, 3 or 8 taxa from its ground-truth topology. Discard branch lengths. Only the pruned topology, full MSA and observed locus lengths enter inference. To assess a query, prune the other missing taxa from prediction and truth and compare the exact induced topology. The completed-tree column additionally tests relationships among simultaneously missing taxa. Repeated deletion sizes are correlated conditions, not independent alignments.

| Panel | Task | Alignments / conditions | Exact attachments | 95% alignment-bootstrap CI | Entire tree exact |
|---|---|---:|---:|---:|---:|
| historical | gene | 40 / 120 | 429/480 (89.38%) | [85.00, 93.54]% | 83/120 (69.17%) |
| historical | species | 56 / 168 | 620/672 (92.26%) | [89.58, 94.64]% | 125/168 (74.40%) |
| fresh | gene | 32 / 96 | 363/384 (94.53%) | [91.41, 97.14]% | 78/96 (81.25%) |
| fresh | species | 32 / 96 | 332/384 (86.46%) | [80.99, 91.41]% | 55/96 (57.29%) |

Historical confirmation uses 40 gene alignments (256–4096 taxa) and 56 species alignments (32–1024 taxa, ILS/complex). They come from historically exposed development sources; these are regression results, not an independent final holdout. The frozen placement-confirmation source records differ from the placement development panel.

The fresh panel was generated after freeze without subsequent tuning: 32 new gene records (256/1024 taxa, new random topologies and AliSim seeds, heterogeneous branch regimes, GTR rate variation and selected indels); 32 new species records (16 ILS + 16 complex, 65 taxa including the simulator outgroup, 32 loci of 512 sites, three fixed GTR+G4 parameter sets). Gene and species information content differs from the historical corpus; do not pool them to claim one universal accuracy. This verifies these simulator families, not arbitrary biological data.

Nearest-tip mismatch attachment is a deliberately simple comparator, not APPLES or maximum-likelihood placement. It uses all observed sites with pairwise missing-data exclusion.

- Historical gene nearest-tip: 245/480 (51.04%).
- Historical species nearest-tip: 361/672 (53.72%).

## Whole missing clades

Select exact-size true clades of 2, 4 or 8 taxa on a pre-spaced subset of confirmation alignments. This is an additional exploratory robustness check, not another independent sample.

- gene: 70/70 exact induced attachments; 13/15 entire completed trees exact across 5 alignments. A whole missing clade can attach at the right backbone edge while its internal relationships remain wrong.
- species: 180/196 exact induced attachments; 32/42 entire completed trees exact across 14 alignments. A whole missing clade can attach at the right backbone edge while its internal relationships remain wrong.

## Warm paired end-to-end timing

Same input and backbone, GPU0 L40S, four CPU threads, no support annotations, alternating execution order, one warmup followed by three timed repetitions per arm. The comparator is the archived AP-NJ-guide + constrained full pipeline. Values are medians of file-to-tree time; they include validation, parsing, format conversion/decompression when needed, setup, placement/search, verification and export, but exclude interpreter/engine construction. Warm filesystem caches and the narrow panel limit extrapolation.

| Input | Missing | New (s) | Previous (s) | Speedup |
|---|---:|---:|---:|---:|
| gene/n256/000 | 1 | 0.0278 | 0.2143 | 7.72× |
| gene/n256/000 | 8 | 0.0634 | 0.2136 | 3.37× |
| gene/n4096/000 | 1 | 0.5875 | 2.2704 | 3.86× |
| gene/n4096/000 | 8 | 1.6262 | 3.1511 | 1.94× |
| species/small/n32/ILS/0 | 1 | 0.0843 | 0.1037 | 1.23× |
| species/small/n32/ILS/0 | 8 | 0.0935 | 0.1103 | 1.18× |
| species/large/n1024/ILS/0 | 1 | 2.3620 | 3.0631 | 1.30× |
| species/large/n1024/ILS/0 | 8 | 2.7704 | 3.1019 | 1.12× |

Most species wall time on long compressed inputs is input preparation. Reducing inference work does not remove that I/O cost. The bounded old/new accuracy panel is retained in the project production validation; it is not used to claim universal noninferiority.

## Large-scale stress

Three already exposed AliSim inputs from the DIPPER-paper collection are used only for completion stress. These results do not enter the existing main technical-report comparisons. One record per size provides execution evidence, not a reliable large-scale accuracy estimate.

| Taxa × sites | Missing | Correct attachments | Read/validate (s) | Predictor setup (s) | Placement (s) |
|---|---:|---:|---:|---:|---:|
| 10,000 × 16,092 | 1 | 1/1 | 2.61 | 1.67 | 0.42 |
| 10,000 × 16,092 | 8 | 4/8 | 2.61 | 1.67 | 2.85 |
| 20,000 × 20,812 | 1 | 1/1 | 6.61 | 0.41 | 0.76 |
| 20,000 × 20,812 | 8 | 6/8 | 6.61 | 0.41 | 5.72 |
| 50,000 × 37,993 | 1 | 1/1 | 30.20 | 2.03 | 1.95 |
| 50,000 × 37,993 | 8 | 7/8 | 30.20 | 2.03 | 14.80 |

Read/setup are measured once per alignment and shared by the deletion conditions; adding the columns estimates a fresh per-condition scientific run, not measured process-startup wall time. Large sparse alignments remain difficult: A10 recovers only 4/8 additions in the multiple-deletion condition despite small whole-tree nRF. Do not describe these cases as uniformly high accuracy.

## Execution validation

- Six dedicated tests: all attachment edges under ideal quartet evidence, brute-vs-reroot objective equality, whole missing clades and insertion orders, bad/flat evidence, and a 2004-taxon ladder.
- Five backbone tests, eight I/O/runtime contracts, three standalone-runtime tests and 34 web API tests passed.
- Dedicated engine runs patch AP-NJ initialization and both complete refinement pipelines to raise if called: all selected completion jobs still succeed. No hidden fallback is allowed.
- Standard gene regression retains the exact serialized output; the standard species regression retains its topology (and previously checked support behavior).
- Browser validation and current release manifest verification are recorded in the project validation ledger.

- Six additional native boundary runs passed: both tasks with constant sites, disjoint observations, and an already complete backbone. Uninformative evidence yields finite zero-gap placements with explicit warnings; a complete backbone performs zero quartet scoring.
- Input audit covers 482 paths: all 450 previously recorded hashes match; 32 additional paths were fingerprinted without a prior hash and are not described as pre/post integrity comparisons.

## Rejected alternatives

A correction sweep was rejected because it added cost without a reliable accuracy gain. A late per-locus homogeneous gene-MLP average was also closed on the original species development panel, using matched alignment insertion order: the released statistical kernel recovers 171/192 attachments (mean placement 0.06784 s; median 0.02369 s), versus 166/192 for locus MLP (0.30514 s; 0.16346 s). Production never changed, and this alternative was not evaluated or tuned on fresh validation samples.

## Limits and reproducibility

The algorithm maximizes its fixed quartet-potential objective over every edge; it does not solve maximum-likelihood placement, calibrate a correctness probability, or prove general MSC consistency. Multiple additions are greedy and order-dependent. Backbone correctness is assumed for the recovery experiment; a wrong user backbone is deliberately preserved. Fixed sequence length and a fixed number of additions give linear taxon scaling, whereas adding a fraction of all taxa has quadratic total placement work. Output is topology-only; score gaps are not branch support.

All final simulation jobs preserved the backbone and completed. Preliminary harness errors (compressed input passed directly to the native PHYLIP parser, a baseline constructor mismatch, NumPy bool JSON serialization in a rejected correction candidate, and an incorrect regression locus fixture) were fixed; they remain visible in exploratory logs. They are not silently substituted with accurate trees.

The case/deletion specification is [TAXON_PLACEMENT_INPUTS.json](TAXON_PLACEMENT_INPUTS.json); final per-condition results are [TAXON_PLACEMENT_RECORDS.csv](TAXON_PLACEMENT_RECORDS.csv); aggregates and timing repetitions are [TAXON_PLACEMENT_VALIDATION.json](TAXON_PLACEMENT_VALIDATION.json). Project scripts, Newick outputs and raw timing repetitions remain under `output/taxon-placement-20261113-0707/`. Model weights are unchanged. No new release version was created.
