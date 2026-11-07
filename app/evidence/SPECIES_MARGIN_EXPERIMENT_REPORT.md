# Species branch-support optimization: bounded experiment

**Decision: retain the current production method. The continuous-margin prototype does not pass the predeclared confirmation gate.**

## What was tested

The existing support labels each cached quartet by its winning topology, then averages those votes. A weak win and a strong win therefore contribute equally. Of 4,548 erroneous development branches, 507 received exactly 100; their informative-quartet count median was 96, versus 100 for correct branches at exactly 100. Small evidence count is not a sufficient explanation.

A count pseudovote correction reduced development AUC from 0.896730 to 0.886006. Using only the vote gap against the stronger alternative gave 0.895777. Those routes were closed before extracting continuous evidence.

Four parameter-free summaries were frozen in `CONTRACT.json`, then computed in one pass over the **same cached scores and final quartet classifications**. No extra inference rounds, fits, queries, model predictions or DNA scoring were added by the summaries.

| Candidate | Definition | Development AUC | ILS AUC | Complex AUC |
| --- | --- | ---: | ---: | ---: |
| agreement | Original fractional best-topology agreement | 0.896730 | 0.929476 | 0.877414 |
| mean_margin | Minimum of the two mean current-minus-alternative gaps | 0.915899 | 0.944278 | 0.898105 |
| mean_worst_margin | Mean of current score minus the strongest alternative for each quartet | 0.920417 | 0.947963 | 0.903353 |
| standardized_margin | Minimum mean gap divided by its RMS; no sqrt(Q) independence factor | 0.910877 | 0.947220 | 0.891780 |
| normalized_margin | Mean gap after dividing each quartet by its score span; minimum over alternatives | 0.917201 | 0.950199 | 0.897587 |

The winner **mean_worst_margin** was frozen in `SELECTED.json` before starting confirmation. Its development-fixed high-region cutoff is `2.68710784827` in the scorer’s raw units. This is a score cutoff, not a correctness probability or a 0–100 support label.

## Record-disjoint confirmation

The confirmation panel contains 40 additional simulations: 5 taxa sizes × 2 families × 4 records. All 80 development/confirmation simulation seeds are unique. The simulation pools were previously exposed in the research project, so this is record-disjoint confirmation for this support-formula decision, not a fresh external generalization claim.

| Confirmation group | Original AUC | Selected AUC | Difference | Paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| all | 0.917083 | 0.922821 | +0.005738 | [-0.001281, +0.014640] |
| ILS | 0.906225 | 0.906907 | +0.000682 | [-0.008084, +0.013304] |
| complex | 0.922936 | 0.940508 | +0.017573 | [+0.007388, +0.035754] |

Intervals resample whole trees within taxa/family strata, 2,000 paired draws. Exact score ties receive half credit. A separate randomized reference verifies weighted AUC against explicit record replication in 60 cases.

## High-region errors and matched coverage

| Confirmation group | Original support ≥99 | Selected development-fixed cutoff | Selected at the original retained branch count |
| --- | --- | --- | --- |
| all | 355 / 35,264; correctness 98.993%; coverage 73.454% | 323 / 33,176; correctness 99.026%; coverage 69.105% | 446 / 35,264; correctness 98.735%; coverage 73.454% |
| ILS | 199 / 15,848; correctness 98.744%; coverage 66.022% | 224 / 15,420; correctness 98.547%; coverage 64.239% | 255 / 15,848; correctness 98.391%; coverage 66.022% |
| complex | 156 / 19,416; correctness 99.197%; coverage 80.887% | 99 / 17,756; correctness 99.442%; coverage 73.971% | 159 / 19,416; correctness 99.181%; coverage 80.887% |

The apparent high-region improvement in the pooled fixed-cutoff comparison is partly a coverage tradeoff. At exactly the original retained branch count, selected-method errors increase from **355 to 446**. In ILS, even the fixed development cutoff increases errors from 199 to 224 while retaining fewer branches. Therefore the prototype fails the intended reliable-high-support behavior.

The positive complex-family AUC improvement is retained as a limited finding. The overall AUC interval includes zero, and the ILS result is weak; these must not be reported as a robust universal improvement or as confirmation of the larger development gain.

## Computational and scientific integrity

- Completed all 80 species jobs, 96,016 final internal branches. All 40 development reruns reproduce original unannotated trees, scientific search history and support values exactly.
- Exact taxon-bipartition membership in the generating species tree determines branch correctness. Input/reference hashes are checked, and ground truth is never passed to inference.
- All four summaries share one new CPU C++ pass, O(Q+N) time and O(N) storage; no per-site support matrix or retained query-score copy is required.
- Across 16 4,096-taxon cases, the extra moments reduction median was 0.0282 seconds (range 0.0218–0.0635). This is the reduction kernel timing, not a controlled whole-pipeline overhead benchmark.
- Native moments match an independent scalar implementation, including topology permutations, flat scores, inactive queries and common-score shifts.
- The production release manifest remains unchanged. The prototype, all failures and results stay in this output directory. An initial Python script name shadowed the standard `select` module before selection; the import error is preserved in `selection-import-failure.log`, and the script was renamed without changing candidate formulas.

## Recommended next mechanism: cross-stage and data-block evidence

The best next low-cost candidate is to retain **cross-stage agreement and history** during the already executed MLP-NNI and statistical refinement. A branch supported by both scorers can be distinguished from one supported only by the final scorer. Capture the already computed three-way MLP edge scores and exact split persistence; match evidence only to the same final split. A changed split without matching historical evidence is “unobserved”, not negative evidence. The two scorers still share the alignment, so this is complementary evidence rather than an independent statistical replicate.

This is feasible from current code: `frontend.py:MLPWarmstart.refine` computes MLP probabilities, and `array_mlp_nni.py:refine` already produces an edge-by-three score array every pass. Only selected moves are currently retained; stable-edge evidence is discarded. The species refinement likewise already produces edge alternatives and a move history. Capturing and transporting O(N) edge summaries would require no extra model inference; its benefit and exact branch-identity transport must be tested in a new frozen experiment.

A second candidate is **stability across observed loci or fixed site blocks**. It should use repeated pieces of sequence evidence as its units rather than pretend thousands of overlapping adaptive quartets are independent. Existing bit counting already exposes region/tile counts, but the current nonlinear barycenter kernel returns one pooled score vector. Separate block scores would require extra arithmetic/model evaluations or a validated approximation, so their cost cannot be promised negligible. Whole-tree bootstrap reruns are not required, but this route must first pass a measured low-cost feasibility check.

Final score calibration comes after a ranking method has passed record-disjoint validation. A monotone rescaling alone cannot improve AUC; threshold correctness must always be reported together with branch coverage and family strata.

## Next route and evidence

Stop this four-candidate campaign; return to mechanism design for cross-stage evidence. Do not switch to another candidate based on confirmation results and do not deploy this prototype as an improvement. Any new mechanism needs a new frozen selection/confirmation contract.

- `CONTRACT.json`: candidates, record lists, stop condition and integrity hashes.
- `DIAGNOSIS.json`, `BASELINE_FEATURES.csv`: saturation and count diagnostics.
- `SELECTED.json`: winner and cutoff frozen before confirmation.
- `CONFIRMATION.json`: paired confirmation, coverage controls, strata and all gate outcomes.
- `RUNS_{development,confirmation}.json`, `features/*.csv`, `runs/*/candidate_moments.npz`: every job and branch.
- `support_moments.cpp`, `build_moments.py`, `test_moments.py`, `collect.py`, `choose_candidate.py`, `confirm.py`, `test_auc.py`: reproducible prototype/evaluator.
- `unit.log`, `auc-unit.log`, `build.log`, `development.log`, `confirmation.log`: execution evidence.
