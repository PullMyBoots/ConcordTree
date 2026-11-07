# Lightweight branch evidence

Every successful default run now annotates final internal branches. No bootstrap trees, repeated inference runs, additional parameter fits, or additional quartet model predictions are required. The metric is explicit in `RESULT.json` and `branch-support.json`. These are **uncalibrated local evidence summaries**, not confidence probabilities. They do not inherit bootstrap/SH-aLRT/localPP thresholds.

## Gene trees: Local likelihood weight

The existing full-site NNI scanner evaluates the incumbent topology and two local alternatives, using the same four cut messages and final substitution-model parameters. Its two-pass central-edge profile already provides:

- `c0`: the best observed log-likelihood gain for the current topology's central length;
- `c1`, `c2`: the corresponding approximate-profile gains for the two NNI alternatives.

We compute `w_j = exp(c_j - max(c)) / sum_k exp(c_k - max(c))` and annotate the current split with `100*w_0`. This has a concrete conditional likelihood-weight interpretation, without a temperature parameter. It is not a fully optimized likelihood ratio test or Bayesian integration. All three candidates fix the surrounding subtrees and model parameters, and the central profile is approximate.

Equal local scores give33.3333; a preferred alternative can give the reported branch less than33.3333. A printed100 can arise from a large conditional score difference and numerical rounding; it does not certify truth. More sites, model misspecification and the limited alternative set can make weights extreme.

Only O(N) aggregate arrays are retained from the last search scan. A state revision counter invalidates them after any accepted topology move or parameter refit. If a round budget or stopping rule leaves such a change after the last scan, the algorithm upgrades its **already mandatory final verification** to a local NNI evaluation without proposing or applying moves. The output records `additional_nni_evaluations: 0` or1. This is extra local scoring when needed, not another inference run. Normal stable-stop runs reuse the existing scan.

## Species trees: Quartet margin index

The default species method is `mean_worst_quartet_margin`, selected as `mean_worst_margin` in the frozen development experiment. For each final internal edge, include cached unique quartets with one tip in each of its four incident subtrees. Let `s0(q)` be the frozen score of the final topology and `s1(q), s2(q)` those of the two alternatives. The ranking score is:

`m = mean_q[s0(q) - max(s1(q), s2(q))]`

This averages each quartet's advantage over its strongest alternative. A negative value means the alternatives are preferred on average. Fully flat quartets are excluded using the original roundoff tolerance `64*eps*max(1,max(abs(scores)))`. No informative quartets means `null`, with no numeric Newick label; missing evidence is never replaced with0 or100.

`branch-support.json` retains this signed raw value in **`mean_worst_margin`**, the authoritative field for ROC-AUC and ranking. Numeric Newick labels and the sidecar `value` use the fixed monotone display convention:

`index = 50 + 50*m/(1+abs(m))`

The unit scale1 is fixed, not fitted to correctness labels. Zero margin maps to50, positive margins exceed50, and negative margins lie below50. This is a bounded **index, not percentage agreement or correctness probability**. A value95 or99 has no bootstrap, legacy-agreement or calibrated accuracy meaning. Species numeric labels use17 significant digits to preserve the float64 display value; viewers may round it for presentation. Use the raw field for statistical evaluation, since extreme values can still saturate under finite precision. All80 validation trees retain exactly the raw ranking after this mapping and serialization.

The original fractional best-topology agreement is retained as `sampled_quartet_agreement` (0–100), alongside all three `topology_vote_fractions`, informative/active/uninformative quartet counts and two-way tie counts. Two-way best-score ties split the diagnostic vote equally; they still contribute their raw margin. Existing consumers should inspect the method ID before interpreting `value`.

The compact search reuses its final cached classification, or remaps the same quartets once if the last sweep changed topology. The CPU reducer sums margins and diagnostic votes in one O(Q+N) pass with O(N) storage, retaining no per-site support matrix. The final classification occupies4 bytes per quartet in addition to the existing cache. There are **no extra DNA scores, quartet predictions, fits or inference rounds**. A NumPy implementation is retained as a regression reference.

Quartets are adaptive, correlated and share an alignment; their count is not an independent sample size. This index is not ASTRAL localPP, gCF or sCF, and the frozen scorer's scientific limitations remain. The user explicitly prioritized pooled AUC after reviewing the fixed-coverage tradeoff: development AUC0.896730→0.920417; record/seed-disjoint confirmation AUC0.917083→0.922821. The paired tree-bootstrap95% interval for the confirmation difference crosses zero, and errors at the same73.4544% retained coverage increase355→446. Adoption does not establish universal improvement or satisfy the original stricter experiment gate. See [the adoption record](evidence/SPECIES_MARGIN_ADOPTION.md).

## Files and branch identity

- `prediction.nwk`: final tree with numeric internal labels and `[&support_id=b000001]` comments. Terminal branches and the artificial display root receive no support.
- `prediction.unannotated.nwk`: the original final Newick, preserving the existing serialization of topology and gene branch lengths.
- `branch-support.json`: method definition, calibration flags, coverage, per-edge details and computation provenance.
- `RESULT.json`: a compact `branch_support` summary and sidecar paths. With `--no-support` / `branch_support=False`, this is `{"enabled":false}` and the main Newick remains unannotated.

For example `(... )72.4[&support_id=b000001]:0.012` associates72.4 with an internal edge and keeps its branch length0.012. The sidecar row with the same ID describes how that72.4 was obtained. Some tree viewers hide comments but still display the numeric label.

Exact node pairs assign scores to branches. A rerooting-invariant128-bit XOR taxon-name fingerprint plus smaller-side count is supplied as `split_id` for UI mapping; it is not an exact mathematical split certificate and never controls scoring or inference. No growing bitset or per-branch leaf list is needed, including for skew trees.

For the website, display the method name together with the number, and show evidence counts on hover for species trees. Unsupported branches should remain visually distinct. Do not label these numbers “bootstrap”, “posterior probability” or “accuracy”, or apply a shared high/low threshold without calibration.

## Verification

Tests cover numerical normalization, ties, missing evidence, invalid records, quoted taxon names, annotation removal, exact topology/length preservation and10,000-tip skew-tree traversal. Paired executions verify no additional fitting/quartet prediction and unchanged scientific search histories. The updated default, prototype-equivalence and adoption regression checks are recorded in [evidence/SPECIES_MARGIN_ADOPTION.md](evidence/SPECIES_MARGIN_ADOPTION.md). Historical execution costs and the independent CPU quartet-mapping check are recorded in [evidence/BRANCH_SUPPORT_VALIDATION.md](evidence/BRANCH_SUPPORT_VALIDATION.md).
