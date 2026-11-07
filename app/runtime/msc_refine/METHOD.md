# Aggregate site-contrast species-tree refinement

## Scope and attribution

This is a topology-only, fixed-scaffold **research candidate**, not an accepted
release, novel coalescent theorem, or already demonstrated neural replacement.
The quartet polynomial is CASTER-site's statistic (Zhang, Nielsen & Mirarab,
Science 2025, doi:10.1126/science.adk9688). The CPU/GPU algebra, Euler interval
implementation, and fixed-scaffold NNI integration here are separately written.

Primary references:
- https://github.com/chaoszhang/ASTER/blob/master/src/sequence.hpp (`quadPos`)
- https://github.com/chaoszhang/ASTER/blob/master/tutorial/caster-site.md
- https://www.science.org/doi/10.1126/science.adk9688

## One objective, no confidence gates

Let pi=(a,c,g,t) be the common stationary base frequencies, consistently
estimated once from the complete packed alignment. For a group's per-site
counts x=(xA,xC,xG,xT), define for disjoint groups x,y

    r(x,y) = (a²+g²)(xA*yG+xG*yA) - 2ag(xA*yA+xG*yG)
    q(x,y) = (c²+t²)(xC*yT+xT*yC) - 2ct(xC*yC+xT*yT).

The quartet support for AB|CD is

    w(AB|CD) = sum_sites [r(A,B)q(C,D)+q(A,B)r(C,D)].

The tree objective is the sum of this support over its displayed quartets.
The formula is multilinear in the four incident clade counts, so it EXACTLY
sums one-per-clade quartet scores without enumerating the quartets. An NNI
changes only quartets intersecting all four incident clades: its objective
delta is the alternative support minus current support. No score blend,
MLP prior, learned weights, confidence threshold, or dataset-dependent route.

## Search and controls

At each sweep, compute both alternative deltas at every internal edge. Apply
a deterministic positive-gain radius-one noninteraction packing and rescore.
The exclusion is an algebraic dependency schedule, not a biological gate.
CPU brute-force quartet tests check simultaneous gain additivity.

The minimal GPU0 entry exposes only `stop_ratio=epsilon` (default.005), stopping
when accepted moves/(N-3)<=epsilon, including zero moves. There is no scientific
iteration cap in this entry. An iteration-budget option survives only in the
historical CLI; it was not used for the broad comparison. No edge lengths are
fitted or reported. Initial AP-NJ construction is unchanged.

## Computational design

DFS orders the leaves so every subtree is an interval. For each bounded site
block, four integer prefix sums provide all incident-clade base counts with
two endpoint reads; a parent-side component uses the complement. Integer counts
are exact up to the stated int32 taxon limit; polynomial reductions use FP64,
fixed site-block order, no floating atomic updates or TF32 approximation.

Per scoring sweep: O(NL) time, O(NB) GPU scratch for block width B, O(N) tree state.
Deterministic gain sorting adds O(N log N) per search sweep. The complete
search costs the sum over its sweeps, not an unconditional O(NL) bound.
Input is native nibble-packed (N*ceil(L/2) bytes), on CPU or resident GPU when
within a fixed engineering allocation bound. No NxN distances or N^4 lists.
These are computational bounds, NOT evidence of100K accuracy or actual speed.

The exact executor replaces Python Euler-plan construction with a native
traversal returning the same six ordered arrays. Lossless128-leaf uint8 local
prefixes plus int32 carries reconstruct all endpoints before FP64 polynomial
evaluation. Larger launches retain the original2048-site reduction order.
Only edge neighborhoods affected by accepted NNIs are rescored; unchanged
ordered four-clade partitions retain their exact cached score and remain
eligible. State is O(N); this avoids persistent all-site clade messages but
does not eliminate the O(NL) prefix pass of a scoring sweep. It introduces no
statistical parameter or taxon-size route. Actual JIT costs are retained in
measurements, not claimed as cold-machine timing if caches are already warm.

## Statistical consistency boundary

CASTER-site's theorem concerns its specified MSC+F84 process and assumptions,
not arbitrary heterogeneous GTR/Gamma mixtures. Here a single estimated pi
is shared across the alignment: model-matched claims require a common
stationary pi. We do not import the official implementation's regional pi
estimation assumptions without justification. Independent effective gene
histories/site-pattern convergence, positive identifiable species edges, and
appropriate missingness assumptions remain necessary.

Under conditions giving a strict expected quartet preference, summing all
quartets gives the true tree as the unique population GLOBAL optimum. This
does NOT prove arbitrary-initial-tree NNI reaches that optimum. Finite stopping,
local optima and nonrandom missingness prevent an unconditional claim that this
implemented search is a statistically consistent full-tree estimator. Such a
claim requires a separate search theorem or a consistent reconstruction stage.
Do not replace this distinction with the phrase 'MSC-consistent backend'.

Missing/IUPAC states contribute zero base counts; there is no imputation.
This is complete-case quartet scoring, not an automatic correction for
missing-not-at-random, gene flow, paralogy, or biased locus retention.

## Measurement

Paired runs start from the same saved AP-NJ scaffold, never an MLP final tree.
Input parsing is measured separately and shared. Refinement includes method
setup, empirical statistics, GPU transfers and all search sweeps. Compilation
is retained in cold measurements, not silently dropped. Historical full MLP/
Transformer accuracy is separately reported; historical times are not used
as same-hardware speed ratios. Results are exposed development/regression.
