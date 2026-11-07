# Uniform AP-NJ observation coordinates

The production candidate uses one rule for every public input. It does not use
file names, dataset identifiers, reference trees, task accuracy, or a missingness
threshold to select a constructor. Gene/species retain their previously fixed
distance models; both use the same construction and correction implementation.

## Source of the failure

Primary pair distances use the intersection of observed A/C/G/T positions.
The old derivative was evaluated at all-site, zero-filled statistics. Even a
small denominator discrepancy near a distance-model singularity can create a
large finite row correction. Finite-value checks alone cannot detect that error.

## Corrected control

For two disjoint active representative mixtures define normalized coordinates

- a = 1 - matched observed mass / L
- b = transversion mass / L
- c = jointly observed mass / L.

The primary proportions are p = 1 - (1-a)/c and q = b/c for c>0.
The existing packed count pass supplies a,b,c with no additional distance query.
If g(p,q) is the fixed distance correction, its linear control coefficients are

    beta_a = g_p / c
    beta_b = g_q / c
    beta_c = [g_p*(1-a) - g_q*b] / c^2.

For each active row, evaluate those coefficients at its EXACT population means.
Let M=m-1, h_i the number of sampled landmarks other than i, and A_i,B_i,C_i
the exact sums of these coordinates. Then estimate

    row_i = M*mean_sample(D_ij)
      - beta_a*(M*mean_sample(a_ij)-A_i)
      - beta_b*(M*mean_sample(b_ij)-B_i)
      - beta_c*(M*mean_sample(c_ij)-C_i).

With coefficients fixed by the current representative population, uniform
landmarks imply the ideal conditional expectation is the true distance row sum.
This is a control-variate identity, not a finite-sample variance, topology
accuracy, consistency, or finite-precision guarantee. The model's existing
saturation derivative convention is retained, with the corresponding mean
observed-site count in the half-observation floor.

Let f_i(s,b) be the observed-base mixture and o_i(s)=sum_b f_i(s,b).
For F(s,b)=sum_i f_i(s,b), O(s)=sum_i o_i(s), the new exact overlap row is

    C_i = sum_s o_i(s)*(O(s)-o_i(s)) / L.

It reuses the SAME four-channel aggregate as the original two controls.
No additional all-pairs search or dense global matrix is introduced.

## Boundary definitions and arithmetic

All public inputs use the overlap coordinate. There is no complete/missing-file
branch. Compute p as a+(a-1)*(1/c-1); at c=1 this reduces to a without introducing
an unnecessary floating subtraction, and the third centered control is zero.
For zero mean overlap, define inverse overlap as zero, making all three ratio
coefficients zero. The sampled row is retained and missing evidence is disclosed.
This does not turn finite zero-overlap distance placeholders into measurements.

An exact packed scan checks whether any column has two different observed bases.
If none does, return an explicitly uninformative balanced binary completion.
This is a defined zero-information boundary for every input, not a file exception.
Entirely unobserved taxa retain the existing explicit rejection.

## Work, memory, and progress

Each sparse round still uses the fixed32 landmarks, fixed candidate budgets and
fixed representative width. The overlap term adds a constant amount of work to
existing kernels. Raw aggregate scratch is streamed in blocks of at most131072
sites, at most64MiB; this is scratch only, not a total GPU-memory claim.

The existing contraction invariant remains active, guaranteeing at least
ceil(m/514) disjoint merges while m>512. It bounds the sum of active sizes and
round count without a wall-time cutoff or aborting a valid search. Terminal NJ
and four distance-cleanup passes retain their existing bounds. Overall initial
construction remains O(NL+N log N+B^2 R^2 L+B^3), B=512,R=2, and
O(NL+N+B^2) space. No claim is made for downstream search's global convergence.

The scan is O(NL); the no-information topology completion is O(N).
Native sorting rejects nonfinite values before comparator use. Health reports
zero-overlap pair evaluations, gamma triplet evidence and scratch use, and the
engine forwards initial evidence warnings into RESULT.json.quality.

## Validation and scientific limits

All sites, seeds, default models, budgets, checkpoints and downstream refinement
rules remain frozen. Comparisons use archived matching baseline results.
The previous calibrated-everywhere and variance-guard scalar attempts were
rejected, including their per-record losses. There is no threshold sweep or
per-file rescue path in this candidate.

The900 simulations are previously exposed development/regression data. Their
results cannot prove accuracy for every sequence, universal absence of numerical
bugs, or new biological consistency. Initial-tree changes must be followed
through the entire pipeline before final-tree claims or promotion.

## Symmetric ties: match sets of minima, not a single rank winner

Old reciprocity first picks one rank-tied argmin per vertex. If many edges have
the SAME score, every vertex may point to one low-rank neighbor; only one edge
is reciprocal even though many disjoint edges are equally optimal.

Retain the exact same scores and deterministic edge sort. Define b_i as the
minimum incident Q. Eligible edges satisfy Q_ij=b_i=b_j exactly. Greedily accept
disjoint eligible edges in the existing order, then retain the original bounded
path completion if needed. This gives a maximal matching among mutual minima,
without a numeric tolerance or overriding strict Q preferences. When every Q is
equal, the preserved Hamiltonian path guarantees at least floor((m+2)/4) merges
in the round. This is a progress theorem for equal scores, not for arbitrary
near-equal/noisy scores or a proof of better nRF. No new storage order is added.

Synthetic repeated profiles dropped from209/256 rounds to4/2 (gene/species).
Block-missing profiles still take124/84, so general weak/no-overlap tails are
not declared solved. Their progress remains strictly bounded and absent
evidence is disclosed. Do not replace exact equality with tuned tolerances.
