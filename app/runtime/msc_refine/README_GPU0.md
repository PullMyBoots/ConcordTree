# GPU0 fixed-scaffold species-tree refinement — research implementation

**Not promoted as a release default or as an MLP/Transformer replacement.**
The new executor passed the recorded finite CPU/GPU equivalence checks and
984 saved-result comparisons; this is not a universal floating-point proof.
Completed evidence and failures are in
`output/species-msc-gpu0-tenhour-20261024/`.

## Minimal interface

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=statistical-methods \
  release/v-0.1.3/.conda/env/bin/python -B -m msc_refine.species_gpu0 \
  --msa /path/to/alignment.phy \
  --tree /path/to/unchanged-scaffold.nwk \
  --output output/my-species-refinement \
  --stop-ratio 0.005
```

MSA accepts nucleotide PHYLIP or `.phy.zst`. The initial binary topology must
have exactly the same taxon labels. This component does not build or optimize
the initial tree. It neither infers gene trees nor fits/outputs branch lengths.
It has no reference-tree input or neural-model dependency in its scoring path.

The only scientific search control is `--stop-ratio`: accepted NNI moves divided
by `N-3` in a sweep. It is **not** an error probability, posterior confidence or
likelihood convergence tolerance. There is no hidden scientific round cap.
An optional iteration limit exists only in the historical CLI, not this entry.

Output must be a new project-owned directory outside `data/` and every data
symlink target. Decompression, runtime caches and all derived files stay in
project-owned non-data output directories. GPU0 is enforced before CUDA setup;
GPU1/2 are never exposed by this entry.

## One mathematical target

The fixed target is the published CASTER-site quartet polynomial under its
MSC+F84 assumptions; see [`METHOD.md`](METHOD.md) for the formula and primary
attribution. Every taxon quartet and every usable observed site participates.
One global empirical stationary-frequency vector is fixed before search.
The objective is not mixed with neural scores, fitted likelihoods, support
thresholds or data-name routes. Unknown/gap/IUPAC observations contribute zero:
this is complete-case scoring, not a correction for nonrandom missingness.

At each sweep, all eligible internal edges have two alternative NNI gains.
The original deterministic positive-gain, noninterfering packing is retained.
Only exact **execution** changes:

1. Native linear Euler/clade plan.
2. Lossless128-leaf byte prefixes plus int32 carries.
3. Larger launch chunks, retaining the original2048-site accumulation order.
4. O(N) score cache: only NNI-affected edge neighborhoods are recomputed.

Unchanged clade partitions have unchanged scores. They are not pruned for low
confidence; their cached scores remain eligible in every full-sweep decision.
The cache introduces no statistical parameter. GPU/scalar oracle gates must
verify scores, move trajectories, stopping and final topology before promotion.

## Explicit limitations

- Conditional quartet consistency under a specified MSC/F84 process is not a
  theorem for arbitrary heterogeneous GTR/Gamma/I, heterotachy or MNAR inputs.
- A unique population global optimum does not prove finite NNI search finds it.
- This execution optimization cannot improve the frozen target's accuracy.
  A faster known-inferior target is **not** a successful neural replacement.
- The separate observed-locus Fitch rate-matching experiment has a verified
  outcome-conditioning counterexample. It cannot borrow CASTER-pair's theorem
  and is not silently selected as an alternative backend or MSA-only fallback.
- All current benchmark records are exposed development/regression data;
  empirical reference trees are not biological truth. No independent holdout
  or100K biological-task accuracy claim is made.

## Outputs and verification

`prediction.nwk` contains only topology. `result.json` records hashes, the
stop/trajectory, input preparation and complete refinement time, and peak
PyTorch allocation. `tree.partial.nwk` is retained on a numerical cycle/failure;
it is not relabelled as a converged successful prediction.

The test and audit scripts in the campaign directory include independent
quartet-enumeration oracles, all-edge score-cache checks, complete-trajectory
comparisons and exact RF recounting. Timings include actual preparation/JIT/
transfers/search, with shared input parsing separately reported. Old GPU1
measurements are never used for fresh GPU0 speed ratios.
