# Direct taxon completion

Supply the full aligned DNA matrix and an unrooted binary Newick backbone on
at least four of its taxa. The dedicated placement algorithm adds only missing
taxa. Pruning them out recovers the supplied backbone topology exactly.

This mode **does not construct an AP-NJ guide, run whole-tree MLP-NNI or run
whole-tree refinement**. Gene completion uses cached local likelihood fits;
species completion uses the observed-locus kernel. Normal
inference without a backbone retains the existing pipeline.

## Input and output

Enable **Taxon completion** in the browser. Upload MSA/PHYLIP on the left and
Newick on the right, in either order. Submission starts when both are present.
Species input accepts PHYLIP/PAML locus blocks; their observed boundaries are
used directly. A single species block uses the MSA-only scorer. Completion has
no nearest24 requirement: the backbone needs four taxa, and the alignment may
contain those four plus any number of additions.

The result is a **topology-only Newick**. This fast mode does not re-estimate
branch lengths or whole-tree branch support. The normal branch-support option
and de novo module settings are hidden in completion mode; CPU/GPU settings
remain available. `placement.json` records each placement, timing, predictor hash, local fitting
metadata or kernel score gaps, and evidence warnings. Gaps are uncalibrated ranking scores,
not support percentages or probabilities. No supplied branch receives 100%.

Invalid taxa, duplicate names, multifurcating backbones and malformed input are
rejected. Root position, existing lengths and existing support labels are not
constraints. An incorrect supplied backbone remains fixed by design.

```bash
bash scripts/run_local.sh --task gene --input /absolute/alignment.phy \
  --backbone /absolute/backbone.nwk --out /project/output/new-job

bash scripts/run_local.sh --task species --input /absolute/supermatrix.phy \
  --loci /absolute/locus_manifest.jsonl --backbone /absolute/backbone.nwk \
  --out /project/output/new-species-job
```

`UnifiedEngine.infer(..., backbone=Path(...))` and the web multipart `backbone`
field use the same direct path. De novo search settings are retained as
`requested-configuration.json`; `configuration.json` and `RESULT.json` record
the actual placement configuration. They do not imply that skipped modules ran.

## Deployed gene completion

The homogeneous MLP reroot DP shortlists five edges. Bounded representative
contexts are fitted with GTR+F+I+G4 on all input sites, using exact subtree-repeat
compression. Queries with overlapping candidate edges share a local reference
fit (up to eight queries and 120 reference taxa per query). Cached directional
likelihoods optimize three insertion branches at candidate positions. Each
insertion updates affected messages and candidate descendant edges, allowing
new taxa to form clades. There is no full-tree de novo inference.

Reference nuisance fits have a 32-round budget; insertion optimization uses
40 rounds and at most 160 evaluations with a bounded fallback. Fit and insertion
metadata are retained rather than claiming global ML convergence. Local nuisance
lengths are not published as inferred whole-tree branch lengths. These fixed
budgets keep sequence length and context size bounded in the placement cost.

All Python modules, four native libraries and their C++ source files reside in
`placement_likelihood/`. Rebuild on a different host with the release's Python
and `placement_likelihood/build.py` (requires g++). Runtime imports do not use
experimental output directories. The release model selector does not change the
frozen completion model, GTR+F+I+G4.

For fixed sequence length, context and model budgets, `n` queries on an initial
`N`-taxon backbone require `O(nN+n²)` time and `O(N+n)` peak space, including the
input alignment at fixed length. This is not a claim of linear total work for
a fixed missing fraction. The local shortlist is approximate, not exhaustive
full-tree maximum-likelihood placement.

## Deployed species kernel algorithm

1. Compute bounded nearest-leaf representatives for every directed backbone
   edge in two iterative traversals. Species uses up to four representative triples per internal node. Duplicate triples are removed.
2. For each missing taxon, obtain quartet scores with one representative from
   each of an internal node's three incident components. Species reuses the observed-locus statistical kernel
   or its MSA-only asset according to the actual input.
   Species retains its existing per-locus scoring
   and quartet-constant conditioning.
3. Average raw kernel scores into three node
   potentials. A reroot dynamic program evaluates **every attachment edge**.
4. Subdivide the highest-scoring edge and attach the taxon. Process missing taxa
   in alignment row order, once each. There is no iterative convergence loop or
   post-placement correction sweep.
5. Verify the complete taxon set and exact induced backbone before publishing
   the final checkpoint and result.

Let `s(v,i)` denote a node's potential for incident component `i`, and let
`d(v,e)` be the direction from node `v` toward candidate edge `e`. The placement
objective is

```
F(e) = sum over internal v of s(v, d(v,e)).
```

When crossing node `v` from edge `e1` in direction `i` to edge `e2` in direction
`j`, every other node retains its direction, so

```
F(e2) = F(e1) + s(v,j) - s(v,i).
```

One initial sum plus one tree traversal gives the exact maximizer of this
**quartet-potential objective**. It is not a claim of maximum-likelihood
placement or universally correct biological topology. Consistent, strictly
correct ideal quartet evidence uniquely recovers the correct attachment edge;
finite sequence evidence can disagree or tie. Completely flat evidence is
reported as uninformative, with a deterministic binary attachment.

For fixed representative width and model size, one missing taxon takes
`O(N * L)` scoring work and `O(N)` tree work. With fixed sequence length this is
linear in backbone size. For `M` additions, the tree grows each time, giving
`O(L * (M*N0 + M*(M-1)/2))` work, plus input setup. **It is not linear total
work when a fraction of all taxa is missing.** Alignment/bitplane storage is
`O(N*L)`, representative and edge-score storage is `O(N)`, and scoring scratch
is batched. No all-pairs distance matrix is built. Long alignments and file
parsing still cost time; subsecond placement is not a whole-job time promise.

## Evidence and limits

See [completion validation](evidence/TAXON_PLACEMENT_VALIDATION.md) for frozen
simulation recovery, true-clade deletion tests, 10k–50k stress cases, paired
warm timings and execution checks. Evaluation prunes taxa from ground truth,
then exposes only the pruned topology, the full MSA and observed locus lengths
to inference. Ground-truth branch lengths are discarded. Exact attachment and
whole completed-tree recovery are reported separately; whole-tree nRF alone
can hide misplaced taxa.

The historical corpus is exposed development data. New seeds were generated
only after configuration freeze, and their results were not used for tuning.
These model-family checks do not prove general MSC consistency or performance
on incorrect backbones, arbitrary missingness or real biological ground truth.

Direct phylogenetic placement predates this implementation; the component-query
and reroot design is an implementation tailored to these scorers, not a claim
to have invented placement. Related primary methods include
[APPLES](https://pmc.ncbi.nlm.nih.gov/articles/PMC7164367/),
[EPA-ng](https://academic.oup.com/sysbio/article/68/2/365/5079844) and
[quartet node queries](https://pmc.ncbi.nlm.nih.gov/articles/PMC3561654/).
