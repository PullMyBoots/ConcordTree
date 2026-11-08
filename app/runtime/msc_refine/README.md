# Species-tree refinement runtime

This package implements the species refinement selected by ConcordTree 1.0.2.
The supported entry points are the local Web application, `run.py` and
`UnifiedEngine`. There is no separate experimental MSC CLI in this release.

From the release directory, using its scientific CPython 3.10 environment:

```bash
.conda/env/bin/python -I -B run.py --task species --input /path/input.phy --out runs/species
```

An optional `--loci /path/locus_manifest.jsonl` supplies observed, ordered locus
lengths. They must form a nonempty partition covering the complete alignment.
Without partitions, the MSA is treated as one observed region and uses its
separately frozen model asset. Partitions are never inferred from reference
trees or benchmark labels. The current species scorer requires at least 24 taxa.

Both tasks initialize with the current native full-site K2P-gamma AP-NJ and
MLP-NNI. Species refinement then uses the frozen Gaussian barycenter kernel
and accumulated nearest24 quartet evidence. See [METHOD.md](METHOD.md).
The default posterior stopping ratio is `.005`: accepted NNI moves divided
by `N-3`, tested with `<=`. It is not an accuracy probability. Existing repeated
full-state detection and the explicit emergency sweep guard remain unchanged;
there is no Web task wall-clock deadline.

`TiledBarycenterScorer` directly extends `BarycenterState`, which owns alignment
validation, bitplanes, model loading and execution-layout metadata.
`StreamSumActiveScorer` owns persistent evidence and exact accumulated-evidence
NNI gain evaluation; `refine_active` selects and applies compatible moves.
The old CASTER polynomial, rank/fixed-evidence controls and obsolete scorer
inheritance chain are retired. Retained native build inputs are listed in
[../../NATIVE.md](../../NATIVE.md).

Outputs include the prediction, optional local-evidence branch annotations,
`species.json` with search trajectory and stop reason, and `RESULT.json` with
quality notices. Species branch lengths are not inferred. Branch annotations
are not calibrated correctness probabilities; see
[../../BRANCH_SUPPORT.md](../../BRANCH_SUPPORT.md).

The cleanup preserves assets, live arithmetic, query ordering and stopping
rules. Paired development regression evidence and a file-level change ledger
are in `../../evidence/RELEASE_CLEANUP.json`. This is not a fresh accuracy study,
a general MSC consistency proof or a guarantee of a global search optimum.
