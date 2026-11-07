# Experimental MSC topology refinement

**2026-10-24 GPU0 work:** use [`README_GPU0.md`](README_GPU0.md) and
`python -m msc_refine.species_gpu0`. The GPU1 commands below are historical
reproduction instructions, not the current ten-hour campaign entry.

**Not a replacement for the release MLP/Transformer backend.** Accuracy gates
have not passed. This package is an isolated research implementation, not a
new proof of statistical consistency for general GTR+Gamma or finite NNI search.
See `METHOD.md` and `output/species-msc-refinement-20261024/` for evidence.

## Inputs and controls

- An aligned nucleotide PHYLIP MSA (`.phy` or `.phy.zst`).
- Its unmodified initial binary Newick topology, with matching leaf labels.
- Optionally, an **observed** `locus_manifest.jsonl`: only ordered
  `alignment_length` fields are consumed. Missing partitions are never guessed.
- Exactly one search control: `--stop-ratio EPS` (default `.005`), or
  `--max-sweeps K`. EPS is accepted NNI moves divided by `N-3`, **not** an
  error probability, likelihood tolerance, or accuracy guarantee.

No gene trees, reference trees, generating models or simulation seeds are
prediction inputs. No branch lengths are inferred. AP-NJ construction is not
part of this component and has not been changed.

## Run from the project root

Use the existing CPython 3.10 research environment and physical GPU1 only:

```bash
release/v-0.1.3/.conda/env/bin/python -B statistical-methods/msc_refine/build_plan.py
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=statistical-methods \
  release/v-0.1.3/.conda/env/bin/python -B -m msc_refine.cli \
  --msa /absolute/path/input.phy \
  --tree /absolute/path/scaffold.nwk \
  --output output/my-new-species-refinement --stop-ratio .005
```

For the partition-aware input mode, add `--loci /path/locus_manifest.jsonl`.
For a fixed work budget, replace `--stop-ratio .005` with `--max-sweeps 5`.
Outputs must be new project-owned directories outside `data/` and its symlink
targets. Compressed input is copied/decompressed only to that output directory.

## Method and implementation

One published CASTER-site quartet-polynomial objective; exact all-clade
Euler-prefix aggregation; strictly positive-gain noninterfering NNI batches.
The optional boundary mode estimates the same statistic's base frequencies
per observed locus. It is an input-information ablation, not a general GTR
model upgrade. Missing/IUPAC observations contribute zero, without imputation
or a missing-not-at-random guarantee.

The equivalent compact execution uses a native linear tree-plan compiler,
one hierarchical int32 prefix buffer, and unchanged FP64 score reductions.
Scratch is `O(NB)` for fixed site block B; score work is `O(NL)` per sweep,
**not** an unconditional linear bound for the complete search. Deterministic
Python gain sorting costs `O(N log N)` per sweep. Input storage is nibble-packed
`N*ceil(L/2)` bytes; large input blocks stream from host rather than all expanding
to GPU. The number of sweeps, transfers, I/O and cold JIT must be measured.

`prediction.nwk` is topology-only. `result.json` records input/output hashes,
input mode, stopping reason/trajectory, preparation/refinement time, and peak
torch allocation. CLI output is explicitly marked experimental and unpromoted.

## Validation

```bash
PYTHONPATH=statistical-methods release/v-0.1.3/.conda/env/bin/python -B \
  -m msc_refine.test_core
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=statistical-methods \
  release/v-0.1.3/.conda/env/bin/python -B \
  output/species-msc-refinement-20261024/test_compact_execution.py
```

All currently used benchmark records are exposed development/regression data,
not independent final-test evidence. A faster exact implementation preserves
the score's existing accuracy limitations; it does not remove them.

The compact CLI now uses a native exact byte histogram to avoid NumPy's implicit
int64 copy of the packed input; observed partition frequencies are counted
without fitting an unused spectral basis. The optional research-only
`byte_prefix.py` representation reduces100K single-sweep peak allocation further,
but first-use JIT hurts some small tasks; it is NOT the CLI/release default.
Partition pi currently has dense O(L) storage in addition to prefix scratch.
