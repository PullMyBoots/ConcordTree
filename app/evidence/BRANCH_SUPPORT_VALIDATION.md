> Historical validation of the earlier sampled-quartet-agreement default. Species support now uses the AUC-prioritized margin index; see [SPECIES_MARGIN_ADOPTION.md](SPECIES_MARGIN_ADOPTION.md). The measurements below retain their original meaning.

# Branch support delivery checks — 2026-11-05

The default unified API and CLI now annotate final internal edges and write
`branch-support.json`. Gene trees report **Local likelihood weight**; species
trees report **Sampled quartet agreement**. Both are uncalibrated local evidence
summaries on a 0–100 scale. Neither is bootstrap support, a posterior probability,
or a calibrated probability of a correct split. Their definitions and limitations
are in [BRANCH_SUPPORT.md](../BRANCH_SUPPORT.md).

## Execution equivalence and final-state mapping

Thirteen CPU unit tests passed, including normalization, ties, missing evidence,
invalid evidence, quoted names, Newick annotation removal, preservation of topology
and lengths, and iterative fingerprints for a 10,000-tip caterpillar tree. The
native species reducer agrees with its independent NumPy implementation.

Nine paired GPU executions with support disabled/enabled all passed:

| Case | Purpose |
| --- | --- |
| Gene 32, default | Existing final scan reused; original release output preserved |
| Gene 32, one round | Changed final state triggers upgraded final verification |
| Gene 32, JC69, one round | Alternate substitution model and final scan fallback |
| Gene 32, HKY+G4, two rounds | Variable-rate model path and fallback |
| Gene 256, default | Larger topology; original release output preserved |
| Species 32, default | Original release output and final mapping preserved |
| Species 32, stop ratio 0.4 | Final change triggers cached-quartet remapping |
| Species 32 × 135,900, MSA only | Long ILS alignment; original output preserved |
| Same alignment, observed loci | Partition-aware path; original output preserved |

For every pair, the unannotated final tree and scientific search history were
identical. Gene model parameters were identical. Instrumentation confirmed equal
parameter-fit counts and equal quartet prediction/query counts. For gene trees,
the last captured NNI scan's tree and model were checked against the final output,
so scores from an earlier topology or earlier fitted model cannot silently pass.
Five cases also matched pre-feature V1.0.1 output bytes.

An independent scalar four-component traversal checked all 70 possible quartets
of an eight-tip unrooted tree against the production GPU classifier and native
vote summaries on all five internal edges, including tied and flat scores.
Input hashes were unchanged after regression. Public CLI runs checked both the
`--no-support` gene path and the default native-enabled species path.

The original inference native binaries remain unchanged. The new CPU-only
`_branch_support_reduce` extension was explicitly compiled from the supplied source;
no build occurs during inference. Build command and hashes accompany this report.

## Additional work

There are no bootstrap replicates, repeated inference runs, additional parameter
fits, or additional quartet scores. Gene runs normally reuse O(N) aggregate scores
from the final search scan. If a subsequent move or refit invalidated that scan,
the mandatory final likelihood verification becomes one NNI evaluation, with no
proposals or moves. This can add local likelihood scoring; its occurrence is
explicitly recorded as `additional_nni_evaluations: 1`.

Species runs reuse cached unique quartet scores and the last integer topology
classification. If the last sweep changed the tree, one classification pass maps
those same quartets to the final tree. This adds integer tree mapping and O(Q+N)
aggregation, without adding queries or reading/scoring DNA again. The retained
classification requires four bytes per cached quartet. Branches without informative
evidence get `null`, not an invented score.

## Paired whole-API timing

L40S GPU0, 16 CPU threads, NUMA node 0. Both tasks used the same previously exposed
gene simulation: `data/test/simulation/alisim-homogeneous-v3/n4096/000/alignment.phy`
(4,096 taxa × 2,048 sites). The species run is an execution stress case, **not** a
species accuracy experiment. Each arm had one warmup and three measured runs,
alternating support-on/off order in one process. Both checkpoints were preloaded.
Timing includes input processing, full inference, support export and all API file
writing; it excludes Python startup and cold checkpoint loading.

| Task | Support disabled median | Enabled median | Additional time |
| --- | ---: | ---: | ---: |
| Gene | 6.0022 s | 6.0837 s | 0.0816 s / 1.36% |
| Species | 2.4291 s | 2.6347 s | 0.2056 s / 8.46% |

All paired tree/history/model equivalence assertions passed. Gene runs reused the
last search scan. Species runs remapped 1,850,787 cached unique quartets once after
the final topology change, without new predictions. Both outputs covered all 4,093
internal edges. Three measured repeats on one record establish a local execution
cost, not a universal slowdown bound or a biological calibration result.

An earlier NumPy-only reduction measured 0.1511 s / 2.45% extra for gene and
0.4965 s / 20.05% for species on the same case. The delivered CPU reducer and compact
sidecar reduce that cost; the earlier summary is retained for transparency. No
inference/search configuration was tuned in this performance pass.

## Evidence

- `BRANCH_SUPPORT_UNIT.log`: 13 passing unit tests.
- `BRANCH_SUPPORT_REGRESSION.json`: nine paired execution checks and operation counts.
- `BRANCH_SUPPORT_QUARTET_REFERENCE.json`: independent exhaustive quartet check.
- `BRANCH_SUPPORT_INPUT_INTEGRITY.json`: regression input hashes.
- `BRANCH_SUPPORT_CLI.json`: public command-line checks.
- `BRANCH_SUPPORT_BENCHMARK_NATIVE_{SUMMARY,GENE_RAW,SPECIES_RAW}.json`: delivered timing.
- `BRANCH_SUPPORT_BENCHMARK_INITIAL_SUMMARY.json`: earlier reduction timing.
- `BRANCH_SUPPORT_BUILD.log`: explicit native build command and source/binary hashes.
- `BRANCH_SUPPORT_SOURCE_AUDIT.json`: imported sources, baseline, binary and input checks.

Reproduction scripts and complete job outputs remain in the project-local
`output/branch-support-validation-20261105/` directory (`validate_support.py` and
`benchmark_support.py`). Unit tests can be rerun from the project root:

```bash
release/v-0.1.3/.conda/env/bin/python -B -m unittest discover \
  -s release/V1.0.1/tests -v
```

No calibration experiment against simulated true splits was performed. These
checks establish execution integrity and cost, not that a value of 95 has any
particular error rate. No common bootstrap/localPP threshold should be attached
to the two metrics.
