# Current species refinement method

ConcordTree builds a full-site K2P-gamma AP-NJ tree and applies the frozen
MLP-NNI predictor before entering this component. The component refines the
topology with a frozen, group-invariant Gaussian kernel model. It does not
use the retired CASTER-site polynomial or infer gene trees as intermediate
inputs. Reference trees and generating models are not prediction inputs.

## Quartet representation and scoring

`BarycenterState` validates the packed alignment and observed locus partition,
encodes the alignment into bitplanes and loads the selected frozen asset.
`pattern_layout.py` defines the 15 canonical nucleotide equality-pattern
identities; `msc_kernel.py` defines their candidate and within-split group
permutations and loads model parameters. `DirectMSCKernel` expands the frozen
landmarks and coefficients for the within-split symmetry group.

`TiledBarycenterScorer.predict` counts each requested quartet over all original
sites in each observed locus. Missing or ambiguous bases are omitted from
complete quartet observations; they are not imputed. Constant quartet patterns
are conditioned out within each locus. Per-locus square-root frequency features
are combined and normalized into a barycenter, then scored for the three
quartet resolutions by the Gaussian evaluator. The tiled/streamed counter
choice changes scratch layout only. No query, locus or site subsampling is
introduced by that layout choice. The active transposed Gaussian kernel retains
its existing FP32 operation order and FP64 partial-sum reduction.

MSA-only and observed-locus inputs select their respective frozen assets.
The observed partition must cover the alignment exactly. Treating an MSA as
one region does not reconstruct unobserved gene boundaries.

## Persistent evidence and topology search

`StreamSumActiveScorer` proposes the current nearest24 discriminating quartets,
canonicalizes and deduplicates them, evaluates new quartets once, and retains
their frozen scores. Native plans and GPU topology classification compute
NNI gains over all accumulated evidence. The objective is the sum of frozen
quartet log scores consistent with the current topology. Evidence can grow
between sweeps; objective verification accounts for newly added evidence.

`refine_active` selects positive-gain compatible NNI actions and applies them
in the existing deterministic order. It stops when accepted moves divided by
`N-3` is at most the requested ratio (default `.005`). Repetition detection
uses the complete compiled topology and append-only evidence epoch. A repeated
full state returns an unconfirmed-convergence warning; nonfinite values or an
exhausted emergency sweep guard fail explicitly. No stopping rule changed in
the cleanup.

Final branch support summarizes the signed strongest-alternative margin from
cached quartet evidence. If needed, existing quartets are remapped to the final
tree; no additional quartet model scores or parameter fits are introduced.
The display transform is defined in `BRANCH_SUPPORT.md` at the release root.
It is not bootstrap support or a calibrated correctness probability.

## Scope and validation

The method is a finite learned scoring model with heuristic adaptive queries
and local NNI search. General MSC consistency, decisiveness of the eventual
query set and a global optimization guarantee are not proved. Species branch
lengths are not fitted. Existing development benchmarks retain their original
provenance and limitations.

The 2026-11-08 cleanup removes unused experimental scorers, entry points and
native variants. Shared tree/NNI actions and canonical pattern identities were
retained or extracted. Constructor operations retain their original order;
active score arithmetic, trained assets, native binaries and stopping rules
remain unchanged. `evidence/RELEASE_CLEANUP.json` at the release root records
paired tree, parameter, objective, support and search-trajectory checks.
