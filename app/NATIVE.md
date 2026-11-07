# Native runtime

All inference uses bundled binaries; no compiler runs automatically. The tested ABI is CPython 3.10, Linux x86_64, PyTorch 2.5.1+cu121. A fresh host or rebuild has not been certified.

`runtime/gene/native_src/` contains sequence packing/filtering, pattern frequency CUDA, optimized candidate graph, learned NNI planning and panel scoring sources. `runtime/gene/runtime/` contains the sources for packed extraction, global patterns, repeat builders and tree plans, together with the inherited explicit build scripts. `runtime/msc_refine/` contains the original compact posterior sources, headers and explicit `build_*.py` scripts for all its bundled extensions.

The learned NNI planner source is copied from `release/v-0.1.3/native_src/learned_nni_plan_backend.cpp`; the panel source is copied from `release/V-0.2.0/native_src/panel_score_backend.cpp`. These are source provenance paths, not runtime dependencies. The full imported-file provenance is in `SOURCE_PROVENANCE.json`.

Rebuilding changes binary hashes. The checked numerical assets are frozen to their tested hashes in `runtime/gene/src/concordtree/assets.py` and the release manifest. A rebuild is a new artifact requiring its own audit and regression, rather than an automatic fallback after an integrity failure.

The active `_repeat_builder_schedule` extension was explicitly rebuilt for the
V1.0.2 engineering robustness repair. The exact tuple radix fallback and build
hashes are recorded in `evidence/KERNEL_ROBUSTNESS.json`; other pre-existing
inference binaries retain their previous hashes. Rebuild this extension with
`runtime/gene/runtime/build_repeat_schedule.py --replace`, then repeat validation.

The branch-support extension `_branch_support_reduce` is a new CPU-only aggregation module. Its source and explicit build script are `runtime/gene/runtime/branch_support_reduce.cpp` and `build_branch_support.py`; it uses pybind11 and does not score sequences or select moves. Its legacy `reduce` entry still returns votes; the new `reduce_margins` entry accumulates the signed strongest-alternative gap in the same pass. It was explicitly rebuilt for this adoption and checked against the NumPy and independent scalar references. No runtime compilation occurs.

Some inherited class modules and extension variants remain because the selected MSC scorer inherits through them. The public supported entries are `run.py` and `UnifiedEngine`; historical experimental entry points are not a second supported pipeline.


The active initializer is the bundled `_ct_apnj_packed`, `_ct_apnj_graph`, `_ct_apnj_terminal`, `_ct_apnj_topology` and `_ct_apnj_local` closure. Complete CPU/CUDA sources and an explicit build script are under `native/apnj/`; run `python native/apnj/build.py --arch 89` from the scientific environment to rebuild explicitly. Its numerical targets and exact graph tie rules are frozen in `evidence/APNJ_SOURCE_FREEZE.json`. Packed rows upload directly or through cudaMemcpy2D, with checked integer domains, owned CUDA buffers, compact integer candidate keys and owner-local duplicate-walk removal. These execution changes preserve all 900 frozen initial trees and actions. No automatic compiler, external process or FASTA conversion is invoked. Other GPU architectures and ABIs require a separate rebuild and regression.

The earlier `_dipper_native` adapter and its pinned MIT-attributed sources remain under `native/dipper/` for historical comparison. UnifiedEngine no longer selects it. Its original integration evidence is retained.

For the 2026-11-07 search repair, `_tree_plan_native` was explicitly rebuilt to
reject empty, asymmetric and unknown-leaf graphs during its existing traversal.
Cycle/disconnection rejection already existed in this native traversal. Python
and species serializers now reject those invalid graphs as well. Normal native
message ordering and Newick output are retained. Build and regression evidence
is recorded in `evidence/SEARCH_ROBUSTNESS.json`.
