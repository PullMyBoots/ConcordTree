# Native extension sources

This directory contains the complete source used by ConcordTree's bundled native
modules:

- `candidate_graph_backend.cpp`: exact multicore construction of the
  projection-window edge union and per-node distance/tie-rank top-k relation,
  plus the ordered sparse-NJ Q-score/reciprocal-matching fold;
- `panel_score_backend.cpp`: exact reroot-DP compilation of fixed-width
  directed-edge nearest-leaf messages, linear-space deterministic connected
  contextual-panel covers, integer compilation of contextual-panel
  component groups, ordered quartet rows, perfect-matching classes, one reused
  pair-distance table for current displayed-quartet classes, the paired
  finite-float64 median used by the panel stability gate, and native ordered
  sparse MLP log-likelihood sums for SplitBank;
- `learned_nni_plan_backend.cpp`: exact native compilation of each learned-NNI
  internal edge's four nearest-leaf groups and ordered/canonical quartet rows;
- `split_compat_backend.cpp`: exact packed-word greedy selection of one ordered
  anchored-laminar split stream;
- `sequence_processor_backend.cpp`: sequential-PHYLIP parsing and packed
  sequence processing through pybind11;
- `pattern_freq_cuda_backend_host.cpp` and
  `pattern_freq_cuda_backend.cu`: PyTorch/CUDA quartet-pattern frequencies,
  fixed-partition blockwise quartet-pattern frequencies without repacking,
  exact signed projections from complete uint8 states, complete-state Hamming
  distances, complete-state NJ mismatch row sums, sparse marginalized and
  coverage-calibrated profile distances, and their exact aggregate NJ row
  sums.

The release wheel already bundles tested CPython 3.10/Linux x86_64 binaries.
Rebuilding is optional and is not part of the normal package installation.

To rebuild in an environment containing PyTorch, pybind11, a C++17 compiler,
the CUDA toolkit, and Ninja:

```bash
cd native_src
TORCH_CUDA_ARCH_LIST="8.9+PTX" python setup.py build_ext --inplace
```

Set `CONCORDTREE_BUILD_EXTENSION=pattern_freq_cuda_backend` to rebuild only that
module during backend development.

`8.9+PTX` reproduces the architecture envelope of the bundled CUDA binary
(SM 8.9 cubin plus forward-compatible PTX). Choose an architecture list suited
to the deployment GPUs when rebuilding for another fleet. The build recipe
maps local source/environment prefixes out of compiler diagnostics so private
build-machine paths are not embedded in the generated modules.

The resulting `.so` files are emitted in this directory. To use rebuilt files,
replace the corresponding files under
`src/concordtree/assets/backends/` before building the main release wheel,
then update the hashes in `src/concordtree/assets.py`.

The historical tree also contained `pattern_freq_cuda_backend.cpp`; it is
byte-identical to the included `pattern_freq_cuda_backend_host.cpp` and is not
duplicated here.
