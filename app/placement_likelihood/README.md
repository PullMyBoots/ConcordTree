# Cached local likelihood placement

Release-local implementation of the validated bounded gene completion candidate.
The engine calls `grouped_placement.complete` only when a gene task supplies a
backbone. Species completion retains `taxon_placement` with the statistical kernel.

All sites are retained. The homogeneous MLP selects five candidate edges. Bounded
local contexts share GTR+F+I+G4 reference fits, exact subtree repeats and directional
message caches. Each insertion optimizes its three branches. Existing backbone
topology is preserved; new taxa can form clades. Finite reference budgets do not
establish global ML convergence. Outputs are topology only.

The four `.so` libraries and their C++ sources are local. On another CPU, rebuild:

```bash
.conda/env/bin/python placement_likelihood/build.py
```

Run from the release directory. A C++17 compiler (`g++`) is required to rebuild;
normal execution loads the supplied libraries. No experiment directory is imported.
The package uses the release's existing NumPy/SciPy and inference assets.

Scientific configuration matches the retained `placement-likelihood-fast-20261113`
candidate. Only package-relative imports and an optional progress callback were
added during deployment. Main documentation: `../BACKBONE_COMPLETION.md`.
