# Search robustness in V1.0.2

The 2026-11-07 repair stays in this release. DNA distance models, checkpoints,
likelihood/MSC objectives, branch-support definitions and normal `.005` stopping
ratios are retained. Special termination is recorded separately from convergence.

| Process | Finding | Repair |
| --- | --- | --- |
| MLP warmstart | Deterministic NNI can revisit a complete state; an emergency limit formerly allowed 10,000 passes before failure. | Exact Brent recurrence detection keeps one byte-for-byte neighbor-array checkpoint. A repeated state ends the warmstart, records a notice and passes its tree to the task-specific posterior. |
| Small taxon counts | `floor(.005*(N-3))` is zero below 203 taxa. A one-edge cycle therefore cannot satisfy the normal stop rule. | Preserve the threshold and detect recurrence, rather than rounding the threshold up and skipping legitimate improvements. |
| Gene topology search | Parameters can change while topology repeats. Tiny repeated changes may be indistinguishable from the existing full-likelihood verification tolerance. | Require eight low-gain rounds, repeated occurrence of every topology in that window, and total gain within `.01 + 1e-6*L`. Retain the best verified state and mark search convergence unconfirmed. New topologies or meaningful cumulative gains prevent this stop. |
| Continuous parameter fitting | The 32-iteration optimizer budget applies per local solve, not to the outer rate-mode continuation loop. | Detect repeated parameter seeds, allow at most 64 continuations and report any intervention as nonconvergence. The invariant-profile fitter retains a better seed if a local solve regresses. |
| MSC active evidence | Repeated topology alone is not a cycle when accumulated evidence grows. | Compare the complete compiled topology and append-only evidence epoch. Objective and NNI gains must be finite; retain the existing objective-consistency and positive-gain checks. |
| Missing/constant observations | All-zero quartet features can feed network biases; an entirely constant alignment formerly failed at the MLP stage. | No jointly observed quartet sites produce uniform probabilities. No variable sites skip the uninformative warmstart with an explicit notice. |
| Tree utilities | Degree checks alone do not exclude disconnected/cyclic graphs; unguarded traversals can run indefinitely. | Validate binary-tree node counts, symmetric edges and connectivity, and reject repeated/disconnected traversal states before serialization or identity generation. |
| AP-NJ contraction | The 75-second A50 case has slow but strictly decreasing active-node count, not an oscillation. | Preserve pairing/contraction rules. An exact distance-cache trial increased wall time and was rejected; production retains the original constructor. |

Recurrence and the fixed plateau window add O(N) storage. The constructor retains its existing bounded-contraction proof and resource
bounds. No distance cache or other unvalidated speed change was promoted.

Both posterior loops have a 10,000-sweep emergency failure guard. The website
also retains its 900-second subprocess deadline. These limits are operational
backstops, not claims that a budget-limited result has converged. A native/GPU
call cannot be forcibly interrupted safely inside the library process; production
services should retain subprocess isolation and the external deadline.

`RESULT.json.quality` exposes posterior search convergence, parameter convergence
where relevant, and all initialization/search interventions. `status: complete`
means the pipeline produced a verified tree; it is not a global-optimum or
universal-accuracy certificate. The web adapter preserves notices with downloads.

Validation and limits are recorded in `evidence/SEARCH_ROBUSTNESS.json`. Original
inputs remain immutable; no research memory is written without Human approval.
