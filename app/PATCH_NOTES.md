# 2026-11-07 · Missing protected-input targets

Cache/output validation reserves protected symlink targets even when they are
missing. An unrelated dangling data link no longer prevents valid inference
from starting. The missing target remains blocked for writes, and an actually
requested missing input is still rejected. No data links or targets, scientific
algorithm, weights, parameters or inference budgets changed.

# 2026-11-07 · Uniform AP-NJ design repair

Same-version algorithm repair: coherent observed-intersection row controls; deterministic maximal matching of exact mutual Q minima; explicit zero-information completion;64MiB aggregate scratch; native finite checks and result health propagation. No per-file routing or new user parameter. See [measured results and limits](evidence/APNJ_DESIGN_REPAIR.md); ordinary initialization overhead and the failed gene initial-mean gate are retained explicitly.

# V1.0.2 patch notes

## Search robustness in the existing V1.0.2 (2026-11-07)

The current initializer is bounded full-site AP-NJ. Exact warmstart recurrence,
repeated-topology numerical plateaus, finite score validation and bounded
parameter continuation protect pathological inputs without raising the normal
move-ratio threshold. An optional distance-cache optimization was rejected after
it increased measured runtime; AP-NJ retains its original computation path.
See [SEARCH_ROBUSTNESS.md](SEARCH_ROBUSTNESS.md).

## Historical initializer: DIPPER (2026-11-06, superseded by bounded AP-NJ)

The user authorized replacing AP-NJ with pinned full-site DIPPER GPU placement
inside this existing version. Public input/output and downstream MLP checkpoints,
likelihood/MSC refinement and support definitions remain the same. The native
adapter shares packed input, avoids FASTA and external process overhead, bounds
scratch ownership and sizes launches for the server's L40S. Final topology parity,
pipeline, web and timing evidence is in
[evidence/DIPPER_INITIALIZATION.md](evidence/DIPPER_INITIALIZATION.md).

The following entries describe earlier engineering repairs under their original
algorithm freeze; their accuracy and speed measurements are historical.

## Bounded long-alignment posterior gathering

V1.0.1 remains frozen. This patch replaces the large-CSR-row parent gather in
`runtime/gene/runtime/repeats_fused_counts.py`. The former kernel expanded the
entire next-power-of-two row width, causing minutes of ptxas compilation and a
2,097,152-element tensor error on the 1178 × 311148 plant alignment.

Rows above the inherited 128-element tile budget now stream every parent link
through a runtime loop with four separate FP64 scalar accumulators. Results are
rounded back to the existing posterior dtype before endpoint calculations.
All large rows use one compiled width, and the <=128 path is unchanged.
No alignment sites are dropped. Model, sampling, search and stopping settings
are unchanged. An explicit parameter-fit progress event also prevents stale
MLP status during fitting.

Large-row summation order differs from V1.0.1, so full bitwise floating-point
identity is not asserted. The 72-case component check covers FP32/FP64, 1/4
categories, internal/leaf rows and degrees through 1048579. A first candidate
failed the FP64 oracle check and was rejected; only the scalar implementation
passed. Test artifacts remain in output/gene-long-alignment-fix-20261105/.

The complete plant run succeeded in 202.2107 seconds, with all 311148 sites and
1175 supports. One paired 4096 × 2048 exposed regression had identical topology,
LL difference 3.013e-7 per site, and algorithm times 7.6612/7.7224 seconds.
These are execution/regression results, not new biological accuracy evidence.

The original evidence/ files are inherited V1.0.1 reports; new validation is in
evidence/LONG_ALIGNMENT_FIX.json. Unchanged kernels share the existing project
compilation cache; Triton source hashes distinguish the new kernels.

## Engineering robustness repair in the existing V1.0.2

The scientific algorithm, checkpoints, models, all-site evidence, search,
stopping rules and branch-support definitions are unchanged.

- The active repeat builder no longer rejects all alignments above two million
  sites. It checks the actual child-pattern cardinalities and uses exact stable
  tuple radix sorting when a mixed-radix key would overflow uint64. Local pattern
  IDs retain their real int32 capacity check and first-site order.
- Endpoint-count sums retain the original reduction through capacity4096;
  larger sums use bounded 1024-row FP64 tiles. Long single-locus species sums
  use bounded 256-tile exact int64 accumulation. Floating summation association
  can differ in the large endpoint path; bitwise identity there is not promised.
- Likelihood and species counter address arithmetic is widened where needed.
  Species count layouts exceeding the existing scratch budget stream loci and
  word tiles, preserve all observations and the original 64-locus feature sums,
  and avoid exceeding CUDA's secondary launch-grid dimensions.
- CPU requests above64 now use64 effective workers, consistently with inherited
  packed extraction and global compression extensions. `runtime.json` records
  requested and effective counts; the default16-worker setting is unchanged.

Regression details, full plant completion, paired timings and rebuild hashes:
[evidence/KERNEL_ROBUSTNESS.json](evidence/KERNEL_ROBUSTNESS.json).
Run the focused cases with the tested scientific Python:

```bash
python -B -m unittest discover -s release/V1.0.2/tests -v
```

Inputs remain subject to available host/device memory and explicit structural
index capacity. The repair does not replace invalid data or failed calculations
with a placeholder tree. First-time compilation is separate from warm inference;
the G16 model regression completed but its first JIT took about104 seconds.
