# Experimental modern runtime diagnostic

**Not admitted to the stable distribution. No Blackwell hardware execution has been performed.**

Run `modern-diagnostic-48-v1`: **complete**, 48/48 cases completed. The stable scientific runtime, archive and image remain unchanged.

Candidate: Torch 2.11.0+cu128, CUDA user libraries 12.8, Triton 3.6.0, NumPy 2.2.6, SciPy 1.15.3. All four MLP/MSC weight/configuration files match the stable artifact SHA256. Source changes are the pointer-type constexpr API adaptation, an identical-source C++ ABI backend rebuild/hash, and a startup temporary-alias difference. See the JSON provenance for exact hashes.

## Evidence contract

This diagnostic continues after byte differences to locate their extent. It does not relax the stable byte-identity gate. Unrooted bipartitions are compared directly for every initial/final Newick, independently of reference RF. Branch lengths are compared on shared canonical edge splits; two edges introduced by a binary root are summed. Support labels are compared by the corresponding split. Reordered child lists alone do not count as a topology change. A separate comparator check covers identical/reordered trees, branch/support differences and different topologies with equal reference RF. The JSON includes actual per-case reference accuracy, per-trial timings and per-tree comparisons.

The original 48-case manifest and input hashes are retained. Baseline and candidate run alternately on one L40S with 16 CPU threads and the same CPU affinity. Each case receives an unmeasured complete warm-up; simulated cases have three measured pairs and empirical cases one. Both main threads start on the same CPU before restoring full affinity. Warm file-to-tree timings exclude startup and first JIT compilation. Cases outside 5% are observations requiring investigation, not an admission test passed by averaging.

## Results

Among completed cases: 48/48 have identical initial/final topology sets; 48/48 have identical reference accuracy measurements; 14/48 have byte-identical initial/final artifacts.

Aggregate warm runtime ratio: **1.001292**; median case ratio: **1.006081** (candidate / frozen baseline).

1 completed case is more than 5% slower in this run. Raw measurements, including all mismatches, are retained. These measurements do not certify Blackwell speed or correctness.

| Case | Topology same | Reference metrics same | Bytes same | Max branch delta | Max shared support delta | Runtime ratio |
| --- | --- | --- | --- | ---: | ---: | ---: |
| gene/n256/000 | True | True | False | 2.041e-08 | 0.0005 | 0.9747 |
| gene/n256/001 | True | True | False | 3.651e-08 | 0.0003 | 0.9082 |
| gene/n512/000 | True | True | False | 3.986e-08 | 0.0012 | 1.0532 |
| gene/n512/001 | True | True | False | 3.34355e-05 | 0.011 | 1.0399 |
| gene/n1024/000 | True | True | False | 2.52233e-07 | 0.0017 | 1.0431 |
| gene/n1024/001 | True | True | False | 1.057e-08 | 0.0007 | 0.9210 |
| gene/n2048/000 | True | True | False | 3.52035e-06 | 0.0045 | 0.9949 |
| gene/n2048/001 | True | True | False | 6.58694e-07 | 0.0176 | 1.0174 |
| gene/n4096/000 | True | True | False | 1.60987e-05 | 0.0109 | 1.0066 |
| gene/n4096/001 | True | True | False | 7.10312e-08 | 0.0007 | 0.9336 |
| species/small/n32/ILS/0/observed_loci | True | True | True | 0 | 0 | 1.0135 |
| species/small/n32/ILS/0/msa_only | True | True | False | 0 | 0.000190021 | 1.0400 |
| species/small/n32/complex/0/observed_loci | True | True | True | 0 | 0 | 1.0347 |
| species/small/n32/complex/0/msa_only | True | True | False | 0 | 3.52469e-05 | 1.0305 |
| species/small/n64/ILS/0/observed_loci | True | True | True | 0 | 0 | 1.0009 |
| species/small/n64/ILS/0/msa_only | True | True | False | 0 | 0.000155276 | 0.9918 |
| species/small/n64/complex/0/observed_loci | True | True | True | 0 | 0 | 1.0074 |
| species/small/n64/complex/0/msa_only | True | True | False | 0 | 0.000396383 | 0.9707 |
| species/small/n128/ILS/0/observed_loci | True | True | True | 0 | 0 | 1.0077 |
| species/small/n128/ILS/0/msa_only | True | True | False | 0 | 0.000315992 | 1.0155 |
| species/small/n128/complex/0/observed_loci | True | True | True | 0 | 0 | 1.0145 |
| species/small/n128/complex/0/msa_only | True | True | False | 0 | 0.000109019 | 0.9804 |
| species/small/n192/ILS/0/observed_loci | True | True | True | 0 | 0 | 0.9824 |
| species/small/n192/ILS/0/msa_only | True | True | False | 0 | 0.000167429 | 0.9782 |
| species/small/n192/complex/0/observed_loci | True | True | True | 0 | 0 | 1.0237 |
| species/small/n192/complex/0/msa_only | True | True | False | 0 | 0.000128519 | 1.0264 |
| species/small/n256/ILS/0/observed_loci | True | True | True | 0 | 0 | 0.9825 |
| species/small/n256/ILS/0/msa_only | True | True | False | 0 | 0.000419129 | 0.9804 |
| species/small/n256/complex/0/observed_loci | True | True | True | 0 | 0 | 1.0154 |
| species/small/n256/complex/0/msa_only | True | True | False | 0 | 0.000181796 | 1.0019 |
| species/large/n512/ILS/0/observed_loci | True | True | True | 0 | 0 | 1.0068 |
| species/large/n512/ILS/0/msa_only | True | True | False | 0 | 6.08461e-05 | 0.9947 |
| species/large/n512/complex/0/observed_loci | True | True | True | 0 | 0 | 1.0113 |
| species/large/n512/complex/0/msa_only | True | True | False | 0 | 0.000304605 | 1.0101 |
| species/large/n1024/ILS/0/observed_loci | True | True | True | 0 | 0 | 0.9959 |
| species/large/n1024/ILS/0/msa_only | True | True | False | 0 | 7.08547e-05 | 0.9971 |
| species/large/n1024/complex/0/observed_loci | True | True | True | 0 | 0 | 0.9921 |
| species/large/n1024/complex/0/msa_only | True | True | False | 0 | 0.000310306 | 0.9413 |
| empirical-concatenated/Bacteria | True | True | False | 0 | 1.2728e-05 | 1.0025 |
| empirical-concatenated/fish | True | True | False | 0 | 1.50538e-05 | 1.0182 |
| empirical-concatenated/mammal2 | True | True | False | 0 | 4.16161e-06 | 1.0061 |
| empirical-concatenated/plant1 | True | True | False | 0 | 1.3594e-05 | 0.9802 |
| empirical-concatenated/mammal82 | True | True | False | 0 | 2.32882e-05 | 1.0064 |
| empirical-concatenated/plant | True | True | False | 0 | 0.000714006 | 1.0061 |
| empirical-concatenated/ruminant44 | True | True | False | 0 | 9.24e-06 | 0.9067 |
| empirical-concatenated/Insecta | True | True | False | 0 | 0.000734673 | 1.0498 |
| empirical-concatenated/mammal241 | True | True | False | 0 | 0.000107924 | 0.9482 |
| empirical-concatenated/butterfly | True | True | False | 0 | 2.72194e-05 | 1.0364 |

## Compilation and remaining boundary

The four representative active likelihood/count/species Triton kernels, plus the adapted downward-dispatch kernel in fp32/fp64 with int32/int64 indices, compile for SM100/103/120/121: 32 successful variant/architecture combinations. The original 16 PTX hashes remain identical when adding the dispatch checks. This is compilation coverage, not exhaustive kernel coverage or execution on those GPUs. The actual candidate execution device is L40S. The native AP-NJ backend carries forward PTX70 and the rebuilt Torch pattern backend carries PTX90. PyTorch compiled architecture flags are recorded in the JSON. [NVIDIA explains PTX forward compatibility and the required execution check](https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-compatibility-guide/index.html).

A modern artifact cannot be advertised as preserving the stable output or as validated on every Linux/NVIDIA machine from these diagnostics. Differences and missing hardware evidence remain explicit. The supported stable artifact continues to require Linux x86_64 and CC7.x–9.x; Blackwell/ARM are not admitted.
