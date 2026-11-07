# ConcordTree 1.0.2 Linux distribution validation

## Artifacts

Portable archive: `concordtree-1.0.2-linux-x86_64-cuda121.tar.gz` (4,732,982,106 bytes).
SHA256: `97cce5a23f61e94f85a5cb46088a6b8862b284b35721edd2908324eda024f75c`.

The portable archive and container include the scientific and Web runtimes, weights and native libraries. The unified installer automatically selects portable when glibc/compiler prerequisites are met, otherwise an existing accessible Docker installation. It verifies five ordered parts before extraction/import. Portable extraction additionally verifies 1121 static files; Docker import checks the exact image ID. Outputs and caches are excluded from release archives.

## Matched regression

Frozen V1.0.2 vs final CUDA 12.1 distribution, same L40S GPU, CPU affinity 0–15, 16 threads. 48 inputs: 10 gene simulations (256–4096 taxa), 28 species configurations (32–1024 taxa, observed loci and MSA-only), and all 10 retained empirical inputs. Each artifact receives a complete unmeasured warm-up per case. Simulations have three alternating paired measurements; empirical cases have one pair after warm-up. Inputs and references were hashed and read without modification.

**48/48 initial and final Newick sets match byte for byte, including branches and support labels; all reference RF values match. Repeated outputs are deterministic for these runs.** Empirical reference trees are comparison references, not established biological truth.

Primary median of case runtime ratios: **1.0057**; ratio of summed per-case medians: **1.0045** (package / baseline; lower is faster).

All simulated cases differing by more than 5% in either direction receive a separately reported nine-repeat follow-up. Primary observations remain in the report. CPU work for packaging and compiler probes was assigned separate affinity sets; some early primary measurements overlapped environment/build preparation. The follow-up runs after the main build work. A 5% threshold is an investigation rule, not proof of universal performance equivalence.

The nine-repeat follow-up retained an 8.17% (12.8 ms) delay on one 0.16-second case. A fresh 49-repeat run measured 6.12% (9.0 ms). Profiling showed most difference outside construction and scoring, in CPU input processing; unchanged Python regex/loop work also differed. Aligning both main threads to the same starting CPU before timing, then restoring all 16 allowed CPUs for parallel workers, reduced the 49-repeat delay to 1.73% (2.5 ms). This supports sensitivity to CPU scheduling/placement, not a guarantee that every future run will stay below 5%. All seven flagged cases then receive 17 paired repeats with this additional control. Earlier measurements are retained.

| Case | Primary ratio | Nine repeats | Aligned 17 repeats |
| --- | ---: | ---: | ---: |
| gene/n256/000 | 1.2177 | 0.9942 | 0.9887 |
| gene/n256/001 | 1.0101 | — | — |
| gene/n512/000 | 1.1262 | 0.9537 | 0.9906 |
| gene/n512/001 | 0.9861 | — | — |
| gene/n1024/000 | 1.0108 | — | — |
| gene/n1024/001 | 1.0394 | — | — |
| gene/n2048/000 | 1.0255 | — | — |
| gene/n2048/001 | 0.9943 | — | — |
| gene/n4096/000 | 1.0416 | — | — |
| gene/n4096/001 | 0.9705 | — | — |
| species/small/n32/ILS/0/observed_loci | 1.0113 | — | — |
| species/small/n32/ILS/0/msa_only | 1.1060 | 0.9305 | 0.9996 |
| species/small/n32/complex/0/observed_loci | 1.0810 | 1.0817 | 1.0087 |
| species/small/n32/complex/0/msa_only | 1.0630 | 0.9524 | 1.0104 |
| species/small/n64/ILS/0/observed_loci | 0.9823 | — | — |
| species/small/n64/ILS/0/msa_only | 0.9575 | — | — |
| species/small/n64/complex/0/observed_loci | 1.0199 | — | — |
| species/small/n64/complex/0/msa_only | 1.1471 | 0.9633 | 1.0018 |
| species/small/n128/ILS/0/observed_loci | 0.9746 | — | — |
| species/small/n128/ILS/0/msa_only | 0.9198 | 0.9984 | 0.9889 |
| species/small/n128/complex/0/observed_loci | 1.0198 | — | — |
| species/small/n128/complex/0/msa_only | 0.9979 | — | — |
| species/small/n192/ILS/0/observed_loci | 1.0178 | — | — |
| species/small/n192/ILS/0/msa_only | 1.0238 | — | — |
| species/small/n192/complex/0/observed_loci | 0.9869 | — | — |
| species/small/n192/complex/0/msa_only | 0.9960 | — | — |
| species/small/n256/ILS/0/observed_loci | 0.9989 | — | — |
| species/small/n256/ILS/0/msa_only | 0.9886 | — | — |
| species/small/n256/complex/0/observed_loci | 1.0215 | — | — |
| species/small/n256/complex/0/msa_only | 1.0006 | — | — |
| species/large/n512/ILS/0/observed_loci | 1.0027 | — | — |
| species/large/n512/ILS/0/msa_only | 0.9826 | — | — |
| species/large/n512/complex/0/observed_loci | 1.0206 | — | — |
| species/large/n512/complex/0/msa_only | 1.0444 | — | — |
| species/large/n1024/ILS/0/observed_loci | 1.0110 | — | — |
| species/large/n1024/ILS/0/msa_only | 0.9874 | — | — |
| species/large/n1024/complex/0/observed_loci | 0.9907 | — | — |
| species/large/n1024/complex/0/msa_only | 0.9834 | — | — |
| empirical-concatenated/Bacteria | 1.0168 | — | — |
| empirical-concatenated/fish | 1.0062 | — | — |
| empirical-concatenated/mammal2 | 0.9940 | — | — |
| empirical-concatenated/plant1 | 0.9397 | — | — |
| empirical-concatenated/mammal82 | 0.9986 | — | — |
| empirical-concatenated/plant | 1.0053 | — | — |
| empirical-concatenated/ruminant44 | 1.0141 | — | — |
| empirical-concatenated/Insecta | 0.9923 | — | — |
| empirical-concatenated/mammal241 | 0.9471 | — | — |
| empirical-concatenated/butterfly | 1.0070 | — | — |

Follow-up investigation gate (no case >5% slower): **passed**.

Warm timings use persistent scientific processes, exclude process/model startup and initial JIT compilation, and measure file-to-tree inference. Web tests exercise separate scientific processes per job, so their turnaround includes upload validation, queueing, startup, model loading and kernel compilation. First-run latency must not be described as warm inference time. No statement here guarantees equal wall time on different GPUs.

## Relocation, UI and hardware

- Unified installer: automatic portable selection and reuse passed. A fresh-cache localhost HTTP installation downloaded and verified all five portable parts, extracted atomically and verified 1121 static files. A clean Ubuntu 22.04 CUDA base container without a C compiler selected Docker successfully using ordinary UID/GID and an existing host Docker daemon. Explicit Docker installation verified all five parts and streamed image import using a fixed installer snapshot. A fresh-cache localhost HTTP Docker installation downloaded and verified the launcher, reusing the exact image without downloading its parts again.
- Latest Docker launcher: GPU index 1, four CPU threads, ordinary UID/GID, persistent state, custom loopback port 8882 mapped identically inside/outside. Real Gene and Species upload/inference/download succeeded and both trees matched the final container outputs byte for byte; invalid upload returned 422. Public GitHub download URLs have not yet been exercised. Earlier shell-edit-during-test failure is retained as an invalidated attempt; the fixed-snapshot rerun passed.
- Complete final archive installed from verified parts into a new local directory; all static hashes verified. A directory containing spaces also completed actual Gene and Species Web inference after startup-path adaptation.
- Final container: Ubuntu 22.04 userspace, NVIDIA L40S, ordinary UID/GID, persistent results/cache mounts; actual Gene/Species upload, inference, download and invalid input rejection passed. Image save/load passed.
- Workstation: Ubuntu 22.04 WSL, RTX 4090, driver 610.88. Archive copied to `D:\serve-backup\ConcordTree\distribution-test`, SHA256 verified, extracted independently. Environment doctor, two Web examples, and two existing regression inputs completed.
- Across L40S and RTX 4090 the two retained regression inputs have identical topology (tree-to-tree RF=0). Species output is byte identical; Gene branch lengths differ by at most 1.425e-8. Cross-device byte identity is not claimed.
- Browser checks cover real upload, both task types, preview, download, refresh recovery and mobile rendering. The original Web test suite has 26 passing tests.
- SM70/75/80/86/89/90 native fatbinary coverage and four representative active Triton kernels compile on every listed architecture. This is build evidence, not full hardware correctness/performance certification. L40S and RTX 4090 are the actual tested devices.

## Support boundary

Portable: Linux x86_64, glibc >=2.34, driver >=525.60.13, compute capability 7.x–9.x; GCC/Clang and C headers are needed for first-run Triton helpers. Docker includes compiler prerequisites and requires Docker plus NVIDIA Container Toolkit on the host. No host CUDA Toolkit or Conda setup is needed. Available RAM/VRAM limits input sizes. Web uploads are limited to 100 MiB; larger alignments use CLI. ARM, CPU-only, pre-Volta and Blackwell are not admitted by this stable runtime.

The experimental Torch 2.11/CUDA 12.8/Triton 3.6 candidate required a type-specialization API adaptation. Its first Gene case retained topology and RF but changed branch values (max 2.041e-8), failing the strict artifact gate. It is not shipped as stable Blackwell support; no Blackwell hardware was available.

## Provenance

Scientific source and checkpoints remain frozen. Changes are limited to startup/configuration, Web GPU/CPU/bind selection, integrity/documentation, identical-source sequence-parser rebuild for glibc 2.34, and AP-NJ fatbinary architecture coverage. The original and expanded AP-NJ SM89 builds have 5960 identical SASS instructions. Scientific versions remain Torch 2.5.1+cu121, Triton 3.1.0, NumPy 2.1.2 and SciPy 1.15.3. Full hashes and raw per-trial records are retained in the Main-owned distribution workspace.

Official compatibility references: [NVIDIA Blackwell guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html), [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), [PyTorch version matrix](https://pytorch.org/get-started/previous-versions/).
