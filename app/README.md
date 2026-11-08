# ConcordTree 1.0.2 — Linux GPU distribution

Run this application on your own Linux / WSL NVIDIA computer:

```bash
./concordtree
```

The application opens **http://127.0.0.1:8765** in your desktop browser. All inference, input files, results and logs stay on your own computer. Upload an aligned DNA file, choose Gene Tree or Species Tree, and download the result. Python, scientific dependencies, Web dependencies and model weights are included. No Conda setup or model download is needed. First inference compiles GPU kernels; subsequent jobs reuse the local cache.

```bash
./concordtree doctor --output runs/doctor.json
./concordtree --gpu 1 --cpu-threads 16 --port 8876
./concordtree cli --task gene --input /path/alignment.phy --out runs/my-job
./concordtree cli --task species --input /path/supermatrix.phy --loci /path/loci.jsonl --out runs/my-species-job
```

Keep the entire extracted directory, including hidden files. Jobs/results are in `runs/`; generated caches are in `.runtime/`. To move the installation, stop it and move the whole directory. Do not delete `runs/` when upgrading. Outputs must use a new directory.

## Compatibility

This artifact requires **Linux x86_64, glibc >=2.34, NVIDIA driver >=525.60.13, a CUDA-capable GPU with compute capability 7.x–9.x, and GCC or Clang with C development headers**. Ubuntu 22.04/24.04 are representative systems. Install compiler prerequisites on Ubuntu/Debian with `sudo apt install gcc libc6-dev`. The separate container includes them and can run on older host userspaces with Docker and NVIDIA Container Toolkit. A host CUDA Toolkit is not required.

L40S and RTX 4090 are the available validation devices; see the accompanying distribution validation report for completed checks. Other Volta/Turing/Ampere/Ada/Hopper devices have build coverage, not equivalent hardware validation. This runtime does not cover Blackwell, pre-Volta GPUs, ARM or CPU-only inference. Different GPUs have different speed and memory limits. Input size must fit RAM and GPU memory.

The local interface listens only on loopback. Uploads, server-file imports and normalized matrices have no application-level size cap; available memory and disk determine practical capacity. The UI accepts PHYLIP and normalizes it; the CLI also accepts equal-length FASTA and `.zst` inputs.

## Algorithm and provenance

The active pipeline is full-site bounded AP-NJ (K2P-gamma), task-specific MLP-NNI, then gene likelihood or species statistical refinement. Checkpoints, stopping conditions and support definitions are retained from the current V1.0.2 working release. Historical validation reports identify their own configuration snapshots. Packaging changes cover startup paths, Web CPU/GPU selection, GPU architecture coverage, and rebuilding the identical sequence-parser source for older glibc. PyTorch 2.5.1+cu121, Triton 3.1.0, NumPy 2.1.2 and SciPy 1.15.3 are retained.

`MANIFEST.json` and `verify_release.py` cover code, model weights and static assets. Archive SHA256 covers the complete runtime as well. `./.web-python/bin/python -I -B verify_release.py` verifies the extracted static files. Package notices remain in dependency metadata and `THIRD_PARTY_LICENSES/`; the project license is in `LICENSE`. The retained Conda metadata describes the original environment and can differ from subsequently installed pip packages; it is not an environment lockfile.

See [ORIGINAL_RELEASE.md](ORIGINAL_RELEASE.md), [METHOD_CONTRACT.json](METHOD_CONTRACT.json), [NATIVE.md](NATIVE.md) and [BRANCH_SUPPORT.md](BRANCH_SUPPORT.md) for scientific settings and limitations. Historical evidence in `evidence/` describes earlier releases; the distribution validation report identifies the actual packaging regression results. Branch support is a local evidence indicator, not a calibrated probability or bootstrap value.


The local Web interface includes a collapsed **Host load** monitor (CPU, RAM, GPU utilization and VRAM), a seed-to-tree stage animation and a live runtime log. Expand Host load for all visible NVIDIA devices. Progress represents observed inference stages; the tree is decorative. The unused DIPPER constructor and historical AP-NJ fallback are excluded from this distribution; shared kernels and third-party notices required by the active pipeline are retained.

Pipeline settings are available through the gear button: reorder and enable refinement modules, edit advanced parameters, and restore defaults. Evolutionary models are under Maximum likelihood → Advanced parameters. Pause at complete-tree boundaries to download the current tree; Resume continues, while Cancel removes job files. See [PIPELINE_SETTINGS.md](PIPELINE_SETTINGS.md). Seasons now cycles a fixed landscape and workspace colors every ten minutes; Nature photos retain their original colors.
