# ConcordTree 1.0.2 — Linux GPU distribution

Start the Web interface:

```bash
./concordtree
```

Open **http://127.0.0.1:8765**. Upload an aligned DNA file, choose Gene Tree or Species Tree, and download the result. Python, scientific dependencies, Web dependencies and model weights are included. No Conda setup or model download is needed. First inference compiles GPU kernels; subsequent jobs reuse the local cache.

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

The default server listens on loopback. For a remote server, use SSH forwarding, e.g. `ssh -L 8765:127.0.0.1:8765 user@server`, then open the same URL locally. The UI has a 100 MiB upload limit; use the CLI for larger alignments. The UI accepts PHYLIP and normalizes it; the CLI also accepts equal-length FASTA and `.zst` inputs.

## Algorithm and provenance

The scientific algorithm, checkpoints, models, stopping conditions and support definitions match the frozen V1.0.2 release. Packaging changes cover startup paths, Web CPU/GPU selection, GPU architecture coverage, and rebuilding the identical sequence-parser source for older glibc. PyTorch 2.5.1+cu121, Triton 3.1.0, NumPy 2.1.2 and SciPy 1.15.3 are retained.

`MANIFEST.json` and `verify_release.py` cover code, model weights and static assets. Archive SHA256 covers the complete runtime as well. `./.web-python/bin/python -I -B verify_release.py` verifies the extracted static files. Package notices remain in dependency metadata and `THIRD_PARTY_LICENSES/`; the project license is in `LICENSE`. The retained Conda metadata describes the original environment and can differ from subsequently installed pip packages; it is not an environment lockfile.

See [ORIGINAL_RELEASE.md](ORIGINAL_RELEASE.md), [METHOD_CONTRACT.json](METHOD_CONTRACT.json), [NATIVE.md](NATIVE.md) and [BRANCH_SUPPORT.md](BRANCH_SUPPORT.md) for scientific settings and limitations. Historical evidence in `evidence/` describes earlier releases; the distribution validation report identifies the actual packaging regression results. Branch support is a local evidence indicator, not a calibrated probability or bootstrap value.
