# ConcordTree

## Current application source

The `app/` directory contains the latest local Linux/WSL NVIDIA application source, updated on 2026-11-15. The Web interface includes modular Gene Tree and Species Tree pipelines, per-job CPU/GPU selection, pause/download/resume, continuous GPU monitoring, stage-specific progress tables, and seasonal/geometry/photo backgrounds. Unused experimental runtime branches have been removed and the technical report has been updated.

**Technical report:** [Overview and input formats](app/technical-report/index.html) · [Accuracy](app/technical-report/accuracy.html) · [Runtime and resources](app/technical-report/performance.html) · [Method](app/technical-report/method.html).

The [rendered technical report](https://pullmyboots.github.io/ConcordTree/) includes the 4,096-taxon stage analysis and its DIPPER protocol. These links open the report HTML source on GitHub. To read the rendered report, start the local application and open [Technical Report](http://127.0.0.1:8765/report/) (use your chosen port if different), or click **Technical Report** in the application footer. The report includes the current results, figures and configuration/provenance records.

The stage strip highlights the running algorithm. AP-NJ shows elapsed time; NNI tables show the fields relevant to their own criterion. Configure algorithms and advanced parameters with the settings gear. Local inference has no application-imposed upload-size or execution-time limit.

See [pipeline settings](app/PIPELINE_SETTINGS.md), [Web behavior](app/web/README.md), and [current source checks](SOURCE_UPDATE_20261115.md). The source checkout includes model assets and native extensions but excludes the multi-gigabyte scientific/Python runtime, local jobs, caches and research datasets. Cloning the repository alone does not create the packaged runtime.

**Prebuilt archives and containers have not been rebuilt for this source update.** The installation commands and 48-case packaging measurements below describe the earlier frozen distribution; they require matching published assets and do not validate the latest source changes.

## Earlier frozen distribution

Install ConcordTree and start its Web interface:

```bash
curl -fL https://github.com/PullMyBoots/ConcordTree/releases/download/v1.0.2-linux-r1/install.sh -o install-concordtree.sh && bash install-concordtree.sh
```

This command requires the matching published Release assets. Open **http://127.0.0.1:8765**, upload an aligned DNA file, select Gene Tree or Species Tree, and download the result. Python, scientific/Web dependencies and model weights are included. No Conda setup or host CUDA Toolkit is needed. Initial inference compiles GPU kernels; subsequent jobs reuse the cache.

## Runtime selection

The installer defaults to `--runtime auto`. It uses the portable runtime when the host has glibc >=2.34 and a working GCC/Clang compiler with C headers. Otherwise it uses an existing accessible Docker installation. Docker also requires NVIDIA Container Toolkit and a supported host NVIDIA driver. The installer does not install host packages or configure Docker.

To choose explicitly, or select GPU, port and CPU threads:

```bash
bash install-concordtree.sh --runtime portable
bash install-concordtree.sh --runtime docker
bash install-concordtree.sh -- --gpu 1 --port 8876 --cpu-threads 16
```

Both runtimes start Web by default. `--no-start` installs without starting. The installer prints the installed `concordtree` launcher path for subsequent use. The portable default is `~/.local/share/concordtree/1.0.2-linux-r1`; Docker uses the same path with `-docker` appended. Set `--prefix DIRECTORY` for another destination. A destination that already contains a different runtime is rejected.

Portable results and caches live in the installation's `runs/` and `.runtime/` directories. Docker results persist under `~/.local/share/concordtree/jobs` and caches under `~/.local/share/concordtree/cache`; set `CONCORDTREE_STATE` to choose another directory. Keep your results when upgrading. Docker runs with the current user's UID/GID and binds Web to host loopback.

For a remote server, forward the Web port with `ssh -L 8765:127.0.0.1:8765 user@server`, then open the same URL locally. The current source has no application-imposed Web upload-size or task-time limit. See [app/README.md](app/README.md) for the portable CLI and scientific settings.

## Offline installation

Download `install.sh`, `SHA256SUMS` and all five `concordtree-1.0.2-linux-x86_64-cuda121.tar.gz.part-*` files into one directory:

```bash
bash install.sh --runtime portable --from /directory/containing/release-assets
```

For Docker, download `install.sh`, `docker-web.sh`, `DOCKER_SHA256SUMS` and all five `concordtree-1.0.2-docker.tar.gz.part-*` files:

```bash
bash install.sh --runtime docker --from /directory/containing/release-assets
```

The installer verifies all parts, streams extraction or Docker import, checks the installed runtime and installs atomically. If the exact Docker image is already installed, it verifies and installs the launcher without downloading the image again. Online downloads resume partial files and reuse valid cached parts.

Manual Docker import remains available: verify `sha256sum -c DOCKER_SHA256SUMS`, then run `cat concordtree-1.0.2-docker.tar.gz.part-* | gzip -dc | docker load` followed by `bash docker-web.sh`.

## Validated scope

The stable artifact targets **Linux x86_64, NVIDIA driver >=525.60.13, compute capability 7.x–9.x**, and sufficient RAM/VRAM. Portable additionally requires glibc >=2.34 and GCC/Clang with C headers. Docker includes these userspace prerequisites. L40S and RTX 4090 were tested; SM70/75/80/86/89/90 have representative native/Triton compilation coverage. Blackwell, pre-Volta, ARM and CPU-only inference are not admitted by this release.

The frozen scientific algorithm and weights are retained. All 48 paired cases produced byte-identical initial/final trees on the same L40S, including branches/support and reference RF. Aggregate warm inference time differed by +0.45%; timing outliers and follow-up investigations are retained in [VALIDATION.md](VALIDATION.md). Across L40S and RTX 4090, the two tested regression inputs have identical topology/RF, with a Gene branch difference up to 1.425e-8. These observations do not guarantee identical timing or bytes on every GPU.

This repository retains the earlier distribution tooling and evidence alongside the updated `app/` source. Building binary artifacts requires a scientific environment; the existing `scripts/` freeze checks and build records describe the earlier snapshot and have not been requalified for this source update. [RELEASE_NOTES.md](RELEASE_NOTES.md) and [VALIDATION.md](VALIDATION.md) describe the delivery and its evidence.

[regression-trials.json](regression-trials.json) preserves the original/follow-up per-trial timings and output hashes. A separate [modern-runtime diagnostic](MODERN_VALIDATION.md) completed all 48 cases with identical topology/RF and small numerical changes; it has not been admitted and has no Blackwell hardware validation.
