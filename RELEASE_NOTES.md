ConcordTree 1.0.2 now has a Linux x86_64 GPU distribution with the Web interface as its default entry point. The download includes Python, scientific/Web dependencies and model weights. No Conda setup or model download is needed.

Install and start the Web interface:

```bash
curl -fL https://github.com/PullMyBoots/ConcordTree/releases/download/v1.0.2-linux-r1/install.sh -o install-concordtree.sh && bash install-concordtree.sh
```

Open http://127.0.0.1:8765. The installer automatically selects portable when glibc >=2.34 and a working GCC/Clang compiler with C headers are available; otherwise it uses an existing accessible Docker installation. It downloads and verifies five parts, installs atomically, and starts Web. Docker includes the compiler prerequisites and requires Docker plus NVIDIA Container Toolkit on the host. Host CUDA Toolkit is not required. The installer does not change host packages.

Requires Linux x86_64, glibc >=2.34, NVIDIA driver >=525.60.13, compute capability 7.x–9.x and sufficient RAM/VRAM for the input. L40S and RTX 4090 were tested; other listed architectures have representative compilation coverage. Blackwell, pre-Volta, ARM and CPU-only inference are not supported by this stable artifact. See VALIDATION.md for the complete regression and timing results, including cross-device floating-point differences and the rejected modern-runtime candidate.

For offline portable installation, download install.sh, SHA256SUMS and all five `concordtree-1.0.2-linux-x86_64-cuda121.tar.gz.part-*` files into one directory, then run `bash install.sh --from .`.

For Docker, use `bash install-concordtree.sh --runtime docker`. Offline, download install.sh, docker-web.sh, DOCKER_SHA256SUMS and all five Docker parts into one directory, then run `bash install.sh --runtime docker --from .`. The installer verifies the parts and imports the image directly. If the exact image is already installed, it installs the checked launcher without downloading the image again. Docker results persist under `~/.local/share/concordtree/jobs`.

Use `--no-start` to install without launching, `--prefix DIRECTORY` for an installation destination, and `-- --gpu 1 --port 8876 --cpu-threads 16` to select Web runtime options. Both portable and Docker start Web by default. Manual offline Docker import remains available with `sha256sum -c DOCKER_SHA256SUMS`, `cat concordtree-1.0.2-docker.tar.gz.part-* | gzip -dc | docker load` and `bash docker-web.sh`.

`regression-trials.json` includes the original and follow-up per-trial timings, input hashes, RF measurements and output hashes. `MODERN_VALIDATION.md` and `modern-regression-results.json` describe the separate 48-case modern-runtime diagnostic: topology/RF matched throughout, but numerical output differences remain and the candidate is not admitted. These additional diagnostics do not expand stable hardware support.
