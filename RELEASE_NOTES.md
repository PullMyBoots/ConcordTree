ConcordTree 1.0.2 now has a Linux x86_64 GPU distribution with the Web interface as its default entry point. The download includes Python, scientific/Web dependencies and model weights. No Conda setup or model download is needed.

Install and start the Web interface:

```bash
curl -fL https://github.com/PullMyBoots/ConcordTree/releases/download/v1.0.2-linux-r1/install.sh -o install-concordtree.sh && bash install-concordtree.sh
```

Open http://127.0.0.1:8765. The installer downloads five checked parts, installs atomically, verifies the extracted static files, and starts the Web UI. GCC/Clang and C headers are prerequisites for the portable runtime; the separate Docker image includes them. Host CUDA Toolkit is not required.

Requires Linux x86_64, glibc >=2.34, NVIDIA driver >=525.60.13, compute capability 7.x–9.x and sufficient RAM/VRAM for the input. L40S and RTX 4090 were tested; other listed architectures have representative compilation coverage. Blackwell, pre-Volta, ARM and CPU-only inference are not supported by this stable artifact. See VALIDATION.md for the complete regression and timing results, including cross-device floating-point differences and the rejected modern-runtime candidate.

For offline portable installation, download install.sh, SHA256SUMS and all five `concordtree-1.0.2-linux-x86_64-cuda121.tar.gz.part-*` files into one directory, then run `bash install.sh --from .`.

For Docker, download docker-web.sh, DOCKER_SHA256SUMS, the Docker archive checksum and all five Docker parts. Verify them with `sha256sum -c DOCKER_SHA256SUMS`, concatenate the ordered parts into `concordtree-1.0.2-docker.tar.gz`, then run `bash docker-web.sh`. Docker and NVIDIA Container Toolkit must already work on the host. Results persist under `~/.local/share/concordtree/jobs`.
