# ConcordTree 1.0.2 Linux distribution

The default application is the Web interface. The portable artifact includes Python, CUDA user libraries, scientific dependencies and model weights.

This branch contains the frozen 1.0.2 runtime and Linux distribution tooling. See VALIDATION.md for the exact regression results, compatibility matrix and limitations. Binary installation requires the matching published Release assets.

## Install local release assets

```bash
bash install.sh --from /directory/containing/release-assets
```

For the matching published release:

```bash
curl -fL https://github.com/PullMyBoots/ConcordTree/releases/download/v1.0.2-linux-r1/install.sh -o install-concordtree.sh && bash install-concordtree.sh
```

See [app/README.md](app/README.md) for GPU/driver/glibc requirements, Web/CLI usage, output persistence and scientific limitations. Building the binary artifacts requires the frozen scientific environment; `scripts/` records the build and regression procedures.
