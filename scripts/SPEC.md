# ConcordTree Linux distribution

Human request (2026-11-08 Asia/Shanghai): package the current algorithm for Linux + NVIDIA devices, one download/start entry, Web by default, and retest with existing data to preserve speed and accuracy.

Source: release/V1.0.2. This Main-owned workspace copies the frozen release for baseline and distribution development; the source project and its data remain unchanged.

Acceptance:
- All weights and algorithm source stay fixed. Rebuilding native extensions and adapting launch/runtime paths are allowed, with hashes and regression evidence.
- A packaged installation works outside the original project directory, without reading original server paths. Web is the default launch, with persistent jobs and caches.
- GPU/driver/CPU architecture checks report supported and unverified cases honestly. Hardware below the algorithm stack requirements cannot be promised equivalent speed.
- Baseline vs packaged: matched records, models, supports and stops, same GPU/CPU allocation, paired repeated measurements. Compare full-tree topology, reference RF, branch lengths/support and initialization/search outputs; report warm inference, cold startup and Web turnaround separately. A 5% median warm-inference change is a provisional investigation threshold, not proof of universal equivalence.
- Use existing inputs read-only; write every derived file under this workspace. Dataset comparisons are packaging regression, not new scientific generalization evidence.
- Public download URLs and externally published assets must be real and verified before claiming public availability.
