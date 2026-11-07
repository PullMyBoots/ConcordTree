#!/usr/bin/env bash
set -euo pipefail
ct_distribution="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ct_image="${CONCORDTREE_IMAGE:-concordtree:1.0.2-linux-cuda121}"
ct_gpu="${CONCORDTREE_GPU:-0}"
ct_port="${CONCORDTREE_PORT:-8765}"
ct_state="${CONCORDTREE_STATE:-$HOME/.local/share/concordtree}"
[[ "$ct_gpu" =~ ^[0-9]+$ && "$ct_port" =~ ^[0-9]+$ ]] || { echo 'Invalid GPU or port.' >&2; exit 2; }
command -v docker >/dev/null || { echo 'Docker is required for this launcher. Use the portable download or install Docker and NVIDIA Container Toolkit.' >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo 'Cannot access Docker. Start its daemon and grant your user Docker access.' >&2; exit 2; }
if ! docker image inspect "$ct_image" >/dev/null 2>&1; then
    ct_archive="$ct_distribution/concordtree-1.0.2-docker.tar.gz"
    [[ -f "$ct_archive" ]] || { echo "Missing image: $ct_archive" >&2; exit 2; }
    docker load --input "$ct_archive"
fi
mkdir -p -- "$ct_state/jobs" "$ct_state/cache"
printf 'Open http://127.0.0.1:%s\nPersistent results: %s/jobs\n' "$ct_port" "$ct_state"
exec docker run --rm --init --gpus "device=$ct_gpu" \
    --user "$(id -u):$(id -g)" --shm-size=1g \
    -p "127.0.0.1:$ct_port:8765" \
    --mount "type=bind,src=$ct_state/jobs,dst=/opt/concordtree/runs" \
    --mount "type=bind,src=$ct_state/cache,dst=/opt/concordtree/.runtime" \
    -e CONCORDTREE_CPU_THREADS="${CONCORDTREE_CPU_THREADS:-16}" \
    "$ct_image" --no-browser "$@"
