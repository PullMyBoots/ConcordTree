#!/usr/bin/env bash
set -euo pipefail
release_root="$(cd "$(/usr/bin/dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
scientific_python="${release_root}/.conda/env/bin/python"
packages="${release_root}/.conda/env/lib/python3.10/site-packages"
if [[ ! -x "${scientific_python}" ]]; then
  echo "Missing local environment: ${scientific_python}" >&2
  exit 2
fi
unset PYTHONPATH PYTHONHOME LD_PRELOAD CONCORDTREE_EXPERIMENTAL_PATTERN_BACKEND
export PATH="${release_root}/.conda/env/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${packages}/torch/lib:${release_root}/.conda/env/lib"
for vendor_lib in "${packages}"/nvidia/*/lib; do
  if [[ -d "${vendor_lib}" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${vendor_lib}"
  fi
done
export TRITON_CACHE_DIR="${release_root}/.runtime/cache/triton"
export TORCH_EXTENSIONS_DIR="${release_root}/.runtime/cache/extensions"
export CUDA_CACHE_PATH="${release_root}/.runtime/cache/cuda"
export TMPDIR="${CONCORDTREE_SAFE_TMPDIR:-${release_root}/.runtime/cache/tmp}"
export XDG_CACHE_HOME="${release_root}/.runtime/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="${release_root}/.runtime/cache/inductor"
export TORCH_HOME="${release_root}/.runtime/cache/torch"
mkdir -p "${TMPDIR}" "${CUDA_CACHE_PATH}"
if [[ "${1:-}" == doctor ]]; then
  shift
  exec "${scientific_python}" -I -B "${release_root}/scripts/doctor.py" "$@"
fi
exec "${scientific_python}" -I -B "${release_root}/run.py" "$@"
