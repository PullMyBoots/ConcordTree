#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_root="${release_root}/.conda/env"
source_environment="${CONCORDTREE_SOURCE_ENV:-}"
conda_bin="${CONDA_EXE:-$(command -v conda || true)}"

if [[ -z "${conda_bin}" || ! -x "${conda_bin}" ]]; then
  echo "conda executable not found; set CONDA_EXE=/absolute/path/to/conda" >&2
  exit 2
fi

if [[ ! -x "${environment_root}/bin/python" ]]; then
  if [[ -n "${source_environment}" ]]; then
    "${conda_bin}" create --yes --prefix "${environment_root}" --clone "${source_environment}"
  else
    "${conda_bin}" env create --yes --prefix "${environment_root}" \
      --file "${release_root}/environment.yml"
  fi
fi

"${environment_root}/bin/python" -m pip install \
  --no-deps --no-build-isolation --force-reinstall "${release_root}"
"${environment_root}/bin/concordtree" --version
