#!/usr/bin/env bash
set -euo pipefail
release_root="$(cd "$(/usr/bin/dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
web_python="${release_root}/.web-python/bin/python"
if [[ ! -x "${web_python}" ]]; then
  echo "Missing local web environment: ${web_python}" >&2
  exit 2
fi
unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${release_root}/.web-python/lib:${release_root}/.conda/env/lib"
export PATH="${release_root}/.conda/env/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec "${web_python}" -I -B "${release_root}/scripts/serve.py" "$@"
