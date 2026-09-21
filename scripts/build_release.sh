#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${release_root}/.conda/env/bin/python"
version="0.1.2"
wheel="concordtree-${version}-cp310-cp310-linux_x86_64.whl"
sdist="concordtree-${version}.tar.gz"
checksums="SHA256SUMS"

if [[ ! -x "${python_bin}" ]]; then
  echo "missing dedicated environment; run scripts/create_environment.sh" >&2
  exit 2
fi

cd "${release_root}"
rm -f \
  "${release_root}/dist/${wheel}" \
  "${release_root}/dist/${sdist}" \
  "${release_root}/dist/${checksums}" \
  "${release_root}/dist/concordtree-0.1.2.dev0-cp310-cp310-linux_x86_64.whl" \
  "${release_root}/dist/concordtree-0.1.2.dev0.tar.gz" \
  "${release_root}/dist/SHA256SUMS.dev0"
PYTHONPATH="${release_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${python_bin}" -m unittest discover -s tests -v
"${python_bin}" setup.py clean --all
"${python_bin}" setup.py sdist
"${python_bin}" -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
if tar -tzf "${release_root}/dist/${sdist}" | \
  grep -Eq '(^|/)(\.conda|build|dist|__pycache__)(/|$)|\.py[co]$'; then
  echo "source distribution contains generated local files" >&2
  exit 3
fi
(
  cd dist
  sha256sum "${wheel}" "${sdist}" > "${checksums}"
  sha256sum --check "${checksums}"
)
