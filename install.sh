#!/usr/bin/env bash
# Download/install the frozen Linux runtime, then open the Web UI.
set -euo pipefail
ct_name=concordtree-1.0.2-linux-x86_64-cuda121
ct_prefix="${CONCORDTREE_INSTALL_DIR:-$HOME/.local/share/concordtree/1.0.2-linux-r1}"
ct_base="${CONCORDTREE_DOWNLOAD_BASE:-https://github.com/PullMyBoots/ConcordTree/releases/download/v1.0.2-linux-r1}"
ct_source=''
ct_start=1
while (($#)); do
    case "$1" in
        --from) ct_source="$2"; shift 2;;
        --base-url) ct_base="$2"; shift 2;;
        --prefix) ct_prefix="$2"; shift 2;;
        --no-start) ct_start=0; shift;;
        --help) printf '%s\n' 'install.sh [--base-url HTTPS_URL | --from LOCAL_ASSET_DIRECTORY] [--prefix DIRECTORY] [--no-start] [-- WEB_OPTIONS]'; exit 0;;
        --) shift; break;;
        *) echo "Unknown installer argument: $1" >&2; exit 2;;
    esac
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'This artifact requires Linux x86_64.' >&2; exit 2; }
for ct_tool in tar gzip sha256sum awk; do
    command -v "$ct_tool" >/dev/null || { echo "Missing prerequisite: $ct_tool" >&2; exit 2; }
done
if [[ ! -x "$ct_prefix/concordtree" ]]; then
    [[ ! -e "$ct_prefix" ]] || { echo "Destination already exists: $ct_prefix. Choose an empty --prefix." >&2; exit 2; }
    [[ -n "$ct_source" || "$ct_base" == https://* || "$ct_base" == http://127.0.0.1:* ]] || {
        echo 'Specify the published HTTPS asset directory with --base-url, or local assets with --from.' >&2; exit 2;
    }
    ct_parent="$(dirname -- "$ct_prefix")"
    mkdir -p -- "$ct_parent"
    ct_cache="${XDG_CACHE_HOME:-$HOME/.cache}/concordtree-downloads/1.0.2-linux-r1"
    mkdir -p -- "$ct_cache"
    ct_fetch() {
        local ct_file="$1"
        if [[ -n "$ct_source" ]]; then
            cp -- "$ct_source/$ct_file" "$ct_cache/$ct_file"
        else
            command -v curl >/dev/null || { echo 'curl is required for downloads.' >&2; exit 2; }
            curl --fail --location --retry 4 --connect-timeout 20 --continue-at - \
                --output "$ct_cache/$ct_file.partial" "${ct_base%/}/$ct_file"
            mv -- "$ct_cache/$ct_file.partial" "$ct_cache/$ct_file"
        fi
    }
    ct_fetch SHA256SUMS
    ct_parts=()
    while read -r ct_hash ct_file ct_extra; do
        [[ "$ct_hash" =~ ^[0-9a-f]{64}$ && "$ct_file" =~ ^concordtree-1\.0\.2-linux-x86_64-cuda121\.tar\.gz\.part-[0-9]{3}$ && -z "$ct_extra" ]] || {
            echo 'Invalid asset checksum manifest.' >&2; exit 2;
        }
        ct_parts+=("$ct_cache/$ct_file")
        if ! (cd -- "$ct_cache" && printf '%s  %s\n' "$ct_hash" "$ct_file" | sha256sum --check --status); then
            ct_fetch "$ct_file"
            (cd -- "$ct_cache" && printf '%s  %s\n' "$ct_hash" "$ct_file" | sha256sum --check) || exit 1
        fi
    done < "$ct_cache/SHA256SUMS"
    ((${#ct_parts[@]} > 0)) || { echo 'Empty asset manifest.' >&2; exit 2; }
    ct_stage="$(mktemp -d "$ct_parent/.concordtree-install.XXXXXXXX")"
    trap 'rm -rf -- "$ct_stage"' EXIT
    cat -- "${ct_parts[@]}" | tar -xz --no-same-owner --warning=no-timestamp -C "$ct_stage"
    "$ct_stage/$ct_name/.web-python/bin/python" -I -B "$ct_stage/$ct_name/verify_release.py"
    mv -T -- "$ct_stage/$ct_name" "$ct_prefix"
    rmdir -- "$ct_stage"
    trap - EXIT
fi
printf 'Installed: %s\nStart again: %s/concordtree\n' "$ct_prefix" "$ct_prefix"
if ((ct_start)); then
    exec "$ct_prefix/concordtree" "$@"
fi
