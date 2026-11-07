#!/usr/bin/env bash
# Download a verified runtime and start the Web UI. No host packages are changed.
set -euo pipefail
ct_name=concordtree-1.0.2-linux-x86_64-cuda121
ct_image=concordtree:1.0.2-linux-cuda121
ct_image_id=sha256:09515e53a65eb59518617c6eab9556ccc4f5e625cadfb016bea474b8c20e5686
ct_prefix="${CONCORDTREE_INSTALL_DIR:-$HOME/.local/share/concordtree/1.0.2-linux-r1}"
ct_base="${CONCORDTREE_DOWNLOAD_BASE:-https://github.com/PullMyBoots/ConcordTree/releases/download/v1.0.2-linux-r1}"
ct_source=''
ct_runtime=auto
ct_start=1
ct_custom_prefix=0
[[ -z "${CONCORDTREE_INSTALL_DIR:-}" ]] || ct_custom_prefix=1
while (($#)); do
    case "$1" in
        --from|--base-url|--prefix|--runtime)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --from) ct_source="$2";;
                --base-url) ct_base="$2";;
                --prefix) ct_prefix="$2"; ct_custom_prefix=1;;
                --runtime) ct_runtime="$2";;
            esac
            shift 2;;
        --no-start) ct_start=0; shift;;
        --help)
            echo 'install.sh [--runtime auto|portable|docker] [--base-url HTTPS_URL | --from LOCAL_ASSETS] [--prefix DIRECTORY] [--no-start] [-- WEB_OPTIONS]'
            echo 'Auto uses portable when glibc and C compiler prerequisites are met; otherwise it uses an existing working Docker installation.'
            exit 0;;
        --) shift; break;;
        *) echo "Unknown installer argument: $1" >&2; exit 2;;
    esac
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'This artifact requires Linux x86_64.' >&2; exit 2; }
[[ "$ct_runtime" == auto || "$ct_runtime" == portable || "$ct_runtime" == docker ]] || { echo 'Choose --runtime auto, portable or docker.' >&2; exit 2; }
for ct_tool in tar gzip sha256sum; do
    command -v "$ct_tool" >/dev/null || { echo "Missing prerequisite: $ct_tool" >&2; exit 2; }
done

ct_portable_ok=0
ct_compiler=''
ct_libc="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
if [[ "$ct_libc" =~ ^glibc[[:space:]]+([0-9]+)\.([0-9]+) ]]; then
    if ((10#${BASH_REMATCH[1]} > 2 || (10#${BASH_REMATCH[1]} == 2 && 10#${BASH_REMATCH[2]} >= 34))); then
        ct_compiler="$(command -v gcc || command -v clang || true)"
        if [[ -n "$ct_compiler" ]]; then
            ct_probe="$(mktemp -d)"
            if printf '#include <stdio.h>\nint main(void) { return 0; }\n' | "$ct_compiler" -x c -c -o "$ct_probe/check.o" - >/dev/null 2>&1; then
                ct_portable_ok=1
            fi
            rm -rf -- "$ct_probe"
        fi
    fi
fi
if [[ "$ct_runtime" == auto ]]; then
    if ((ct_portable_ok)); then ct_runtime=portable; else ct_runtime=docker; fi
fi
if [[ "$ct_runtime" == portable ]] && ((!ct_portable_ok)); then
    echo 'Portable requires glibc >=2.34 and GCC/Clang with C headers. Ubuntu/Debian: sudo apt install gcc libc6-dev. Or choose --runtime docker.' >&2
    exit 2
fi
ct_image_ready=0
if [[ "$ct_runtime" == docker ]]; then
    command -v docker >/dev/null || { echo 'Portable prerequisites are missing and Docker is unavailable. Install GCC + C headers on glibc >=2.34, or set up Docker and NVIDIA Container Toolkit, then rerun.' >&2; exit 2; }
    docker info >/dev/null 2>&1 || { echo 'Cannot access Docker. Start its daemon and grant this user Docker access, then rerun.' >&2; exit 2; }
    [[ "$(docker image inspect "$ct_image" --format '{{.Id}}' 2>/dev/null || true)" != "$ct_image_id" ]] || ct_image_ready=1
    ((ct_custom_prefix)) || ct_prefix="${ct_prefix}-docker"
fi
printf 'Selected runtime: %s\n' "$ct_runtime"

if [[ -x "$ct_prefix/concordtree" ]]; then
    if [[ "$ct_runtime" == portable && -d "$ct_prefix/.conda/env" ]]; then
        "$ct_prefix/.web-python/bin/python" -I -B "$ct_prefix/verify_release.py"
    elif [[ "$ct_runtime" == docker && -f "$ct_prefix/docker-web.sh" ]]; then
        ((ct_image_ready)) || { echo 'The installed launcher image is missing or changed. Reinstall to a new --prefix to restore the verified image.' >&2; exit 2; }
    else
        echo 'Destination contains a different runtime. Choose another --prefix.' >&2
        exit 2
    fi
else
    [[ ! -e "$ct_prefix" ]] || { echo "Destination already exists: $ct_prefix. Choose an empty --prefix." >&2; exit 2; }
    [[ -n "$ct_source" || "$ct_base" == https://* || "$ct_base" == http://127.0.0.1:* ]] || { echo 'Use an HTTPS asset directory or --from for local assets.' >&2; exit 2; }
    ct_parent="$(dirname -- "$ct_prefix")"
    mkdir -p -- "$ct_parent"
    ct_cache="${XDG_CACHE_HOME:-$HOME/.cache}/concordtree-downloads/1.0.2-linux-r1"
    mkdir -p -- "$ct_cache"
    ct_fetch() {
        local ct_file="$1"
        if [[ -n "$ct_source" ]]; then
            [[ "$ct_source/$ct_file" -ef "$ct_cache/$ct_file" ]] || cp -- "$ct_source/$ct_file" "$ct_cache/$ct_file"
        else
            command -v curl >/dev/null || { echo 'curl is required for downloads.' >&2; exit 2; }
            curl --fail --location --retry 4 --connect-timeout 20 --continue-at - \
                --output "$ct_cache/$ct_file.partial" "${ct_base%/}/$ct_file"
            mv -- "$ct_cache/$ct_file.partial" "$ct_cache/$ct_file"
        fi
    }
    ct_manifest=SHA256SUMS
    [[ "$ct_runtime" != docker ]] || ct_manifest=DOCKER_SHA256SUMS
    ct_fetch "$ct_manifest"
    ct_parts=()
    ct_part_count=0
    ct_files=()
    ct_hashes=()
    ct_launchers=0
    while read -r ct_hash ct_file ct_extra; do
        [[ "$ct_hash" =~ ^[0-9a-f]{64}$ && -z "$ct_extra" ]] || { echo 'Invalid asset checksum manifest.' >&2; exit 2; }
        ct_expected="$(printf '%03d' "$ct_part_count")"
        if [[ "$ct_runtime" == portable && "$ct_file" == "$ct_name.tar.gz.part-$ct_expected" ]] || \
           [[ "$ct_runtime" == docker && "$ct_file" == "concordtree-1.0.2-docker.tar.gz.part-$ct_expected" ]]; then
            ct_parts+=("$ct_cache/$ct_file")
            ((ct_part_count+=1))
            if [[ "$ct_runtime" == docker ]] && ((ct_image_ready)); then continue; fi
        elif [[ "$ct_runtime" == docker && "$ct_file" == docker-web.sh ]]; then
            ((ct_launchers+=1))
        else
            echo 'Unexpected or out-of-order asset in checksum manifest.' >&2; exit 2
        fi
        ct_files+=("$ct_file")
        ct_hashes+=("$ct_hash")
    done < "$ct_cache/$ct_manifest"
    ((ct_part_count == 5)) || { echo 'Expected exactly five ordered archive parts.' >&2; exit 2; }
    [[ "$ct_runtime" != docker ]] || ((ct_launchers == 1)) || { echo 'Missing or duplicated Docker launcher checksum.' >&2; exit 2; }
    for ((ct_i=0; ct_i<${#ct_files[@]}; ct_i++)); do
        ct_file="${ct_files[ct_i]}"; ct_hash="${ct_hashes[ct_i]}"
        if [[ ! -f "$ct_cache/$ct_file" ]] || ! (cd -- "$ct_cache" && printf '%s  %s\n' "$ct_hash" "$ct_file" | sha256sum --check --status); then
            ct_fetch "$ct_file"
            (cd -- "$ct_cache" && printf '%s  %s\n' "$ct_hash" "$ct_file" | sha256sum --check) || exit 1
        fi
    done
    ct_stage="$(mktemp -d "$ct_parent/.concordtree-install.XXXXXXXX")"
    trap 'rm -rf -- "$ct_stage"' EXIT
    if [[ "$ct_runtime" == portable ]]; then
        cat -- "${ct_parts[@]}" | tar -xz --no-same-owner --warning=no-timestamp -C "$ct_stage"
        "$ct_stage/$ct_name/.web-python/bin/python" -I -B "$ct_stage/$ct_name/verify_release.py"
        mv -T -- "$ct_stage/$ct_name" "$ct_prefix"
        rmdir -- "$ct_stage"
    else
        if ((!ct_image_ready)); then
            cat -- "${ct_parts[@]}" | gzip -dc | docker load
            [[ "$(docker image inspect "$ct_image" --format '{{.Id}}')" == "$ct_image_id" ]] || { echo 'Loaded image does not match this release.' >&2; exit 1; }
        fi
        cp -- "$ct_cache/docker-web.sh" "$ct_stage/docker-web.sh"
        chmod +x "$ct_stage/docker-web.sh"
        cat > "$ct_stage/concordtree" <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail
ct_here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ct_here/docker-web.sh" "$@"
LAUNCHER
        chmod +x "$ct_stage/concordtree"
        mv -T -- "$ct_stage" "$ct_prefix"
    fi
    trap - EXIT
fi
printf 'Installed: %s\nStart again: %s/concordtree\n' "$ct_prefix" "$ct_prefix"
if ((ct_start)); then exec "$ct_prefix/concordtree" "$@"; fi
