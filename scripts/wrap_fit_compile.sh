#!/usr/bin/env bash
set -euo pipefail

patch_root=$1
patch_file=$2
fit_source=$3
backup_file=$4
shift 4

if [[ ${1:-} == "--" ]]; then
    shift
fi

restore_original() {
    if [[ -f "$backup_file" ]]; then
        cp "$backup_file" "$fit_source"
        rm -f "$backup_file"
    fi
}

run_patch() {
    patch --batch --silent "$@"
}

is_patch_applied() {
    run_patch --dry-run -R -p1 -d "$patch_root" < "$patch_file" >/dev/null 2>&1
}

ensure_unpatched_source() {
    if [[ -f "$backup_file" ]]; then
        cp "$backup_file" "$fit_source"
        rm -f "$backup_file"
        return
    fi

    if is_patch_applied; then
        run_patch -R -p1 -d "$patch_root" < "$patch_file" >/dev/null 2>&1
    fi
}

is_fit_compile=false
for arg in "$@"; do
    if [[ "$arg" == "$fit_source" || "$arg" == */vendor/llama-cpp/common/fit.cpp || "$arg" == "fit.cpp" ]]; then
        is_fit_compile=true
        break
    fi
done

if [[ "$is_fit_compile" != true ]]; then
    "$@"
    exit $?
fi

ensure_unpatched_source
cp "$fit_source" "$backup_file"
trap restore_original EXIT

if ! run_patch --forward -p1 -d "$patch_root" < "$patch_file" >/dev/null 2>&1; then
    if ! is_patch_applied; then
        echo "wrap_fit_compile.sh: failed to apply patch ${patch_file}" >&2
        exit 1
    fi
fi

set +e
"$@"
status=$?
set -e

exit $status
