#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/h-manifest.conf"
. "$SCRIPT_DIR/h-state.sh"

cleanup_legacy_models() {
    local legacy_dir="$CUSTOM_MINER_DIR/models"
    local shared_dir="/hive/miners/custom/models"

    [[ ! -d "$legacy_dir" ]] && return 0
    [[ "$legacy_dir" == "$shared_dir" ]] && return 0

    if [[ ! -d "$shared_dir" ]]; then
        mv "$legacy_dir" "$shared_dir" || true
        return 0
    fi

    # Merge only entries that do not yet exist in the shared cache.
    for entry in "$legacy_dir"/*; do
        [[ -e "$entry" ]] || break
        local name
        name="$(basename "$entry")"
        [[ -e "$shared_dir/$name" ]] && continue
        mv "$entry" "$shared_dir/$name" || true
    done

    # Remove empty directories left behind after merge.
    find "$legacy_dir" -depth -type d -empty -delete 2>/dev/null || true

    if rmdir "$legacy_dir" 2>/dev/null; then
        echo "[keryx] Legacy model cache cleaned up: $legacy_dir" >&2
        return 0
    fi

    if [[ "${KERYX_PURGE_LEGACY_MODELS:-0}" == "1" ]]; then
        rm -rf "$legacy_dir" || true
        echo "[keryx] WARNING: force-purged legacy model cache at $legacy_dir (KERYX_PURGE_LEGACY_MODELS=1)." >&2
    else
        echo "[keryx] WARNING: legacy model cache still exists at $legacy_dir while $shared_dir already exists (possible duplicate disk usage). Set KERYX_PURGE_LEGACY_MODELS=1 to remove it automatically." >&2
    fi
}

# Run the current stop hook first so its preservation logic executes before
# any pre-upgrade migration done here.
if [[ -x "$SCRIPT_DIR/h-stop.sh" ]]; then
    "$SCRIPT_DIR/h-stop.sh" || true
fi

prepare_keryx_state "$CUSTOM_MINER_DIR" || {
    echo "[keryx] ERROR: refusing upgrade because escrow state could not be preserved." >&2
    exit 1
}

cleanup_legacy_models

echo "[keryx] HiveOS Pre-upgrade preparation complete."
echo "[keryx] It is now safe to change the Install URL in the HiveOS custom miner config screen and apply changes."
