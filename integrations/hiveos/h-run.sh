#!/usr/bin/env bash

# HiveOS may launch miners without HOME set, which aborts child processes that
# need it (notably Kubo/IPFS: "Error: $HOME is not defined"). Fall back to /root
# at the earliest launcher point, BEFORE any helper script is sourced, so every
# sourced script and every child sees HOME. A real HOME is never overridden.
export HOME="${HOME:-/root}"

cd "$(dirname "$0")" || exit 1

[ -t 1 ] && . colors

. h-manifest.conf
. "$CUSTOM_MINER_DIR/pre-hive-upgrade.sh"

[[ -z $CUSTOM_LOG_BASENAME ]] && echo -e "${RED}No CUSTOM_LOG_BASENAME is set${NOCOLOR}" && exit 1
[[ -z $CUSTOM_CONFIG_FILENAME ]] && echo -e "${RED}No CUSTOM_CONFIG_FILENAME is set${NOCOLOR}" && exit 1
[[ ! -f $CUSTOM_CONFIG_FILENAME ]] && echo -e "${RED}Custom config ${YELLOW}$CUSTOM_CONFIG_FILENAME${RED} is not found${NOCOLOR}" && exit 1

# Expose the miner dir and CUDA runtime libs (cuBLAS etc.) for OPoI inference.
# The binary self-installs cuBLAS on first run and registers its path with ldconfig,
# so this is only a belt-and-suspenders hint for the dynamic loader.
export LD_LIBRARY_PATH="$(dirname $0):${LD_LIBRARY_PATH:-}:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu"

# GNU screen on HiveOS commonly mis-renders 24-bit colors; prefer ANSI palette there.
if [[ -n "${STY:-}" && -z "${KERYX_TRUECOLOR:-}" ]]; then
  export KERYX_TRUECOLOR=0
fi

# Shared, stable model cache path across package updates/reinstalls. An
# explicitly supplied KERYX_MODELS_DIR (e.g. a sandboxed test root) is
# preserved; HiveOS packages fall back to the shared cache by default.
export KERYX_MODELS_DIR="${KERYX_MODELS_DIR:-/hive/miners/custom/models}"

cleanup_legacy_models() {
  local legacy_dir="$CUSTOM_MINER_DIR/models"
  local shared_dir="$KERYX_MODELS_DIR"

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
    echo "[keryx] WARNING: legacy model cache still exists at $legacy_dir while shared cache exists at $shared_dir (possible duplicate disk usage). Set KERYX_PURGE_LEGACY_MODELS=1 to remove it automatically." >&2
  fi
}

# One-time migration from old local-per-install cache into the shared cache.
mkdir -p "$KERYX_MODELS_DIR"
cleanup_legacy_models
prepare_keryx_state "$CUSTOM_MINER_DIR" || exit 1

config="$(< "$CUSTOM_CONFIG_FILENAME")"
config="${config//$'\n'/ }"
read -r -a miner_args <<< "$config"
"./$CUSTOM_MINERBIN" "${miner_args[@]}" --hiveos --stats-bind 127.0.0.1 --stats-port "$WEB_PORT" "$@"
