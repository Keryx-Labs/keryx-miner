#!/usr/bin/env bash

KERYX_STATE_DIR="${KERYX_STATE_DIR:-/hive/miners/custom/keryx-miner-state}"

prepare_keryx_state() {
  local install_dir="${1:?miner install directory is required}"

  umask 077
  install -d -m 700 "$KERYX_STATE_DIR" || return 1

  local name source destination temporary
  for name in escrow.key escrow_state.json; do
    source="$install_dir/$name"
    destination="$KERYX_STATE_DIR/$name"
    [[ -e "$source" ]] || continue
    [[ -e "$destination" ]] && continue

    temporary="$(mktemp "$KERYX_STATE_DIR/.${name}.XXXXXX")" || return 1
    if ! cp -- "$source" "$temporary" || ! cmp -s -- "$source" "$temporary"; then
      rm -f -- "$temporary"
      echo "[keryx] ERROR: failed to preserve $source" >&2
      return 1
    fi
    chmod 600 "$temporary" || { rm -f -- "$temporary"; return 1; }

    if ! ln -- "$temporary" "$destination" 2>/dev/null; then
      if [[ ! -e "$destination" ]]; then
        rm -f -- "$temporary"
        echo "[keryx] ERROR: failed to install $destination" >&2
        return 1
      fi
    else
      echo "[keryx] Preserved $name in $KERYX_STATE_DIR" >&2
    fi
    rm -f -- "$temporary"
  done

  chmod 600 "$KERYX_STATE_DIR"/escrow.key "$KERYX_STATE_DIR"/escrow_state.json 2>/dev/null || true
}
