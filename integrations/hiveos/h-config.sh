#!/usr/bin/env bash

# Self-locate the manifest from THIS script's own directory, so the package works under any folder
# name (versioned or not) with no hardcoded /hive/miners/custom/keryx-miner path and no symlink.
# No cd / no exit here: HiveOS may source this file.
__MD="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" && pwd)"
. "$__MD/h-manifest.conf"

conf=""
conf+=" -s $CUSTOM_URL --mining-address $CUSTOM_TEMPLATE"
conf+=" --plain-log-file $CUSTOM_LOG_BASENAME.log"

parse_custom_flag() {
  local config="${CUSTOM_USER_CONFIG//$'\n'/ }" flag="$1" word index
  local -a words=()
  CUSTOM_FLAG_COUNT=0
  CUSTOM_FLAG_VALUE=""
  read -r -a words <<< "$config"
  for ((index = 0; index < ${#words[@]}; index++)); do
    word="${words[index]}"
    if [[ "$word" == "$flag" ]]; then
      ((CUSTOM_FLAG_COUNT += 1))
      ((index += 1))
      (( index < ${#words[@]} )) || return 1
      CUSTOM_FLAG_VALUE="${words[index]}"
      [[ "$CUSTOM_FLAG_VALUE" != --* ]] || return 1
    elif [[ "$word" == "$flag="* ]]; then
      ((CUSTOM_FLAG_COUNT += 1))
      CUSTOM_FLAG_VALUE="${word#*=}"
      [[ -n "$CUSTOM_FLAG_VALUE" ]] || return 1
    fi
  done
  (( CUSTOM_FLAG_COUNT <= 1 ))
}

append_escrow_default() {
  local flag="$1" value="$2"
  if ! parse_custom_flag "$flag"; then
    echo "Invalid CUSTOM_USER_CONFIG: $flag must have one value and may appear at most once." >&2
    return 1
  fi
  (( CUSTOM_FLAG_COUNT == 1 )) || conf+=" $flag $value"
}

if ! append_escrow_default --escrow-key-file "$KERYX_STATE_DIR/escrow.key" ||
  ! append_escrow_default --escrow-state-file "$KERYX_STATE_DIR/escrow_state.json"; then
  [[ "${BASH_SOURCE[0]}" != "$0" ]] && return 1
  exit 1
fi

[[ -n ${CUSTOM_USER_CONFIG:-} ]] && conf+=" $CUSTOM_USER_CONFIG"

echo "$conf"
echo "$conf" > "$CUSTOM_CONFIG_FILENAME"
