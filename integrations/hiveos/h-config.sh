#!/usr/bin/env bash

# Self-locate the manifest from THIS script's own directory, so the package works under any folder
# name (versioned or not) with no hardcoded /hive/miners/custom/keryx-miner path and no symlink.
# No cd / no exit here: HiveOS may source this file.
__MD="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" && pwd)"
. "$__MD/h-manifest.conf"

conf=""
conf+=" -s $CUSTOM_URL --mining-address $CUSTOM_TEMPLATE"
conf+=" --plain-log-file $CUSTOM_LOG_BASENAME.log"
[[ "$CUSTOM_USER_CONFIG" != *"--escrow-key-file"* ]] && conf+=" --escrow-key-file $KERYX_STATE_DIR/escrow.key"
[[ "$CUSTOM_USER_CONFIG" != *"--escrow-state-file"* ]] && conf+=" --escrow-state-file $KERYX_STATE_DIR/escrow_state.json"

[[ ! -z $CUSTOM_USER_CONFIG ]] && conf+=" $CUSTOM_USER_CONFIG"

echo "$conf"
echo "$conf" > $CUSTOM_CONFIG_FILENAME
