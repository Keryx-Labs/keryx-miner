#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(mktemp -d)"
trap 'rm -rf "$ROOT"' EXIT

export KERYX_STATE_DIR="$ROOT/state"
. "$SCRIPT_DIR/../h-state.sh"

INSTALL_DIR="$ROOT/install"
mkdir -p "$INSTALL_DIR"
printf '%064d' 1 > "$INSTALL_DIR/escrow.key"
printf '{"entries":[]}' > "$INSTALL_DIR/escrow_state.json"

prepare_keryx_state "$INSTALL_DIR"
cmp "$INSTALL_DIR/escrow.key" "$KERYX_STATE_DIR/escrow.key"
cmp "$INSTALL_DIR/escrow_state.json" "$KERYX_STATE_DIR/escrow_state.json"
[[ "$(stat -c %a "$KERYX_STATE_DIR")" == "700" ]]
[[ "$(stat -c %a "$KERYX_STATE_DIR/escrow.key")" == "600" ]]
[[ "$(stat -c %a "$KERYX_STATE_DIR/escrow_state.json")" == "600" ]]

# Existing durable state wins over stale package-local files.
printf 'stale' > "$INSTALL_DIR/escrow_state.json"
prepare_keryx_state "$INSTALL_DIR"
[[ "$(cat "$KERYX_STATE_DIR/escrow_state.json")" == '{"entries":[]}' ]]

# Migration failure must leave the only key untouched.
FAIL_ROOT="$ROOT/not-a-directory"
printf 'blocked' > "$FAIL_ROOT"
export KERYX_STATE_DIR="$FAIL_ROOT"
if prepare_keryx_state "$INSTALL_DIR"; then
  echo "migration unexpectedly succeeded" >&2
  exit 1
fi
[[ "$(cat "$INSTALL_DIR/escrow.key")" == "$(printf '%064d' 1)" ]]

# Release manifest generation must retain the durable state location.
MANIFEST_DIR="$ROOT/manifest"
mkdir -p "$MANIFEST_DIR"
(
	cd "$MANIFEST_DIR"
	bash "$SCRIPT_DIR/../createmanifest.sh" 9.9.9 keryx-miner
	grep -F 'CUSTOM_MINER_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" && pwd)"' h-manifest.conf
	grep -F 'KERYX_STATE_DIR="${KERYX_STATE_DIR:-/hive/miners/custom/keryx-miner-state}"' h-manifest.conf
)

echo "state migration tests passed"
