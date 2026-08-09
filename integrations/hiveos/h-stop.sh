#!/usr/bin/env bash

set -euo pipefail

# HiveOS can invoke stop from a cwd that no longer exists after package updates.
# Move to a safe location so subsequent script logic is unaffected.
cd / || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/h-manifest.conf"

# This script is executed by HiveOS when stopping the custom miner.

miner_pids() {
	{
		pgrep -x "${CUSTOM_MINERBIN}" || true
		pgrep -f -- "${CUSTOM_MINER_DIR}/${CUSTOM_MINERBIN}" || true
	} | sort -un
}

signal_miner() {
	local signal="$1"
	local -a pids=()
	mapfile -t pids < <(miner_pids)
	(( ${#pids[@]} == 0 )) || kill "-$signal" "${pids[@]}" || true
}

# Give the miner time to flush escrow state before stopping wrappers or screen.
signal_miner TERM
for ((i = 0; i < 15; i++)); do
	[[ -n "$(miner_pids)" ]] || break
	sleep 1
done

# Force-stop only after the complete graceful shutdown window.
signal_miner KILL

if command -v screen >/dev/null 2>&1; then
	screen -S "miner" -X quit || true
	screen -S "${CUSTOM_NAME}" -X quit || true
fi

pkill -f "${CUSTOM_MINER_DIR}/h-run.sh" || true
pkill -f "screen.*${CUSTOM_MINERBIN}" || true
pkill -f "screen.*${CUSTOM_NAME}" || true
