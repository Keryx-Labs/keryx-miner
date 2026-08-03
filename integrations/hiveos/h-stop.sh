#!/usr/bin/env bash

set -euo pipefail

# HiveOS can invoke stop from a cwd that no longer exists after package updates.
# Move to a safe location so subsequent script logic is unaffected.
cd / || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/h-manifest.conf"

# This script is executed by HiveOS when stopping the custom miner.

signal_miner() {
	local signal="$1"
	pkill "-$signal" -x "${CUSTOM_MINERBIN}" || pkill "-$signal" -f "${CUSTOM_MINER_DIR}/${CUSTOM_MINERBIN}" || true
}

# Give the miner time to flush escrow state before stopping wrappers or screen.
signal_miner TERM
for _ in $(seq 1 10); do
	pgrep -x "${CUSTOM_MINERBIN}" >/dev/null 2>&1 || break
	sleep 1
done

# A second graceful signal asks a hung shutdown to exit immediately.
if pgrep -x "${CUSTOM_MINERBIN}" >/dev/null 2>&1; then
	signal_miner TERM
	for _ in $(seq 1 5); do
		pgrep -x "${CUSTOM_MINERBIN}" >/dev/null 2>&1 || break
		sleep 1
	done
fi

if command -v screen >/dev/null 2>&1; then
	screen -S "miner" -X quit || true
	screen -S "${CUSTOM_NAME}" -X quit || true
fi

pkill -f "${CUSTOM_MINER_DIR}/h-run.sh" || true
pkill -f "screen.*${CUSTOM_MINERBIN}" || true
pkill -f "screen.*${CUSTOM_NAME}" || true

# Force-stop only after both graceful shutdown windows.
signal_miner KILL
