#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
	echo "No arguments supplied. Call using createmanifest.sh <VERSION_NUMBER> <MINER BINARY NAME>"
	exit 1
fi

cat > h-manifest.conf << EOF
# The name of the miner
CUSTOM_NAME=keryx-miner

# Optional version of your custom miner package
CUSTOM_VERSION=$1
CUSTOM_BUILD=0
CUSTOM_MINERBIN=$2

# Resolve the actual versioned package directory from this manifest.
CUSTOM_MINER_DIR="\$(cd "\$(dirname "\$(readlink -f "\${BASH_SOURCE[0]:-\$0}")")" && pwd)"

# Full path to miner config file
CUSTOM_CONFIG_FILENAME="\$CUSTOM_MINER_DIR/config.ini"

# Full path to log file basename (without .log extension)
CUSTOM_LOG_BASENAME=/var/log/miner/\$CUSTOM_NAME

WEB_PORT=3338

# Funds-critical mutable state must survive package replacement.
KERYX_STATE_DIR="\${KERYX_STATE_DIR:-/hive/miners/custom/keryx-miner-state}"
EOF
