#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UPGRADE_SCRIPT="$ROOT/integrations/hiveos/pre-hive-upgrade.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

export KERYX_SYNC_COMMAND=true
. "$UPGRADE_SCRIPT"

fail() {
    echo "state-migration: $*" >&2
    exit 1
}

write_key() {
    printf '11%.0s' {1..32} > "$1"
}

write_other_key() {
    printf '22%.0s' {1..32} > "$1"
}

write_state() {
    printf '{"entries":[],"marker":"%s"}\n' "$2" > "$1"
}

assert_mode() {
    [[ "$(stat -c '%a' "$1")" == "$2" ]] || fail "expected mode $2 for $1"
}

case_dir() {
    local name="$1"
    rm -rf "$WORK/$name"
    mkdir -p "$WORK/$name/install"
    printf '%s\n' "$WORK/$name"
}

# Fresh install creates only the protected durable directory.
base="$(case_dir fresh)"
KERYX_STATE_DIR="$base/state" prepare_keryx_state "$base/install"
[[ -d "$base/state" && ! -e "$base/state/escrow.key" ]] || fail "fresh install"
assert_mode "$base/state" 700

# First migration copies, validates, syncs, and protects both files.
base="$(case_dir first)"
write_key "$base/install/escrow.key"
write_state "$base/install/escrow_state.json" first
KERYX_STATE_DIR="$base/state" prepare_keryx_state "$base/install"
cmp -s "$base/install/escrow.key" "$base/state/escrow.key" || fail "key copy"
cmp -s "$base/install/escrow_state.json" "$base/state/escrow_state.json" || fail "state copy"
assert_mode "$base/state/escrow.key" 600
assert_mode "$base/state/escrow_state.json" 600

# Repeated migration keeps an existing valid durable destination.
write_other_key "$base/install/escrow.key"
write_state "$base/install/escrow_state.json" legacy-newer
KERYX_STATE_DIR="$base/state" prepare_keryx_state "$base/install"
[[ "$(cat "$base/state/escrow.key")" == "$(printf '11%.0s' {1..32})" ]] || fail "idempotent key precedence"
grep -q '"marker":"first"' "$base/state/escrow_state.json" || fail "idempotent state precedence"

# A valid durable destination also takes precedence over stale invalid legacy data.
printf 'broken' > "$base/install/escrow.key"
printf '{broken' > "$base/install/escrow_state.json"
KERYX_STATE_DIR="$base/state" prepare_keryx_state "$base/install"
keryx_validate_key "$base/state/escrow.key" || fail "valid durable key lost precedence"
keryx_validate_state "$base/state/escrow_state.json" || fail "valid durable state lost precedence"

# Partial migration preserves an existing valid key and fills the missing state.
base="$(case_dir partial)"
mkdir -p "$base/state"
write_key "$base/state/escrow.key"
write_other_key "$base/install/escrow.key"
write_state "$base/install/escrow_state.json" partial
KERYX_STATE_DIR="$base/state" prepare_keryx_state "$base/install"
[[ "$(cat "$base/state/escrow.key")" == "$(printf '11%.0s' {1..32})" ]] || fail "partial key precedence"
grep -q '"marker":"partial"' "$base/state/escrow_state.json" || fail "partial state copy"

# Invalid durable or legacy data fails without replacing either side.
base="$(case_dir invalid-durable)"
mkdir -p "$base/state"
printf 'broken' > "$base/state/escrow.key"
write_key "$base/install/escrow.key"
if KERYX_STATE_DIR="$base/state" prepare_keryx_state "$base/install" 2>/dev/null; then
    fail "invalid durable key accepted"
fi
[[ "$(cat "$base/state/escrow.key")" == broken ]] || fail "invalid durable key changed"

base="$(case_dir invalid-legacy)"
printf '{broken' > "$base/install/escrow_state.json"
if KERYX_STATE_DIR="$base/state" prepare_keryx_state "$base/install" 2>/dev/null; then
    fail "invalid legacy state accepted"
fi
[[ "$(cat "$base/install/escrow_state.json")" == '{broken' ]] || fail "invalid legacy state changed"

base="$(case_dir invalid-entry)"
printf '{"entries":[{}]}\n' > "$base/install/escrow_state.json"
if KERYX_STATE_DIR="$base/state" prepare_keryx_state "$base/install" 2>/dev/null; then
    fail "state with an invalid entry accepted"
fi

# Validation matches Rust trimming, integer ranges, and optional field types.
base="$(case_dir invalid-key-newline)"
{ printf '11%.0s' {1..16}; printf '\n'; printf '11%.0s' {1..16}; } > "$base/install/escrow.key"
if KERYX_STATE_DIR="$base/state" prepare_keryx_state "$base/install" 2>/dev/null; then
    fail "key with embedded newline accepted"
fi

for case in oversized optional-type; do
    base="$(case_dir "invalid-$case")"
    if [[ "$case" == oversized ]]; then
        printf '{"entries":[{"coinbase_txid":"x","confirm_daa":18446744073709551616,"amount_sompi":1,"claimed":false,"slashed":false}]}\n' > "$base/install/escrow_state.json"
    else
        printf '{"entries":[{"coinbase_txid":"x","confirm_daa":1,"amount_sompi":1,"claimed":false,"slashed":false,"output_index":"1"}]}\n' > "$base/install/escrow_state.json"
    fi
    if KERYX_STATE_DIR="$base/state" prepare_keryx_state "$base/install" 2>/dev/null; then
        fail "$case state accepted"
    fi
done

# Durable paths and legacy sources must be real directories/files, not symlinks.
base="$(case_dir symlink-state-dir)"
mkdir -p "$base/linked-state"
ln -s "$base/linked-state" "$base/state"
if KERYX_STATE_DIR="$base/state" prepare_keryx_state "$base/install" 2>/dev/null; then
    fail "symlink state directory accepted"
fi

base="$(case_dir symlink-source)"
write_key "$base/real.key"
ln -s "$base/real.key" "$base/install/escrow.key"
if KERYX_STATE_DIR="$base/state" prepare_keryx_state "$base/install" 2>/dev/null; then
    fail "symlink legacy key accepted"
fi

base="$(case_dir symlink-destination)"
mkdir -p "$base/state"
write_key "$base/real.key"
ln -s "$base/real.key" "$base/state/escrow.key"
if KERYX_STATE_DIR="$base/state" prepare_keryx_state "$base/install" 2>/dev/null; then
    fail "symlink durable key accepted"
fi

# One lock covers the complete key/state migration.
base="$(case_dir concurrent-lock)"
mkdir -p "$base/state"
write_key "$base/install/escrow.key"
write_state "$base/install/escrow_state.json" locked
exec 8>"$base/state/.migration.lock"
flock 8
(KERYX_STATE_DIR="$base/state" prepare_keryx_state "$base/install") &
migration_pid=$!
sleep 0.2
[[ ! -e "$base/state/escrow.key" ]] || fail "migration ignored held lock"
flock -u 8
wait "$migration_pid"
keryx_validate_key "$base/state/escrow.key" || fail "locked migration did not complete"

# Injected copy and install failures leave no destination or live temporary file.
for stage in copy install; do
    base="$(case_dir "fail-$stage")"
    write_key "$base/install/escrow.key"
    if [[ "$stage" == copy ]]; then
        if KERYX_STATE_DIR="$base/state" KERYX_COPY_COMMAND=false prepare_keryx_state "$base/install" 2>/dev/null; then
            fail "copy failure accepted"
        fi
    else
        if KERYX_STATE_DIR="$base/state" KERYX_INSTALL_COMMAND=false prepare_keryx_state "$base/install" 2>/dev/null; then
            fail "install failure accepted"
        fi
    fi
    [[ ! -e "$base/state/escrow.key" ]] || fail "$stage failure installed destination"
    [[ -z "$(find "$base/state" -name '*.migrate.*' -print -quit)" ]] || fail "$stage failure left temporary file"
done

# A residue from an interrupted older invocation is ignored and never promoted.
base="$(case_dir residue)"
mkdir -p "$base/state"
printf 'partial' > "$base/state/.escrow.key.migrate.abandoned"
write_key "$base/install/escrow.key"
KERYX_STATE_DIR="$base/state" prepare_keryx_state "$base/install"
keryx_validate_key "$base/state/escrow.key" || fail "residue blocked migration"
[[ -e "$base/state/.escrow.key.migrate.abandoned" ]] || fail "unowned residue was removed"

# Simulate HiveOS deleting the package for upgrade and rollback. Persistent Extra
# flags make the unchanged old h-config and the new h-config select the same files.
base="$(case_dir rollback)"
state_dir="$base/keryx-miner-state"
flags="--escrow-key-file $state_dir/escrow.key --escrow-state-file $state_dir/escrow_state.json"
wallet="$base/wallet.conf"
printf "CUSTOM_USER_CONFIG='%s'\n" "$flags" > "$wallet"

make_old_package() {
    local dir="$1"
    mkdir -p "$dir"
    cat > "$dir/h-manifest.conf" <<EOF
CUSTOM_NAME=keryx-miner
CUSTOM_MINERBIN=keryx-miner
CUSTOM_MINER_DIR="$dir"
CUSTOM_CONFIG_FILENAME="$dir/config.ini"
CUSTOM_LOG_BASENAME=/tmp/keryx-miner
EOF
    cat > "$dir/h-config.sh" <<'EOF'
. "$(dirname "${BASH_SOURCE[0]}")/h-manifest.conf"
conf=" -s $CUSTOM_URL --mining-address $CUSTOM_TEMPLATE --plain-log-file $CUSTOM_LOG_BASENAME.log"
[[ -z ${CUSTOM_USER_CONFIG:-} ]] || conf+=" $CUSTOM_USER_CONFIG"
printf '%s\n' "$conf" > "$CUSTOM_CONFIG_FILENAME"
EOF
}

old="$base/keryx-miner"
make_old_package "$old"
write_key "$old/escrow.key"
write_state "$old/escrow_state.json" old
KERYX_STATE_DIR="$state_dir" prepare_keryx_state "$old"

CUSTOM_URL=grpc://node CUSTOM_TEMPLATE=wallet CUSTOM_USER_CONFIG="$flags" bash -c ". '$old/h-config.sh'"
grep -Fq -- "$flags" "$old/config.ini" || fail "old package did not use persistent flags"
rm -rf "$old"

new="$base/keryx-miner"
mkdir -p "$new"
cp "$ROOT/integrations/hiveos/h-config.sh" "$new/h-config.sh"
cp "$ROOT/integrations/hiveos/h-manifest.conf" "$new/h-manifest.conf"
CUSTOM_URL=grpc://node CUSTOM_TEMPLATE=wallet CUSTOM_USER_CONFIG="$flags" KERYX_STATE_DIR="$state_dir" bash -c ". '$new/h-config.sh'" >/dev/null
grep -Fq -- "$flags" "$new/config.ini" || fail "new package did not preserve persistent flags"
if CUSTOM_URL=grpc://node CUSTOM_TEMPLATE=wallet CUSTOM_USER_CONFIG="$flags --escrow-key-file=$state_dir/other.key" KERYX_STATE_DIR="$state_dir" bash -c ". '$new/h-config.sh'" >/dev/null 2>&1; then
    fail "h-config accepted duplicate escrow key flags"
fi
if CUSTOM_URL=grpc://node CUSTOM_TEMPLATE=wallet CUSTOM_USER_CONFIG="--escrow-key-file --escrow-state-file $state_dir/escrow_state.json" KERYX_STATE_DIR="$state_dir" bash -c ". '$new/h-config.sh'" >/dev/null 2>&1; then
    fail "h-config accepted missing escrow key value"
fi
write_state "$state_dir/escrow_state.json" new
rm -rf "$new"

make_old_package "$old"
CUSTOM_URL=grpc://node CUSTOM_TEMPLATE=wallet CUSTOM_USER_CONFIG="$flags" bash -c ". '$old/h-config.sh'"
grep -Fq -- "$flags" "$old/config.ini" || fail "rollback package did not use durable state"
grep -q '"marker":"new"' "$state_dir/escrow_state.json" || fail "rollback lost new-package state"

# The standalone script requires an explicit install directory and verifies,
# but never edits, the persistent wallet configuration.
if KERYX_STATE_DIR="$state_dir" KERYX_WALLET_CONF="$wallet" bash "$UPGRADE_SCRIPT" >/dev/null 2>&1; then
    fail "standalone script accepted no install directory"
fi
wallet_before="$(cat "$wallet")"
KERYX_STATE_DIR="$state_dir" KERYX_WALLET_CONF="$wallet" KERYX_SYNC_COMMAND=true \
    bash "$UPGRADE_SCRIPT" --install-dir "$old" >/dev/null
[[ "$(cat "$wallet")" == "$wallet_before" ]] || fail "standalone script edited wallet.conf"

for bad_flags in \
    "$flags --escrow-key-file=$state_dir/other.key" \
    "--escrow-key-file --escrow-state-file $state_dir/escrow_state.json" \
    "--escrow-key-file=$state_dir/other.key --escrow-state-file $state_dir/escrow_state.json"; do
    printf "CUSTOM_USER_CONFIG='%s'\n" "$bad_flags" > "$wallet"
    if KERYX_STATE_DIR="$state_dir" KERYX_WALLET_CONF="$wallet" verify_keryx_rollback_config >/dev/null 2>&1; then
        fail "rollback verification accepted malformed/conflicting flags"
    fi
done

# A valid manifest without a migrated durable key is never upgrade-safe.
base="$(case_dir standalone-no-key)"
make_old_package "$base/install"
flags="--escrow-key-file $base/state/escrow.key --escrow-state-file $base/state/escrow_state.json"
printf "CUSTOM_USER_CONFIG='%s'\n" "$flags" > "$base/wallet.conf"
if KERYX_STATE_DIR="$base/state" KERYX_WALLET_CONF="$base/wallet.conf" KERYX_SYNC_COMMAND=true \
    bash "$UPGRADE_SCRIPT" --install-dir "$base/install" >/dev/null 2>&1; then
    fail "standalone script reported safe without a durable key"
fi

echo "state-migration: all focused cases passed"
