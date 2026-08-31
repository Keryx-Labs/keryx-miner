#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(cd "$(mktemp -d)" && pwd -P)"
trap 'rm -rf "$TEST_ROOT"' EXIT
install -m 0755 "$ROOT/h-run.sh" "$TEST_ROOT/h-run.sh"
install -m 0644 "$ROOT/h-manifest.conf" "$TEST_ROOT/h-manifest.conf"
install -m 0644 "$ROOT/pre-hive-upgrade.sh" "$TEST_ROOT/pre-hive-upgrade.sh"
# Append a source-time HOME capture to the sandboxed helper copy so it records
# the HOME value in effect WHILE pre-hive-upgrade.sh is being sourced, before
# the launcher proceeds past helper sourcing to the miner launch. A regression
# that moves the launcher's HOME export below helper sourcing leaves HOME unset
# or raw at this point, so the per-case source check below fails. The capture
# guard keeps the appended snippet inert unless the test opts in via
# KERYX_TEST_HOME_CAPTURE.
cat >> "$TEST_ROOT/pre-hive-upgrade.sh" <<'EOF'
if [[ -n "${KERYX_TEST_HOME_CAPTURE:-}" ]]; then
    printf '%s\n' "HOME=$HOME" > "$KERYX_TEST_HOME_CAPTURE"
fi
EOF
install -m 0644 "$ROOT/h-manifest.conf" "$TEST_ROOT/config.ini"
install -d "$TEST_ROOT/bin"
printf '#!/usr/bin/env bash\nexit 0\n' > "$TEST_ROOT/bin/flock"
chmod +x "$TEST_ROOT/bin/flock"

fail() {
    echo "home-fallback: $*" >&2
    exit 1
}

# Stub miner: sourcing h-run.sh runs the real launcher preamble (escrow
# migration, model-cache mkdir) and launches "./$CUSTOM_MINERBIN" as its
# child. This stub replaces the real binary, records the HOME and
# KERYX_MODELS_DIR the child process actually sees, and exits 0 so the
# launcher completes. It writes only to the absolute CAPTURE_FILE path —
# never under $HOME — and the stub flock, KERYX_SYNC_COMMAND=true,
# KERYX_STATE_DIR and KERYX_MODELS_DIR pin every launcher write inside
# TEST_ROOT, so no /root or live HiveOS path can be touched.
cat > "$TEST_ROOT/keryx-miner" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "HOME=$HOME" "KERYX_MODELS_DIR=${KERYX_MODELS_DIR:-}" > "$CAPTURE_FILE"
EOF
chmod +x "$TEST_ROOT/keryx-miner"

# The live HiveOS paths that the launcher would touch if its overrides were
# not honored. A successful test must leave them exactly as it found them.
hive_state_path=/hive/miners/custom/keryx-miner-state
hive_models_path=/hive/miners/custom/models
[[ -e "$hive_state_path" ]] && hive_state_preexisting=1 || hive_state_preexisting=0
[[ -e "$hive_models_path" ]] && hive_models_preexisting=1 || hive_models_preexisting=0
root_before=""
[[ -r /root && -x /root ]] && root_before="$(ls -A /root)"

# Source the launcher with the given HOME environment (extra env(1) args),
# then assert HOME was already correct WHILE pre-hive-upgrade.sh was being
# sourced (before miner launch), that the sourced session ended with HOME
# exported and equal to $2, that the stub child inherited it, and that
# state/model writes stayed sandboxed. Launcher stderr is kept in a per-case
# log under TEST_ROOT and dumped on failure instead of being discarded.
launch_case() {
    local label="$1" expect="$2" capture="$3" source_capture="$4" state_dir="$5" models_dir="$6"
    shift 6
    local launch_log="$TEST_ROOT/launch-$label.log"
    env "$@" PATH="$TEST_ROOT/bin:$PATH" \
        CAPTURE_FILE="$capture" \
        KERYX_TEST_HOME_CAPTURE="$source_capture" \
        KERYX_STATE_DIR="$state_dir" KERYX_MODELS_DIR="$models_dir" \
        KERYX_SYNC_COMMAND=true LAUNCH_LOG="$launch_log" bash -c '
        cd "$1"
        rc=0
        . ./h-run.sh 2>"$LAUNCH_LOG" || rc=$?
        if [[ "$rc" != 0 ]]; then
            echo "launcher did not complete with the stub miner (rc=$rc); launcher stderr:" >&2
            cat "$LAUNCH_LOG" >&2
            exit 1
        fi
        [[ "${HOME:-}" == "$2" ]] || { echo "unexpected HOME after launch (got: ${HOME:-}, want: $2)" >&2; exit 1; }
        export -p | grep -q "declare -x HOME" || { echo "HOME was not exported after launch" >&2; exit 1; }
    ' bash "$TEST_ROOT" "$expect" || fail "$label: launcher session failed"
    # HOME must already be in force at helper-source time (i.e. the export sits
    # before the helper sourcing in the launcher), not only by child launch.
    grep -Fxq "HOME=$expect" "$source_capture" || fail "$label: HOME=$expect was not in effect while pre-hive-upgrade.sh was sourced"
    grep -Fxq "HOME=$expect" "$capture" || fail "$label: child did not see HOME=$expect"
    grep -Fxq "KERYX_MODELS_DIR=$models_dir" "$capture" || fail "$label: child did not see sandboxed models dir"
    [[ -d "$models_dir" ]] || fail "$label: model dir $models_dir was not created inside TEST_ROOT"
    [[ -d "$state_dir" ]] || fail "$label: escrow state dir was not created inside TEST_ROOT"
}

# A launcher session with HOME unset must fall back to /root before any
# helper is sourced, stay exported, and be visible to the launched child.
launch_case "unset" /root \
    "$TEST_ROOT/capture-unset.txt" "$TEST_ROOT/source-unset.txt" \
    "$TEST_ROOT/state" "$TEST_ROOT/models" -u HOME

# HOME set to the empty string is as broken as unset: the launcher must
# replace it with /root rather than exporting an empty HOME to children.
launch_case "empty" /root \
    "$TEST_ROOT/capture-empty.txt" "$TEST_ROOT/source-empty.txt" \
    "$TEST_ROOT/state-empty" "$TEST_ROOT/models-empty" HOME=

# A custom HOME must pass through untouched, and an explicitly supplied
# custom KERYX_MODELS_DIR must be preserved (not reset to /hive/...).
launch_case "custom" /custom/home \
    "$TEST_ROOT/capture-custom.txt" "$TEST_ROOT/source-custom.txt" \
    "$TEST_ROOT/custom-state" "$TEST_ROOT/custom-models" HOME=/custom/home

# Nothing outside TEST_ROOT may have been created or modified: the fallback
# HOME value is only exported to children, never written to, and the only
# launcher writes are the state and model dirs pinned inside TEST_ROOT.
root_after=""
[[ -r /root && -x /root ]] && root_after="$(ls -A /root)"
[[ "$root_before" == "$root_after" ]] || fail "/root changed during the test"
[[ -e "$hive_state_path" ]] && [[ "$hive_state_preexisting" == 1 ]] \
    || [[ ! -e "$hive_state_path" ]] || fail "live HiveOS state path was created: $hive_state_path"
[[ -e "$hive_models_path" ]] && [[ "$hive_models_preexisting" == 1 ]] \
    || [[ ! -e "$hive_models_path" ]] || fail "live HiveOS models path was created: $hive_models_path"

echo 'HiveOS HOME fallback tests passed'
