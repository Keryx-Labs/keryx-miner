#!/usr/bin/env bash

KERYX_STATE_DIR="${KERYX_STATE_DIR:-/hive/miners/custom/keryx-miner-state}"
KERYX_ESCROW_FLAGS="--escrow-key-file $KERYX_STATE_DIR/escrow.key --escrow-state-file $KERYX_STATE_DIR/escrow_state.json"

keryx_validate_key() {
    local path="$1" key order LC_ALL=C
    [[ -f "$path" && ! -L "$path" ]] || return 1
    key="$(< "$path")"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    key="${key,,}"
    order="fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"
    [[ "$key" =~ ^[0-9a-f]{64}$ ]] && [[ "$key" != "$(printf '%064d' 0)" ]] && [[ "$key" < "$order" ]]
}

keryx_validate_state() {
    local path="$1"
    [[ -f "$path" && ! -L "$path" ]] || return 1
    if command -v python3 >/dev/null 2>&1; then
        python3 -I - "$path" <<'PY'
import json, sys

U8, U32, U64 = 2**8 - 1, 2**32 - 1, 2**64 - 1
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert isinstance(value, dict) and isinstance(value.get("entries"), list)

def integer(value, maximum):
    return type(value) is int and 0 <= value <= maximum

for entry in value["entries"]:
    assert isinstance(entry, dict)
    assert isinstance(entry.get("coinbase_txid"), str)
    assert integer(entry.get("confirm_daa"), U64)
    assert integer(entry.get("amount_sompi"), U64)
    assert type(entry.get("claimed")) is bool and type(entry.get("slashed")) is bool
    assert "block_hash" not in entry or isinstance(entry["block_hash"], str)
    assert "output_index" not in entry or integer(entry["output_index"], U32)
    assert "orphan_slashed" not in entry or type(entry["orphan_slashed"]) is bool
    assert "orphan_retries" not in entry or integer(entry["orphan_retries"], U8)
    assert "orphan_retry_after_daa" not in entry or entry["orphan_retry_after_daa"] is None or integer(entry["orphan_retry_after_daa"], U64)
    assert "submit_retries" not in entry or integer(entry["submit_retries"], U8)
    assert "batch_cap" not in entry or integer(entry["batch_cap"], U8)
    assert "cap_set_daa" not in entry or integer(entry["cap_set_daa"], U64)
    assert "is_inference" not in entry or type(entry["is_inference"]) is bool
PY
    elif command -v jq >/dev/null 2>&1; then
        # jq numbers cannot represent every u64 exactly, so its fallback is deliberately conservative.
        jq -e 'def uint($max): type == "number" and floor == . and . >= 0 and . <= $max;
            type == "object" and (.entries | type == "array") and all(.entries[];
                type == "object" and (.coinbase_txid | type == "string") and
                (.confirm_daa | uint(9007199254740991)) and (.amount_sompi | uint(9007199254740991)) and
                (.claimed | type == "boolean") and (.slashed | type == "boolean") and
                (has("block_hash") | not or (.block_hash | type == "string")) and
                (has("output_index") | not or (.output_index | uint(4294967295))) and
                (has("orphan_slashed") | not or (.orphan_slashed | type == "boolean")) and
                (has("orphan_retries") | not or (.orphan_retries | uint(255))) and
                (has("orphan_retry_after_daa") | not or .orphan_retry_after_daa == null or (.orphan_retry_after_daa | uint(9007199254740991))) and
                (has("submit_retries") | not or (.submit_retries | uint(255))) and
                (has("batch_cap") | not or (.batch_cap | uint(255))) and
                (has("cap_set_daa") | not or (.cap_set_daa | uint(9007199254740991))) and
                (has("is_inference") | not or (.is_inference | type == "boolean")))' "$path" >/dev/null
    else
        echo "[keryx] ERROR: jq or python3 is required to validate escrow state." >&2
        return 1
    fi
}

keryx_validate_escrow_file() {
    case "${2:-$(basename "$1")}" in
        escrow.key) keryx_validate_key "$1" ;;
        escrow_state.json) keryx_validate_state "$1" ;;
        *) return 1 ;;
    esac
}

keryx_sync_path() {
    if [[ -n "${KERYX_SYNC_COMMAND:-}" ]]; then
        "$KERYX_SYNC_COMMAND" "$1"
    else
        sync -f "$1"
    fi
}

prepare_keryx_state() (
    set -euo pipefail
    umask 077

    local install_dir="$1" state_dir="${KERYX_STATE_DIR}" name source destination temporary="" lock_file
    local state_dir_existed=0
    [[ ! -L "$state_dir" ]] || {
        echo "[keryx] ERROR: durable escrow directory must not be a symlink: $state_dir" >&2
        return 1
    }
    [[ -d "$state_dir" ]] && state_dir_existed=1
    mkdir -p "$state_dir" || return 1
    [[ -d "$state_dir" && ! -L "$state_dir" ]] || return 1
    chmod 700 "$state_dir" || return 1
    if (( ! state_dir_existed )); then
        keryx_sync_path "$(dirname "$state_dir")" || return 1
    fi

    lock_file="$state_dir/.migration.lock"
    [[ ! -L "$lock_file" ]] || {
        echo "[keryx] ERROR: migration lock must not be a symlink: $lock_file" >&2
        return 1
    }
    command -v flock >/dev/null 2>&1 || {
        echo "[keryx] ERROR: flock is required for escrow migration." >&2
        return 1
    }
    exec 9>"$lock_file" || return 1
    flock 9 || return 1

    trap '[[ -n "$temporary" ]] && rm -f -- "$temporary"' EXIT
    for name in escrow.key escrow_state.json; do
        source="$install_dir/$name"
        destination="$state_dir/$name"

        if [[ -L "$destination" ]]; then
            echo "[keryx] ERROR: durable escrow file must not be a symlink: $destination" >&2
            return 1
        elif [[ -e "$destination" ]]; then
            if ! keryx_validate_escrow_file "$destination" "$name"; then
                echo "[keryx] ERROR: durable escrow file is invalid: $destination" >&2
                return 1
            fi
            chmod 600 "$destination" || return 1
            keryx_sync_path "$destination" || return 1
            continue
        fi
        if [[ -L "$source" ]]; then
            echo "[keryx] ERROR: legacy escrow file must not be a symlink: $source" >&2
            return 1
        fi
        [[ ! -e "$source" ]] && continue
        if ! keryx_validate_escrow_file "$source" "$name"; then
            echo "[keryx] ERROR: legacy escrow file is invalid: $source" >&2
            return 1
        fi

        temporary="$(mktemp "$state_dir/.${name}.migrate.XXXXXX")" || return 1
        "${KERYX_COPY_COMMAND:-cp}" -- "$source" "$temporary" || return 1
        cmp -s -- "$source" "$temporary" || return 1
        keryx_validate_escrow_file "$temporary" "$name" || return 1
        chmod 600 "$temporary" || return 1
        keryx_sync_path "$temporary" || return 1

        if ! "${KERYX_INSTALL_COMMAND:-ln}" -- "$temporary" "$destination"; then
            if [[ -L "$destination" ]] || [[ ! -e "$destination" ]] || ! keryx_validate_escrow_file "$destination" "$name"; then
                echo "[keryx] ERROR: failed to install durable escrow file: $destination" >&2
                return 1
            fi
        fi
        rm -f -- "$temporary" || return 1
        temporary=""
        chmod 600 "$destination" || return 1
        keryx_sync_path "$state_dir" || return 1
    done

    keryx_sync_path "$state_dir" || return 1
)

keryx_flag_value_once() {
    local config="${1//$'\n'/ }" flag="$2" expected="$3" value="" word
    local -a words=()
    local count=0 index
    read -r -a words <<< "$config"
    for ((index = 0; index < ${#words[@]}; index++)); do
        word="${words[index]}"
        if [[ "$word" == "$flag" ]]; then
            ((count += 1))
            ((index += 1))
            (( index < ${#words[@]} )) || return 1
            value="${words[index]}"
            [[ "$value" != --* ]] || return 1
        elif [[ "$word" == "$flag="* ]]; then
            ((count += 1))
            value="${word#*=}"
            [[ -n "$value" ]] || return 1
        fi
    done
    [[ "$count" -eq 1 && "$value" == "$expected" ]]
}

verify_keryx_rollback_config() {
    local wallet_conf="${KERYX_WALLET_CONF:-/hive-config/wallet.conf}" config
    [[ -r "$wallet_conf" ]] || {
        echo "[keryx] ERROR: cannot read HiveOS wallet config: $wallet_conf" >&2
        return 1
    }
    config="$(bash -c 'set +u; . "$1"; printf "%s" "${CUSTOM_USER_CONFIG:-}"' _ "$wallet_conf")" || return 1
    keryx_flag_value_once "$config" --escrow-key-file "$KERYX_STATE_DIR/escrow.key" &&
        keryx_flag_value_once "$config" --escrow-state-file "$KERYX_STATE_DIR/escrow_state.json"
}

cleanup_legacy_models() {
    local install_dir="$1" legacy_dir="$1/models" shared_dir="/hive/miners/custom/models"
    [[ ! -d "$legacy_dir" || "$legacy_dir" == "$shared_dir" ]] && return 0
    if [[ ! -d "$shared_dir" ]]; then
        mv "$legacy_dir" "$shared_dir" || true
        return 0
    fi
    local entry name
    for entry in "$legacy_dir"/*; do
        [[ -e "$entry" ]] || break
        name="$(basename "$entry")"
        [[ -e "$shared_dir/$name" ]] || mv "$entry" "$shared_dir/$name" || true
    done
    find "$legacy_dir" -depth -type d -empty -delete 2>/dev/null || true
    rmdir "$legacy_dir" 2>/dev/null || true
}

pre_hive_upgrade_main() {
    set -euo pipefail
    if [[ "$#" -ne 2 || "$1" != "--install-dir" ]]; then
        echo "Usage: $0 --install-dir /hive/miners/custom/<installed-keryx-directory>" >&2
        return 2
    fi

    local install_dir
    install_dir="$(readlink -f "$2")"
    [[ -d "$install_dir" && -f "$install_dir/h-manifest.conf" ]] || {
        echo "[keryx] ERROR: invalid Keryx install directory: $2" >&2
        return 1
    }

    if [[ -x "$install_dir/h-stop.sh" ]]; then
        "$install_dir/h-stop.sh" || true
    fi
    prepare_keryx_state "$install_dir"
    if ! keryx_validate_key "$KERYX_STATE_DIR/escrow.key"; then
        echo "[keryx] ERROR: no valid durable escrow key was found after migration; upgrade is not safe." >&2
        return 1
    fi

    if ! verify_keryx_rollback_config; then
        echo "[keryx] Escrow files are preserved, but rollback is not configured." >&2
        echo "[keryx] Add this exact text to HiveOS Flight Sheet Extra config, apply it, then run this script again:" >&2
        echo "$KERYX_ESCROW_FLAGS" >&2
        return 2
    fi

    cleanup_legacy_models "$install_dir"
    echo "[keryx] Escrow migration and persistent rollback flags verified."
    echo "[keryx] It is safe to change only the Install URL while preserving Extra config."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    pre_hive_upgrade_main "$@"
fi
