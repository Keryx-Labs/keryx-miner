#!/usr/bin/env bash

# Self-locate the manifest from THIS script's own directory (works under a versioned folder, no
# symlink). No cd / no exit: HiveOS SOURCES this file and reads $khs / $stats from it afterwards,
# so changing the caller's cwd or calling exit would break the HiveOS agent.
__MD="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" && pwd)"
. "$__MD/h-manifest.conf"

# Read the tail of the log ONCE and derive everything below from this in-memory copy, instead of
# re-reading the whole log file once per GPU. This keeps the script cheap on big rigs (12, 20+ GPUs).
# `tr -d '\000'` is a cheap guard in case the miner ever writes a stray NUL into the log.
log=$(tail -n 4000 "$CUSTOM_LOG_BASENAME.log" 2> /dev/null | tr -d '\000')

stats_raw=$(grep "Current hashrate is" <<< "$log" | tail -n 1)

maxDelay=120
time_now=$(date +%s)
api_url="http://127.0.0.1:${WEB_PORT}/v1/miner/stats"

# GPU status (shared by API + log parser paths)
readarray -t gpu_stats < <(jq --slurp -r -c '.[] | .busids, .brand, .temp, .fan, ((.memtemp // .mem_temp // .memory_temp // .temp2 // []) | join(" "))' $GPU_STATS_JSON 2> /dev/null)
read -r -a busids <<< "${gpu_stats[0]}"
read -r -a brands <<< "${gpu_stats[1]}"
read -r -a temps <<< "${gpu_stats[2]}"
read -r -a fans <<< "${gpu_stats[3]}"
read -r -a memtemps <<< "${gpu_stats[4]}"
gpu_count=${#busids[@]}

hash_arr=()
busid_arr=()
fan_arr=()
temp_arr=()
temp2_arr=()

if [ "$(gpu-detect NVIDIA)" -gt 0 ]; then
    BRAND_MINER="nvidia"
elif [ "$(gpu-detect AMD)" -gt 0 ]; then
    BRAND_MINER="amd"
fi

api_json=""
if command -v curl > /dev/null 2>&1; then
    api_json=$(curl -fsS --max-time 1 "$api_url" 2> /dev/null || true)
elif command -v wget > /dev/null 2>&1; then
    api_json=$(wget -qO- -T 1 "$api_url" 2> /dev/null || true)
fi

if [ -n "$api_json" ] && jq -e '.total_hashrate_hs != null and .devices != null and .last_update_epoch_s != null' > /dev/null 2>&1 <<< "$api_json"; then
    time_rep=$(jq -r '.last_update_epoch_s // 0' <<< "$api_json")
    diffTime=$(echo $((time_now - time_rep)) | tr -d '-')
    accepted_blocks=$(jq -r '.accepted_blocks // 0' <<< "$api_json")
    rejected_blocks=$(jq -r '.rejected_blocks // 0' <<< "$api_json")

    if [ "$diffTime" -lt "$maxDelay" ]; then
        total_hashrate_hs=$(jq -r '.total_hashrate_hs // 0' <<< "$api_json")
        total_hashrate=$(((10#${total_hashrate_hs:-0}) / 1000))

        declare -A api_khs_by_idx
        while IFS=$'\t' read -r idx khs_val; do
            [ -z "$idx" ] && continue
            api_khs_by_idx[$idx]=${khs_val:-0}
        done < <(jq -r '.devices[]? | .id as $id | (try ($id|capture("#(?<n>[0-9]+)")|.n) catch empty) as $n | select($n != "") | [$n, ((.hashrate_hs / 1000) | floor)] | @tsv' <<< "$api_json")

        # The miner numbers workers "Device #0..#K-1" over mining-brand GPUs only.
        # Keep a separate counter so indices stay aligned when non-mining GPUs exist.
        miner_dev=0
        for ((i = 0; i < gpu_count; i++)); do
            [[ "${brands[i]}" != "$BRAND_MINER" ]] && continue
            [[ "${busids[i]}" =~ ^([A-Fa-f0-9]+): ]]
            busid_arr+=($((16#${BASH_REMATCH[1]})))
            temp_arr+=("${temps[i]}")
            temp2_arr+=("${memtemps[i]:-0}")
            fan_arr+=("${fans[i]}")
            hashrate=${api_khs_by_idx[$miner_dev]:-0}
            hash_arr+=("${hashrate:-0}")
            miner_dev=$((miner_dev + 1))
        done

        hash_json=$(printf '%s\n' "${hash_arr[@]}" | jq -cs '.')
        bus_numbers=$(printf '%s\n' "${busid_arr[@]}" | jq -cs '.')
        fan_json=$(printf '%s\n' "${fan_arr[@]}" | jq -cs '.')
        temp_json=$(printf '%s\n' "${temp_arr[@]}" | jq -cs '.')
        temp2_json=$(printf '%s\n' "${temp2_arr[@]}" | jq -cs '.')

        uptime=$(($(date +%s) - $(stat -c %Y $CUSTOM_CONFIG_FILENAME)))

        stats=$(jq -nc \
            --argjson hs "$hash_json" \
            --arg ver "$CUSTOM_VERSION" \
            --arg ths "$total_hashrate" \
            --arg accepted "$accepted_blocks" \
            --arg rejected "$rejected_blocks" \
            --argjson bus_numbers "$bus_numbers" \
            --argjson fan "$fan_json" \
            --argjson temp "$temp_json" \
            --argjson mctemp "$temp2_json" \
            --arg uptime "$uptime" \
            '{ hs: $hs, hs_units: "khs", algo: "keryxhash", ver: $ver, accepted_blocks: ($accepted|tonumber), rejected_blocks: ($rejected|tonumber), ar: [($accepted|tonumber), ($rejected|tonumber)], $uptime, $bus_numbers, $temp, $mctemp, $fan }')
        khs=$total_hashrate
        stats_raw="api://v1/miner/stats total_hs=${total_hashrate_hs}"
    else
        khs=0
        stats="null"
    fi
else

    # The miner logs with env_logger, whose default line starts "[2026-06-24T19:11:32Z INFO ...]"
    # (ISO-8601 UTC, leading '['). Older builds logged "2026-06-24 19:11:32.000+02:00 [INFO ]".
    # Pull the timestamp anywhere on the line (bracket/position independent) and let GNU date parse it
    # (it understands both the T...Z form and the "date time+offset" form natively).
    ts_field=$(echo "$stats_raw" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]+)?(Z|[+-][0-9]{2}:?[0-9]{2})?' | head -1)
    time_rep=$(date -d "$ts_field" +%s 2> /dev/null || echo 0)
    diffTime=$(echo $((time_now - time_rep)) | tr -d '-')

    if [ "$diffTime" -lt "$maxDelay" ]; then
        accepted_blocks=$(grep -c "Block submitted successfully" <<< "$log")
        rejected_blocks=$(grep -E -c "Failed submitting block|Failed submitting PoM block" <<< "$log")
        # Value is second-to-last field (before unit), unit is last field.
        # The miner logs the rate with 2 decimals; dropping the dot then appending one 0 yields
        # rate*1000 (e.g. 3.83 -> "383" -> "3830"). NB: do NOT use `cut --output-delimiter=''` to
        # drop the dot — an empty output delimiter makes cut emit a NUL byte, which then trips
        # bash's "command substitution: ignored null byte" warning. `tr -d '.'` is clean.
        # HiveOS expects kilohashes (khs): Ghash/s = rate*1e6 khs = (rate*1000)*1e3, etc.
        total_hashrate=$(echo $stats_raw | awk 'NF>=2{print $(NF-1)}' | tr -d '.' | sed 's/$/0/')
        # Force base 10: a sub-1.0 rate yields a leading zero (e.g. 0.48 -> "0480") which bash would
        # otherwise parse as octal and reject ("value too great for base").
        total_hashrate=$((10#${total_hashrate:-0}))
        if [[ $stats_raw == *"Thash"* ]]; then
            total_hashrate=$(($total_hashrate * 1000000))
        elif [[ $stats_raw == *"Ghash"* ]]; then
            total_hashrate=$(($total_hashrate * 1000))
        elif [[ $stats_raw == *"Mhash"* ]]; then
            : # Mhash/s = rate*1e3 khs = rate*1000 already, no multiplier needed
        fi

        # The miner numbers its workers "Device #0..#K-1" over the GPUs IT enumerates (mining brand
        # only, PCI-bus order). HiveOS's busid list can also contain devices the miner never sees —
        # e.g. an onboard iGPU at bus 00:02.0. Using the raw loop index `i` as the miner device number
        # then desyncs the moment such a device is skipped (every later card reads one slot too high,
        # the last one falls off the end -> 0). Keep a SEPARATE counter that advances only for
        # mining-brand cards, so it tracks the miner's own numbering. No iGPU -> miner_dev == i.
        miner_dev=0
        for ((i = 0; i < gpu_count; i++)); do
            [[ "${brands[i]}" != "$BRAND_MINER" ]] && continue
            [[ "${busids[i]}" =~ ^([A-Fa-f0-9]+): ]]
            busid_arr+=($((16#${BASH_REMATCH[1]})))
            temp_arr+=("${temps[i]}")
            temp2_arr+=("${memtemps[i]:-0}")
            fan_arr+=("${fans[i]}")
            # Per-device line: "... Device #N (GPU name): 5.23 Ghash/s" — the worker id is
            # "#N (name)" so the colon is after the name, not after the number; match "#N" followed
            # by a space or colon (never "#N:" directly, which would also break for N=1 vs N=10).
            gpu_raw=$(grep -E "Device #${miner_dev}[ :]" <<< "$log" | tail -n 1)
            hashrate=$(echo $gpu_raw | awk 'NF>=2{print $(NF-1)}' | tr -d '.' | sed 's/$/0/')
            # Force base 10 (sub-1.0 rates yield a leading zero that bash would parse as octal).
            hashrate=$((10#${hashrate:-0}))
            if [[ $gpu_raw == *"Thash"* ]]; then
                hashrate=$(($hashrate * 1000000))
            elif [[ $gpu_raw == *"Ghash"* ]]; then
                hashrate=$(($hashrate * 1000))
            elif [[ $gpu_raw == *"Mhash"* ]]; then
                : # Mhash/s = rate*1e3 khs = rate*1000 already, no multiplier needed
            fi
            hash_arr+=("$hashrate")
            miner_dev=$((miner_dev + 1))
        done

        hash_json=$(printf '%s\n' "${hash_arr[@]}" | jq -cs '.')
        bus_numbers=$(printf '%s\n' "${busid_arr[@]}" | jq -cs '.')
        fan_json=$(printf '%s\n' "${fan_arr[@]}" | jq -cs '.')
        temp_json=$(printf '%s\n' "${temp_arr[@]}" | jq -cs '.')
        temp2_json=$(printf '%s\n' "${temp2_arr[@]}" | jq -cs '.')

        uptime=$(($(date +%s) - $(stat -c %Y $CUSTOM_CONFIG_FILENAME)))

        stats=$(jq -nc \
            --argjson hs "$hash_json" \
            --arg ver "$CUSTOM_VERSION" \
            --arg ths "$total_hashrate" \
            --arg accepted "$accepted_blocks" \
            --arg rejected "$rejected_blocks" \
            --argjson bus_numbers "$bus_numbers" \
            --argjson fan "$fan_json" \
            --argjson temp "$temp_json" \
            --argjson mctemp "$temp2_json" \
            --arg uptime "$uptime" \
            '{ hs: $hs, hs_units: "khs", algo: "keryxhash", ver: $ver, accepted_blocks: ($accepted|tonumber), rejected_blocks: ($rejected|tonumber), ar: [($accepted|tonumber), ($rejected|tonumber)], $uptime, $bus_numbers, $temp, $mctemp, $fan }')
        khs=$total_hashrate
    else
        khs=0
        stats="null"
    fi
fi

echo "Log file : $CUSTOM_LOG_BASENAME.log"
echo "Time since last log entry : $diffTime"
echo "Raw stats : $stats_raw"
echo "KHS : $khs"
echo "Output : $stats"

[[ -z $khs ]] && khs=0
[[ -z $stats ]] && stats="null"
