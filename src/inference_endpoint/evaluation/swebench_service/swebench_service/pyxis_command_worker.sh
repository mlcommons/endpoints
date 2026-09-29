#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail
root=$1
shift
interpreter=("$@")
request="$root/running"
touch "$root/ready" || exit 70

while [ ! -e "$root/stop" ]; do
    if [ ! -d "$root/request" ]; then
        sleep 0.05
        continue
    fi
    # The host serializes callers and removes the previous result before publishing.
    [ ! -e "$request" ] || exit 70
    mv -- "$root/request" "$request" || exit 70
    timeout_s=$(cat "$request/timeout") || exit 70
    case "$timeout_s" in ''|*[!0-9]*|0) exit 70 ;; esac
    # A sentinel preserves trailing newlines in the command and working directory.
    cwd=$(cat "$request/cwd" && printf x) || exit 70
    command=$(cat "$request/command" && printf x) || exit 70
    unshare --pid --fork --mount-proc \
        timeout -k 5 "$timeout_s" bash -c '
            status=$1; cwd=$2; shift 2
            if cd -- "$cwd"; then
                "$@"
                rc=$?
            else
                rc=125
            fi
            printf "%s\n" "$rc" > "$status" || exit 70
            exit "$rc"
        ' command-status "$request/command_status" "${cwd%x}" \
        "${interpreter[@]}" "${command%x}" > "$request/output" 2>&1
    returncode=$?
    timed_out=0
    # Explicit exits 124/137 are command results, not timeout notifications.
    if [ "$(cat "$request/command_status" 2>/dev/null)" != "$returncode" ]; then
        case "$returncode" in 124|137) timed_out=1 ;; *) exit 70 ;; esac
    fi
    size=$(wc -c < "$request/output") || exit 70
    printf '%s %s %s\n' "$returncode" "$timed_out" "$((size))" > "$request/complete.tmp" || exit 70
    mv -- "$request/complete.tmp" "$request/complete" || exit 70
done
