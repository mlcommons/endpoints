#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail
root=$1
generation=$2
secret=$3
shift 3
interpreter=("$@")

atomic_write() {
    path=$1
    value=$2
    temporary="${path}.tmp.$$"
    printf '%s\n' "$value" > "$temporary" || exit 70
    mv -f -- "$temporary" "$path" || exit 70
}

mkdir -p "$root/requests" || exit 70
atomic_write "$root/server_status" started
atomic_write "$root/ready" "$generation"

while :; do
    if [ -f "$root/stop" ]; then
        atomic_write "$root/server_status" stopped
        exit 0
    fi
    handled=0
    for request in "$root"/requests/*; do
        [ -f "$root/stop" ] && exit 0
        [ -d "$request" ] || continue
        mkdir "$request/claim" 2>/dev/null || continue
        status=$(cat "$request/status" 2>/dev/null || printf unknown)
        if [ "$status" != pending ]; then
            rmdir "$request/claim" 2>/dev/null || true
            continue
        fi
        handled=1
        atomic_write "$request/status" started
        timeout_s=$(cat "$request/timeout" 2>/dev/null || printf invalid)
        case "$timeout_s" in
            ''|*[!0-9]*) exit 70 ;;
            *)
                cwd=$(cat "$request/cwd" && printf x) || exit 70
                cwd=${cwd%x}
                command=$(cat "$request/command" && printf x) || exit 70
                command=${command%x}
                (
                    cd -- "$cwd" || exit 125
                    unshare --pid --fork --mount-proc \
                        timeout -k 5 "$timeout_s" bash -c '
                            status=$1; shift
                            "$@"
                            rc=$?
                            printf "%s\n" "$rc" > "$status"
                            exit "$rc"
                        ' command-status "$request/command_status" "${interpreter[@]}" "$command"
                ) > "$request/stdout.tmp" 2>&1
                returncode=$?
                : > "$request/stderr.tmp"
                timed_out=0
                if [ "$(cat "$request/command_status" 2>/dev/null)" != "$returncode" ]; then
                    case "$returncode" in 124|137) timed_out=1 ;; *) exit 70 ;; esac
                fi
                ;;
        esac
        [ -f "$request/stdout.tmp" ] || : > "$request/stdout.tmp"
        [ -f "$request/stderr.tmp" ] || : > "$request/stderr.tmp"
        mv -f -- "$request/stdout.tmp" "$request/stdout" || exit 70
        mv -f -- "$request/stderr.tmp" "$request/stderr" || exit 70
        stdout_size=$(wc -c < "$request/stdout") || exit 70
        stderr_size=$(wc -c < "$request/stderr") || exit 70
        stdout_size=$((stdout_size))
        stderr_size=$((stderr_size))
        nonce=${request##*/}
        digest=$(
            {
                printf '%s\0%s\0%s\0%s\0%s\0%s\0' \
                    "$secret" "$nonce" "$returncode" "$stdout_size" \
                    "$stderr_size" "$timed_out"
                cat "$request/stdout"
                printf '\0'
                cat "$request/stderr"
                printf '\0%s' "$secret"
            } | sha256sum
        ) || exit 70
        digest=${digest%% *}
        atomic_write "$request/status" "finished:$returncode"
        atomic_write "$request/complete" \
            "$returncode $stdout_size $stderr_size $timed_out $digest"
    done
    [ "$handled" -eq 1 ] || sleep 0.05
done
