#!/bin/bash
# tune.sh - Profiles the GPU and sweeps the BWARP_TUNE_* parameters (src/cuda/tuning.h) on one file.
#
#   tools/tune.sh <file> [profile|sweep|all]          (default: all)
#
# Builds bwarp with -DBWARP_TUNING=ON into build-tune/, then:
#   profile  bwarp info, nvidia-smi, selftest, bench --context --pinned and ratio --gpu --verbose with the defaults
#   sweep    varies one parameter at a time from the defaults; each setting is benchmarked (bench --context
#            --pinned, best of RUNS) and its BWT checked (bwarp test) unless CHECK=0
#
# Environment:
#   RUNS=5           runs per bench (best of)
#   CHECK=1          verify the BWT of every setting with 'bwarp test' (0 to skip; it takes a while on enwik9)
#   CHECK_FILE=      a smaller file to verify with instead of <file>, e.g. enwik8
#   ONLY="A B"       sweep only these parameters (names without BWARP_TUNE_, e.g. "PARTITION_BITS MERGE_G")
#   BWARP_TUNE_*     set before running to sweep around other values than the defaults
#   BIN=             use this bwarp binary (built with BWARP_TUNING=ON) instead of building build-tune/
#
# Results go to tune-results/<host>-<time>/: everything in log.txt, the sweep in sweep.csv (times in ms),
# and a summary of the best value per parameter at the end. Run it on an otherwise idle GPU.
#
# Part of bwarp, a GPU-based BWT file compressor.
# Copyright (c) 2026 Alex Schumacher
# Licensed under the MIT License; see the LICENSE file for details.

set -u
cd "$(dirname "$0")/.."

if [ $# -lt 1 ] || [ ! -f "$1" ]; then
    sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

FILE=$(realpath "$1")
MODE=${2:-all}
RUNS=${RUNS:-5}
CHECK=${CHECK:-1}
CHECK_FILE=${CHECK_FILE:-$FILE}
ONLY=${ONLY:-}

# The 3090 box's quirks: a CUDA_VISIBLE_DEVICES that hides the GPU, and Ubuntu's older nvcc first on the PATH.
unset CUDA_VISIBLE_DEVICES
if [ -z "${CUDACXX:-}" ] && [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDACXX=/usr/local/cuda/bin/nvcc
fi
if ! command -v nvidia-smi >/dev/null 2>&1 && [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
    export PATH="$PATH:/usr/lib/wsl/lib"
fi

OUT="tune-results/$(hostname)-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/log.txt"
CSV="$OUT/sweep.csv"
BUILD=1
if [ -n "${BIN:-}" ]; then
    BUILD=0
else
    BIN=build-tune/bin/Release/bwarp
fi

log() {
    echo "$@" | tee -a "$LOG"
}

# Runs a command, its output to the log and the screen.
run() {
    log "\$ $*"
    "$@" 2>&1 | tee -a "$LOG"
    return "${PIPESTATUS[0]}"
}

log "tune.sh on $(hostname), $(date), git $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
log "file $FILE ($(stat -c %s "$FILE") bytes), mode $MODE, runs $RUNS, check $CHECK ($CHECK_FILE)"
env | grep '^BWARP_TUNE_' | tee -a "$LOG"

if [ "$BUILD" = 1 ]; then
    log ""
    log "== build (BWARP_TUNING=ON, build-tune/)"
    if ! cmake --preset linux-cuda -B build-tune -DBWARP_TUNING=ON >>"$LOG" 2>&1 || ! cmake --build build-tune >>"$LOG" 2>&1; then
        log "build failed, see $LOG"
        tail -n 30 "$LOG"
        exit 1
    fi
fi

gpuLoad() {
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1
}

# nvidia-smi averages utilization over its last sample period, so wait for the previous run to drop out.
warnIfBusy() {
    local load
    sleep 2
    load=$(gpuLoad)
    if [ -n "$load" ] && [ "${load%%,*}" -gt 5 ] 2>/dev/null; then
        log "WARNING: GPU utilization ${load%%,*}% (memory ${load##*, } MiB) before this run; timings may be off"
    fi
}

profile() {
    log ""
    log "== profile"
    run "$BIN" info
    run nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit,temperature.gpu,utilization.gpu,memory.used --format=csv
    run "$BIN" selftest || log "SELFTEST FAILED"
    warnIfBusy
    run "$BIN" bench "$FILE" --context --pinned --runs "$RUNS"
    warnIfBusy
    run "$BIN" ratio "$FILE" --gpu --verbose
}

# The bench row's columns, from the end: ... setup upload rank class merge bwt down total MB/s peak/n
COLUMNS_CSV="setup,upload,rank,class,merge,bwt,down,total,mbps,peak"

# One setting: NAME=VALUE (or several, space-separated) on top of the environment. Appends a CSV row.
measure() {
    local label=$1
    shift
    local row check="skipped"

    warnIfBusy
    row=$(env "$@" "$BIN" bench "$FILE" --context --pinned --runs "$RUNS" 2>>"$LOG" | tee -a "$LOG" | tail -n1)

    if [ "$CHECK" = 1 ]; then
        if env "$@" "$BIN" test "$CHECK_FILE" >>"$LOG" 2>&1; then
            check="ok"
        else
            check="FAILED"
        fi
    fi

    local values
    values=$(echo "$row" | awk '{ printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s", $(NF-9), $(NF-8), $(NF-7), $(NF-6), $(NF-5), $(NF-4), $(NF-3), $(NF-2), $(NF-1), $NF }')
    echo "$label,\"$*\",$values,$check" >>"$CSV"
    echo "$values" | awk -F, -v label="$label" -v setting="$*" -v check="$check" \
        '{ printf "  %-15s %-60s rank %6s  class %6s  merge %6s  total %7s ms  %s\n", label, setting, $3, $4, $5, $8, check }' \
        | tee -a "$LOG"
}

wanted() {
    [ -z "$ONLY" ] || [[ " $ONLY " == *" $1 "* ]]
}

# sweepOne NAME VALUE... : BWARP_TUNE_NAME at each value.
sweepOne() {
    local name=$1
    shift
    wanted "$name" || return 0
    log ""
    log "-- $name"
    for value in "$@"; do
        measure "$name" "BWARP_TUNE_$name=$value"
    done
}

sweep() {
    log ""
    log "== sweep (best of $RUNS runs per setting)"
    echo "parameter,setting,$COLUMNS_CSV,check" >"$CSV"

    measure baseline

    sweepOne PARTITION_THRESHOLD 262144 524288 1048576 2097152 4194304 8388608 16777216 4294967295
    sweepOne PARTITION_BITS 4 5 6 7 8 9 10 11 12
    sweepOne THREADS 128 256 512 1024
    sweepOne MERGE_G 32 64 96 128 192 256 512
    sweepOne MERGE_TILE 128 192 256 320 384 448
    sweepOne MERGE_LEVELS 0 1 2

    # Capacity and threads together (capacity <= 2 * threads).
    if wanted MERGE_CAPACITY; then
        log ""
        log "-- MERGE_CAPACITY:MERGE_THREADS"
        for pair in 256:128 384:192 480:256 512:256 640:320 768:384 960:512; do
            measure MERGE_CAPACITY "BWARP_TUNE_MERGE_CAPACITY=${pair%%:*}" "BWARP_TUNE_MERGE_THREADS=${pair##*:}"
        done
    fi

    if wanted MERGE_OVERFLOW; then
        log ""
        log "-- MERGE_OVERFLOW_CAPACITY:MERGE_OVERFLOW_THREADS"
        for pair in 640:320 768:384 960:512 1280:640; do  # 1024 threads: too many registers
            measure MERGE_OVERFLOW "BWARP_TUNE_MERGE_OVERFLOW_CAPACITY=${pair%%:*}" \
                "BWARP_TUNE_MERGE_OVERFLOW_THREADS=${pair##*:}"
        done
    fi

    sweepOne STAGING_MB 2 4 8 16 32 64

    measure baseline  # again, to see the drift over the sweep

    log ""
    log "== summary: the fastest setting per parameter by total ms (rank/class/merge ms), against the baselines"
    awk -F, 'NR > 1 {
        p = $1; total = $10 + 0; phases = sprintf("%s/%s/%s", $5, $6, $7)
        if (p == "baseline") { base = base sprintf("  %.1f (%s)", total, phases); next }
        if (!(p in best) || total < best[p]) { best[p] = total; bestPhases[p] = phases; setting[p] = $2 }
        if ($NF != "ok" && $NF != "skipped") failed = failed "\n  CHECK FAILED: " $1 " " $2
    } END {
        printf "  baseline:%s\n", base
        for (p in best) printf "  %-22s %8.1f (%s)  %s\n", p, best[p], bestPhases[p], setting[p]
        if (failed != "") print failed
    }' "$CSV" | tee -a "$LOG"
}

case "$MODE" in
    profile) profile ;;
    sweep) sweep ;;
    all) profile; sweep ;;
    *) echo "mode must be profile, sweep or all"; exit 1 ;;
esac

log ""
log "done: $OUT (log.txt, sweep.csv)"
