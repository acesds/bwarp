#!/bin/bash
# tune.sh - Tunes the BWARP_TUNE_* parameters (src/cuda/tuning.h) for this machine's GPU, then benchmarks
# enwik8 and enwik9 with the default and the tuned parameters.
#
#   tools/tune.sh [all|tune|bench]          (default: all)
#
# Expects enwik8 and enwik9 in the repository root. Builds bwarp with -DBWARP_TUNING=ON into build-tune/, then:
#   tune   Tunes on enwik9, one parameter at a time: every candidate value is timed with bench --context --pinned
#          (the BWT construction, best of TUNE_RUNS), and the fastest is kept if it beats the current value by
#          more than MIN_GAIN percent. Repeated for PASSES passes, or until a pass changes nothing. The result
#          goes to best.env and is checked with selftest.
#   bench  For enwik8 and enwik9, with the default and the tuned parameters: bench --context --pinned (the BWT
#          phases), and as in tools/ltcb.ps1, the wall clock of compress --segment 16 --slice 16 and of
#          decompress, best of RUNS, with each round trip checked byte for byte. Alone, "bench" takes the tuned
#          parameters from BWARP_TUNE_* in the environment (source a best.env first).
#
# Environment:
#   RUNS=3          runs per file of the final compress and decompress (best of)
#   TUNE_RUNS=5     runs per candidate while tuning (best of)
#   PASSES=2        passes over the parameters
#   MIN_GAIN=0.5    percent a value must gain to replace the current one (the noise floor)
#   BIN=            use this bwarp binary (built with BWARP_TUNING=ON) instead of building build-tune/
#   DATA=           look for enwik8 and enwik9 here instead of the repository root
#   ONLY="A B"      tune only these parameter groups, e.g. "PARTITION_BITS MERGE_CAPACITY:MERGE_THREADS"
#
# Results go to tune-results/<host>-<time>/: log.txt (everything), tune.csv (every candidate), best.env and
# report.txt (the final tables). Run it on an otherwise idle GPU.
#
# Part of bwarp, a GPU-based BWT file compressor.
# Copyright (c) 2026 Alex Schumacher
# Licensed under the MIT License; see the LICENSE file for details.

set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)

MODE=${1:-all}
RUNS=${RUNS:-3}
TUNE_RUNS=${TUNE_RUNS:-5}
PASSES=${PASSES:-2}
MIN_GAIN=${MIN_GAIN:-0.5}
DATA=${DATA:-$ROOT}
ONLY=${ONLY:-}
LTCB_OPTIONS=(--segment 16 --slice 16)

case "$MODE" in
    all | tune | bench) ;;
    *)
        sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac

# The input: DATA/name, or DATA/name/name.
findInput() {
    local candidate
    for candidate in "$DATA/$1" "$DATA/$1/$1"; do
        if [ -f "$candidate" ]; then
            realpath "$candidate"
            return 0
        fi
    done
    echo "tune.sh: $1 not found in $DATA (download it into the repository root)" >&2
    return 1
}

ENWIK8=$(findInput enwik8) || exit 1
ENWIK9=$(findInput enwik9) || exit 1

# The 3090 box's quirks: a CUDA_VISIBLE_DEVICES that hides the GPU, and Ubuntu's older nvcc first on the PATH.
unset CUDA_VISIBLE_DEVICES
if [ -z "${CUDACXX:-}" ] && [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDACXX=/usr/local/cuda/bin/nvcc
fi
if ! command -v nvidia-smi >/dev/null 2>&1 && [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
    export PATH="$PATH:/usr/lib/wsl/lib"
fi

OUT="tune-results/$(hostname)-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT/work"
LOG="$OUT/log.txt"
CSV="$OUT/tune.csv"
REPORT="$OUT/report.txt"
START=$(date +%s)

# ---- output --------------------------------------------------------------------------------------

clock() {
    local s=$(($(date +%s) - START))
    printf '%dh%02dm%02ds' $((s / 3600)) $((s / 60 % 60)) $((s % 60))
}

# A progress line: on the screen and in the log.
say() {
    printf '[%s] %s\n' "$(clock)" "$*" | tee -a "$LOG"
}

section() {
    echo | tee -a "$LOG"
    say "== $*"
}

# Runs a command, its output into the log only.
quiet() {
    echo "\$ $*" >>"$LOG"
    "$@" >>"$LOG" 2>&1
}

durationText() {
    printf '%dm%02ds' $(($1 / 60)) $(($1 % 60))
}

thousands() {
    echo "$1" | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta'
}

gpuLoad() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1
}

# nvidia-smi averages utilization over its last sample period, so this waits for the previous run to drop out.
warnIfBusy() {
    local load
    sleep 1
    load=$(gpuLoad)
    if [ -n "$load" ] && [ "$load" -gt 5 ] 2>/dev/null; then
        say "  WARNING: GPU utilization $load% before this run (someone else using the GPU?); timings may be off"
    fi
}

# ---- parameters ----------------------------------------------------------------------------------

# The defaults of tuning.h; the starting point, unless BWARP_TUNE_* in the environment says otherwise.
declare -A DEFAULT=(
    [PARTITION_THRESHOLD]=4194304 [PARTITION_BITS]=8 [THREADS]=256 [MERGE_G]=128 [MERGE_TILE]=256
    [MERGE_CAPACITY]=480 [MERGE_THREADS]=256 [MERGE_OVERFLOW_CAPACITY]=960 [MERGE_OVERFLOW_THREADS]=512
    [MERGE_LEVELS]=1 [STAGING_MB]=8
)
ORDER="PARTITION_THRESHOLD PARTITION_BITS THREADS MERGE_G MERGE_TILE MERGE_CAPACITY MERGE_THREADS
       MERGE_OVERFLOW_CAPACITY MERGE_OVERFLOW_THREADS MERGE_LEVELS STAGING_MB"

declare -A CUR
for name in $ORDER; do
    var="BWARP_TUNE_$name"
    CUR[$name]=${!var:-${DEFAULT[$name]}}
    unset "$var"  # from here on, every run gets its parameters explicitly
done

# What tune() tries: NAME[:NAME]|candidates. Pairs move together (merge capacity <= 2 * threads). The overflow
# merge can't launch with 1024 threads (too many registers).
PARAM_GROUPS=(
    "PARTITION_THRESHOLD|262144 524288 1048576 2097152 4194304 8388608 16777216 4294967295"
    "PARTITION_BITS|4 5 6 7 8 9 10 11 12"
    "THREADS|128 256 512 1024"
    "MERGE_G|32 64 96 128 192 256 512"
    "MERGE_TILE|128 192 256 320 384 448"
    "MERGE_CAPACITY:MERGE_THREADS|256:128 384:192 480:256 512:256 640:320 768:384 960:512"
    "MERGE_OVERFLOW_CAPACITY:MERGE_OVERFLOW_THREADS|640:320 768:384 960:512 1280:640"
    "MERGE_LEVELS|0 1 2"
    "STAGING_MB|2 4 8 16 32 64"
)

if [ -n "$ONLY" ]; then
    selected=()
    for group in "${PARAM_GROUPS[@]}"; do
        [[ " $ONLY " == *" ${group%%|*} "* ]] && selected+=("$group")
    done
    PARAM_GROUPS=("${selected[@]}")
fi

# ENV: the current parameters as NAME=VALUE arguments for env.
setCurrentEnv() {
    ENV=()
    local name
    for name in $ORDER; do
        ENV+=("BWARP_TUNE_$name=${CUR[$name]}")
    done
}

# ENV: the defaults.
setDefaultEnv() {
    ENV=()
    local name
    for name in $ORDER; do
        ENV+=("BWARP_TUNE_$name=${DEFAULT[$name]}")
    done
}

# The parameters that differ from the defaults, or "defaults".
changedText() {
    local name text=""
    for name in $ORDER; do
        if [ "${CUR[$name]}" != "${DEFAULT[$name]}" ]; then
            text="$text $name=${CUR[$name]}"
        fi
    done
    echo "${text:- defaults}" | sed 's/^ //'
}

# BENCH_ROW: the bench row of file with ENV plus overrides; returns 1 if bwarp fails.
# Its columns, from the end: setup upload rank class merge bwt down total MB/s peak/n.
benchRow() {
    local file=$1 runs=$2
    shift 2
    local tmp="$OUT/work/bench.txt"
    echo "\$ env $* bwarp bench $file --context --pinned --runs $runs" >>"$LOG"
    env "${ENV[@]}" "$@" "$BIN" bench "$file" --context --pinned --runs "$runs" >"$tmp" 2>>"$LOG"
    local status=$?
    cat "$tmp" >>"$LOG"
    BENCH_ROW=$(tail -n1 "$tmp")
    [ $status = 0 ] && echo "$BENCH_ROW" | awk '{ exit !(NF >= 12 && $(NF-2) ~ /^[0-9.]+$/) }'
}

column() {  # column N-from-the-end of BENCH_ROW (0 = the last)
    echo "$BENCH_ROW" | awk -v k="$1" '{ print $(NF-k) }'
}

# OVERRIDES: BWARP_TUNE_ arguments setting the names (NAME[:NAME]) to the values (VALUE[:VALUE]).
setOverrides() {
    local -a nameList valueList
    IFS=: read -ra nameList <<<"$1"
    IFS=: read -ra valueList <<<"$2"
    OVERRIDES=()
    for i in "${!nameList[@]}"; do
        OVERRIDES+=("BWARP_TUNE_${nameList[$i]}=${valueList[$i]}")
    done
}

# Whether time a beats time b by more than MIN_GAIN percent.
beats() {
    awk -v a="$1" -v b="$2" -v g="$MIN_GAIN" 'BEGIN { exit !(a < b * (1 - g / 100)) }'
}

minimum() {
    awk -v a="$1" -v b="$2" 'BEGIN { print (a < b ? a : b) }'
}

# ---- build and profile -------------------------------------------------------------------------

say "tune.sh, mode $MODE, on $(hostname); git $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD)); results in $OUT"
say "enwik8: $ENWIK8; enwik9: $ENWIK9"

if [ -z "${BIN:-}" ]; then
    section "build (BWARP_TUNING=ON, build-tune/)"
    if ! quiet cmake --preset linux-cuda -B build-tune -DBWARP_TUNING=ON || ! quiet cmake --build build-tune; then
        say "build failed:"
        tail -n 30 "$LOG"
        exit 1
    fi
    BIN=build-tune/bin/Release/bwarp
fi

section "GPU"
"$BIN" info 2>/dev/null | tee -a "$LOG"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit,temperature.gpu,utilization.gpu,memory.used \
    --format=csv 2>/dev/null | tee -a "$LOG"
GPU_TEXT=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n1)

say "selftest with the defaults..."
setDefaultEnv
if ! quiet env "${ENV[@]}" "$BIN" selftest; then
    say "selftest FAILED with the defaults, see $LOG"
    exit 1
fi
say "selftest passed"

# ---- tune ----------------------------------------------------------------------------------------

tune() {
    local total=0 step=0 tuneStart pass group names values current candidates value changed
    local -a nameList valueList

    for group in "${PARAM_GROUPS[@]}"; do
        total=$((total + $(echo "${group#*|}" | wc -w)))
    done
    total=$((total * PASSES))

    section "tune on enwik9: $PASSES passes over ${#PARAM_GROUPS[@]} parameters, up to $total runs of bench (best of $TUNE_RUNS)"
    say "start: $(changedText)"
    echo "pass,parameter,value,setup,upload,rank,class,merge,bwt,down,total,result" >"$CSV"
    tuneStart=$(date +%s)

    for ((pass = 1; pass <= PASSES; pass++)); do
        changed=0

        for group in "${PARAM_GROUPS[@]}"; do
            names=${group%%|*}
            values=${group#*|}
            IFS=: read -ra nameList <<<"$names"
            current=""
            for name in "${nameList[@]}"; do
                current="$current${current:+:}${CUR[$name]}"
            done
            candidates=$values
            [[ " $values " == *" $current "* ]] || { candidates="$current $values"; total=$((total + 1)); }

            say "-- pass $pass/$PASSES: $names (now $current)"
            local best="" bestValue="" currentTotal=""

            for value in $candidates; do
                step=$((step + 1))
                setOverrides "$names" "$value"

                local elapsed=$(($(date +%s) - tuneStart)) eta="?"
                if [ $step -gt 1 ]; then
                    eta="~$(durationText $((elapsed * (total - step + 1) / (step - 1))))"
                fi

                warnIfBusy
                setCurrentEnv
                local result
                if benchRow "$ENWIK9" "$TUNE_RUNS" "${OVERRIDES[@]}"; then
                    result=$(column 2)
                    echo "$pass,$names,$value,$(echo "$BENCH_ROW" | awk '{ OFS=","; print $(NF-9),$(NF-8),$(NF-7),$(NF-6),$(NF-5),$(NF-4),$(NF-3),$(NF-2) }'),ok" >>"$CSV"
                    if [ -z "$best" ] || awk -v a="$result" -v b="$best" 'BEGIN { exit !(a < b) }'; then
                        best=$result
                        bestValue=$value
                    fi
                    [ "$value" = "$current" ] && currentTotal=$result
                    result="$result ms (rank $(column 7), class $(column 6), merge $(column 5))"
                else
                    echo "$pass,$names,$value,,,,,,,,,failed" >>"$CSV"
                    result="failed (see log)"
                fi

                say "   [$step/$total, $eta left] $names=$value: $result"
            done

            # One slow measurement of the current value shouldn't decide: measure both again, back to back,
            # and compare the best of each.
            if [ -n "$currentTotal" ] && [ -n "$bestValue" ] && [ "$bestValue" != "$current" ] &&
                beats "$best" "$currentTotal"; then
                say "   confirming $names=$bestValue ($best ms) against $current ($currentTotal ms)..."
                setCurrentEnv
                setOverrides "$names" "$current"
                if benchRow "$ENWIK9" "$TUNE_RUNS" "${OVERRIDES[@]}"; then
                    currentTotal=$(minimum "$currentTotal" "$(column 2)")
                fi
                setOverrides "$names" "$bestValue"
                if benchRow "$ENWIK9" "$TUNE_RUNS" "${OVERRIDES[@]}"; then
                    best=$(minimum "$best" "$(column 2)")
                fi
            fi

            if [ -n "$currentTotal" ] && [ -n "$bestValue" ] && [ "$bestValue" != "$current" ] &&
                beats "$best" "$currentTotal"; then
                IFS=: read -ra valueList <<<"$bestValue"
                for i in "${!nameList[@]}"; do
                    CUR[${nameList[$i]}]=${valueList[$i]}
                done
                changed=1
                say "   -> $names=$bestValue: $best ms instead of $currentTotal ms"
            else
                say "   -> keeping $names=$current (best $bestValue: $best ms, current $currentTotal ms)"
            fi
        done

        say "after pass $pass: $(changedText)"

        if [ $changed = 0 ]; then
            say "pass $pass changed nothing, done tuning"
            break
        fi
    done

    for name in $ORDER; do
        echo "export BWARP_TUNE_$name=${CUR[$name]}"
    done >"$OUT/best.env"
    say "tuned parameters in $OUT/best.env: $(changedText)"

    say "selftest with the tuned parameters..."
    setCurrentEnv
    if ! quiet env "${ENV[@]}" "$BIN" selftest; then
        say "selftest FAILED with the tuned parameters, see $LOG; benchmarking the defaults only"
        for name in $ORDER; do
            CUR[$name]=${DEFAULT[$name]}
        done
    else
        say "selftest passed"
    fi
}

# ---- benchmark -----------------------------------------------------------------------------------

declare -a RESULTS PHASES

# The best wall-clock seconds of RUNS runs of bwarp with ENV; label for the progress lines.
timeBest() {
    local label=$1
    shift
    local best="" r t0 t1 seconds
    for ((r = 1; r <= RUNS; r++)); do
        say "   $label, run $r/$RUNS..."
        echo "\$ env ${ENV[*]} bwarp $*" >>"$LOG"
        t0=$(date +%s%N)
        if ! env "${ENV[@]}" "$BIN" "$@" >>"$LOG" 2>&1; then
            say "   bwarp $1 FAILED, see $LOG"
            return 1
        fi
        t1=$(date +%s%N)
        seconds=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", (b - a) / 1e9 }')
        if [ -z "$best" ] || awk -v a="$seconds" -v b="$best" 'BEGIN { exit !(a < b) }'; then
            best=$seconds
        fi
    done
    TIME_BEST=$best
}

# One file with ENV: adds a row to RESULTS and PHASES.
benchFile() {
    local name=$1 file=$2 params=$3
    local bytes packed="$OUT/work/$name.bwarp" unpacked="$OUT/work/$name.out" compress decompress size check

    bytes=$(stat -c %s "$file")
    say "-- $name, $params parameters"
    say "   bench (BWT construction, best of 5)..."
    warnIfBusy
    if benchRow "$file" 5; then
        PHASES+=("$(printf '%-8s %-8s %8s %8s %8s %8s %8s %9s' "$name" "$params" "$(column 8)" "$(column 7)" \
            "$(column 6)" "$(column 5)" "$(column 3)" "$(column 2)")")
    else
        PHASES+=("$(printf '%-8s %-8s FAILED' "$name" "$params")")
    fi

    timeBest "compress" compress "$file" "$packed" --force --quiet "${LTCB_OPTIONS[@]}" || return 1
    compress=$TIME_BEST
    timeBest "decompress" decompress "$packed" "$unpacked" --force --quiet || return 1
    decompress=$TIME_BEST
    size=$(stat -c %s "$packed")
    say "   checking the round trip..."
    if cmp -s "$file" "$unpacked"; then check="ok"; else check="MISMATCH"; fi
    rm -f "$packed" "$unpacked"

    RESULTS+=("$(awk -v n="$name" -v p="$params" -v size="$(thousands "$size")" -v c="$compress" -v d="$decompress" \
        -v bytes="$bytes" -v check="$check" 'BEGIN {
            printf "%-8s %-8s %13s %9.3f %9.2f %9.3f %9.2f   %s", n, p, size, c, 1e9 * c / bytes, d, 1e9 * d / bytes, check
        }')")
    say "   $name, $params: compress ${compress}s, decompress ${decompress}s, $(thousands "$size") bytes, round trip $check"
}

benchAll() {
    local tuned
    tuned=$(changedText)
    section "benchmark: enwik8 and enwik9, defaults and tuned ($tuned); compress and decompress best of $RUNS"

    for name in enwik8 enwik9; do
        local file=$ENWIK8
        [ $name = enwik9 ] && file=$ENWIK9
        setDefaultEnv
        benchFile $name "$file" default
        if [ "$tuned" != "defaults" ]; then
            setCurrentEnv
            benchFile $name "$file" tuned
        fi
    done

    {
        echo
        echo "== Results on ${GPU_TEXT:-the GPU}, $(hostname), git $(git rev-parse --short HEAD), $(date '+%Y-%m-%d %H:%M')"
        echo
        echo "Tuned parameters: $tuned"
        echo
        echo "compress ${LTCB_OPTIONS[*]} and decompress: wall clock of the whole process, best of $RUNS runs,"
        echo "input read warm from the file cache, round trips checked byte for byte (as tools/ltcb.ps1)"
        echo
        printf '%-8s %-8s %13s %9s %9s %9s %9s   %s\n' file params compressed "comp s" "comp ns/B" "dec s" "dec ns/B" check
        printf '%s\n' "${RESULTS[@]}"
        echo
        echo "BWT construction: bench --context --pinned, best of 5 runs, in ms"
        echo
        printf '%-8s %-8s %8s %8s %8s %8s %8s %9s\n' file params upload rank class merge down total
        printf '%s\n' "${PHASES[@]}"
    } | tee -a "$LOG" >"$REPORT"
    cat "$REPORT"
}

# ---- main ----------------------------------------------------------------------------------------

case "$MODE" in
    all) tune; benchAll ;;
    tune) tune ;;
    bench) benchAll ;;
esac

rm -rf "$OUT/work"
echo
say "done in $(clock): $OUT"
