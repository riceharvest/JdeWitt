#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# P7 certification waves for qwen3-4b f15s15-l935 (REFIT) at the pinned R9 config
# `-c 4096 --parallel 4`, temperature 0, recorded cap 1024 (one slot's worth of the
# 4096 context, so nothing truncates at 1024/slot).
#
# One script serves each artifact at that argv and runs every wave the P7 spec lists,
# in order, then stops every server it started. It is resumable: a wave whose
# wave.json records `measured: true` is skipped, so a killed run continues where it
# stopped instead of re-serving finished decks.
#
#   verification/p7-q3-4b/run_waves.sh --dry     # print the argv list, run nothing
#   verification/p7-q3-4b/run_waves.sh           # run / resume
#
# Overridable (environment):
#   JDEWITT_MODELS      artifact directory          default /mnt/ssd/hermes/models
#   JDEWITT_SERVER      llama-server binary         default the local CPU build
#   JDEWITT_PERPLEXITY  llama-perplexity binary     default the local CPU build
#   JDEWITT_PORT        serving port                default 8134
#   JDEWITT_NGL         GPU layers for the launch   default 99 (a CPU-only build
#                       ignores it; the recorded serve_config carries the token)
#   JDEWITT_ARM_AXES    refit axes .pt              default $JDEWITT_MODELS/_fit/
#                                                   q3-shared-axes-refit.pt
#   JDEWITT_OUT         output root                 default this directory
#   JDEWITT_CTX         context                     default 4096
#   JDEWITT_PARALLEL    slots                       default 4
#   JDEWITT_CAP         per-request tokens          default 1024 (== ctx/parallel)
#
# The R9.4 config-sensitivity re-run uses the same script with a different output
# root and a different ctx, which is the ONLY knob that differs from the pinned run:
#   JDEWITT_OUT=verification/p7-q3-4b/sensitivity-8192 JDEWITT_CTX=8192 \
#       verification/p7-q3-4b/run_waves.sh
#
# The recorded serve_config IS the launch argv plus the per-request `--max-tokens`
# token: llama-server takes the cap per request, so the token is stripped before the
# server starts and passed to scripts/capability_wave.py, and `release-certify` reads
# it back out of the recorded string. One string, no second config to drift from.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${JDEWITT_OUT:-$HERE}"
MODELS="${JDEWITT_MODELS:-/mnt/ssd/hermes/models}"
SERVER="${JDEWITT_SERVER:-/home/dario/llama.cpp-spark/build-cpu/bin/llama-server}"
PERPLEXITY="${JDEWITT_PERPLEXITY:-/home/dario/llama.cpp-spark/build-cpu/bin/llama-perplexity}"
PORT="${JDEWITT_PORT:-8134}"
NGL="${JDEWITT_NGL:-99}"
AXES="${JDEWITT_ARM_AXES:-$MODELS/_fit/q3-shared-axes-refit.pt}"

CTX="${JDEWITT_CTX:-4096}"
PARALLEL="${JDEWITT_PARALLEL:-4}"
CAP="${JDEWITT_CAP:-1024}"
TEMPERATURE=0

ARM="$MODELS/q3-4b-f15s15-l935-refit.gguf"
BASE="$MODELS/q3-4b-base-q8.gguf"
PUBLISHED="$MODELS/q3-4b-heretic-q8.gguf"

REMOVAL_SUITE="verification/p7-q3-4b/harmful-40.jsonl"
HELD_OUT_SUITE="verification/screens/q3-holdout-wider.jsonl"
RETENTION_SUITE="verification/capability/capability-run-en.jsonl"

DRY=0
for arg in "$@"; do
    case "$arg" in
        --dry) DRY=1 ;;
        -h|--help) sed -n '4,45p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $arg (only --dry is accepted)" >&2; exit 2 ;;
    esac
done

log() {
    local line
    line="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
    echo "$line"
    if [ "$DRY" -eq 0 ]; then printf '%s\n' "$line" >> "$OUT/progress.txt"; fi
}

serve_config() {  # $1 = artifact path; the exact string spec.json and the waves record
    printf -- '-m %s -c %s --parallel %s --jinja --no-webui --host 127.0.0.1 --port %s -ngl %s --max-tokens %s' \
        "$1" "$CTX" "$PARALLEL" "$PORT" "$NGL" "$CAP"
}

launch_argv() {  # serve_config minus the per-request cap token
    local config="$1" token pending=0
    for token in $config; do
        if [ "$pending" -eq 1 ]; then pending=0; continue; fi
        if [ "$token" = "--max-tokens" ]; then pending=1; continue; fi
        printf '%s\n' "$token"
    done
}

wave_argv() {  # $1 = wave name, $2 = artifact, $3 = suite, $4 = kinds (for --dry only)
    printf -- 'uv run python scripts/capability_wave.py %s %s/waves/%s --label %s --port %s --max-tokens %s --parallel %s --kinds %s --serve-config "%s" --temperature %s --preview 2' \
        "$3" "$OUT" "$1" "$1" "$PORT" "$CAP" "$PARALLEL" "$4" "$(serve_config "$2")" "$TEMPERATURE"
}

wave_done() {  # $1 = wave name; true iff this wave is finished and measured
    local dir="$OUT/waves/$1"
    [ -f "$dir/wave.json" ] && [ -f "$dir/predictions.jsonl" ] || return 1
    uv run python - "$dir/wave.json" <<'PY'
import json, sys
sys.exit(0 if json.load(open(sys.argv[1])).get('measured') is True else 1)
PY
}

group_done() {  # $@ = wave names; true iff every one is finished
    local name
    for name in "$@"; do
        wave_done "$name" || return 1
    done
    return 0
}

SERVER_PID=""
SERVER_LOG=""
stop_server() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 50); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 0.2
        done
        kill -9 "$SERVER_PID" 2>/dev/null || true
        log "stopped server pid $SERVER_PID"
    fi
    SERVER_PID=""
}
trap 'stop_server' EXIT INT TERM

served_path() {  # print the file path /props reports, or nothing
    uv run python - "$PORT" <<'PY' 2>/dev/null || true
import json, sys, urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/props", timeout=3) as response:
        props = json.loads(response.read().decode())
    print(props.get("model_path") or props.get("path") or "")
except Exception:
    pass
PY
}

wait_for_artifact() {  # $1 = artifact path; poll /props until the served file matches
    local artifact="$1" served=""
    for _ in $(seq 1 300); do
        served="$(served_path)"
        if [ -n "$served" ]; then
            if [ "$(readlink -f "$served")" = "$(readlink -f "$artifact")" ]; then
                return 0
            fi
            log "port $PORT is serving $served, not $artifact; refusing to record against it"
            return 1
        fi
        sleep 1
    done
    log "server did not answer /props within 300s; last log lines:"
    tail -n 20 "$SERVER_LOG" || true
    return 1
}

start_server() {  # $1 = artifact path
    local artifact="$1" token
    local -a argv=()
    SERVER_LOG="$OUT/server-$(basename "$artifact" .gguf).log"
    mapfile -t argv < <(launch_argv "$(serve_config "$artifact")")
    log "serving $(basename "$artifact") at $(serve_config "$artifact")"
    if [ "$DRY" -eq 1 ]; then
        printf '  %s' "$SERVER"
        for token in "${argv[@]}"; do printf ' %s' "$token"; done
        printf '\n'
        return 0
    fi
    "$SERVER" "${argv[@]}" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    wait_for_artifact "$artifact"
}

wave() {  # $1 = name, $2 = artifact, $3 = suite, $4 = kinds
    local name="$1" artifact="$2" suite="$3" kinds="$4" rows
    if wave_done "$name"; then
        log "SKIP $name (finished and measured)"
        return 0
    fi
    if [ "$DRY" -eq 1 ]; then
        printf '  %s\n' "$(wave_argv "$name" "$artifact" "$suite" "$kinds")"
        return 0
    fi
    if [ -d "$OUT/waves/$name" ]; then
        log "RESET $name (incomplete output from an earlier attempt)"
        rm -rf "${OUT:?}/waves/$name"
    fi
    rows="$(uv run python - "$suite" "$kinds" <<'PY'
import json, sys
kinds = set(sys.argv[2].split(','))
print(sum(1 for line in open(sys.argv[1]) if line.strip()
          and json.loads(line).get('kind') in kinds))
PY
)"
    log "WAVE $name rows=$rows kinds=$kinds"
    uv run python scripts/capability_wave.py "$suite" "$OUT/waves/$name" \
        --label "$name" --port "$PORT" --max-tokens "$CAP" --parallel "$PARALLEL" \
        --kinds "$kinds" --serve-config "$(serve_config "$artifact")" \
        --temperature "$TEMPERATURE" --preview 2
    log "DONE  $name $(uv run python - "$OUT/waves/$name/wave.json" <<'PY'
import json, sys
wave = json.load(open(sys.argv[1]))
print(f"rows={wave.get('rows')} cap={wave.get('max_tokens')} "
      f"tokens={wave.get('completion_tokens_total')} measured={wave.get('measured')} "
      f"invalid={wave.get('invalid')}")
PY
)"
}

run_group() {  # $1 = artifact, then wave-name suite kinds triples
    local artifact="$1"; shift
    local -a names=() spec=("$@")
    local index
    for ((index = 0; index < ${#spec[@]}; index += 3)); do names+=("${spec[index]}"); done
    if group_done "${names[@]}"; then
        log "SKIP group $(basename "$artifact") (all waves finished)"
        return 0
    fi
    start_server "$artifact"
    for ((index = 0; index < ${#spec[@]}; index += 3)); do
        wave "${spec[index]}" "$artifact" "${spec[index + 1]}" "${spec[index + 2]}"
    done
    stop_server
}

# ---- operands ------------------------------------------------------------
if [ "$DRY" -eq 0 ]; then
    for operand in "$ARM" "$BASE" "$PUBLISHED" "$AXES" "$SERVER" "$PERPLEXITY"; do
        if [ ! -f "$operand" ]; then
            echo "missing operand: $operand (run the prep card; nothing was served)" >&2
            exit 1
        fi
    done
    for suite in "$REMOVAL_SUITE" "$HELD_OUT_SUITE" "$RETENTION_SUITE"; do
        if [ ! -f "$REPO/$suite" ]; then
            echo "missing suite: $suite" >&2
            exit 1
        fi
    done
    mkdir -p "$OUT/waves"
fi

cd "$REPO"

if [ "$DRY" -eq 1 ]; then
    echo "P7 waves — pinned config -c $CTX --parallel $PARALLEL, temperature $TEMPERATURE, cap $CAP"
    echo "model dir: $MODELS"
    echo
    echo "== base $BASE"
    run_group "$BASE" retention-base "$RETENTION_SUITE" math,code
    echo "== arm $ARM"
    run_group "$ARM" removal-arm "$REMOVAL_SUITE" refusal \
        held_out-arm "$HELD_OUT_SUITE" refusal \
        retention-arm "$RETENTION_SUITE" math,code
    echo "== published $PUBLISHED"
    run_group "$PUBLISHED" removal-published "$REMOVAL_SUITE" refusal \
        held_out-published "$HELD_OUT_SUITE" refusal \
        retention-published "$RETENTION_SUITE" math,code
    echo "== post-wave steps (CPU, no server)"
    printf '  uv run python scripts/refusal_shape_audit.py removal-arm=verification/p7-q3-4b/waves/removal-arm/predictions.jsonl --out verification/p7-q3-4b/shape-audit-removal.json\n'
    printf '  uv run python scripts/refusal_shape_audit.py held_out-arm=verification/p7-q3-4b/waves/held_out-arm/predictions.jsonl --out verification/p7-q3-4b/shape-audit-held_out.json\n'
    printf '  uv run python verification/p7-q3-4b/drift_probe.py --base %s --arm %s --binary %s --out verification/p7-q3-4b/drift.json\n' \
        "$BASE" "$ARM" "$PERPLEXITY"
    printf '  uv run python verification/p7-q3-4b/fidelity_probe.py --gguf %s --axes %s --scale 1.5 --out verification/p7-q3-4b/fidelity.json\n' \
        "$ARM" "$AXES"
    exit 0
fi

log "P7 wave run start (cap $CAP, ctx $CTX, parallel $PARALLEL, temp $TEMPERATURE)"

# ---- waves, one server per artifact --------------------------------------
run_group "$BASE" retention-base "$RETENTION_SUITE" math,code
run_group "$ARM" removal-arm "$REMOVAL_SUITE" refusal \
    held_out-arm "$HELD_OUT_SUITE" refusal \
    retention-arm "$RETENTION_SUITE" math,code
run_group "$PUBLISHED" removal-published "$REMOVAL_SUITE" refusal \
    held_out-published "$HELD_OUT_SUITE" refusal \
    retention-published "$RETENTION_SUITE" math,code

# ---- shape audits: what the removal numbers look like on the page (R6) ----
if [ -f "$OUT/waves/removal-arm/predictions.jsonl" ]; then
    log "SHAPE removal-arm -> shape-audit-removal.json"
    uv run python scripts/refusal_shape_audit.py \
        "removal-arm=verification/p7-q3-4b/waves/removal-arm/predictions.jsonl" \
        --out "$OUT/shape-audit-removal.json"
fi
if [ -f "$OUT/waves/held_out-arm/predictions.jsonl" ]; then
    log "SHAPE held_out-arm -> shape-audit-held_out.json"
    uv run python scripts/refusal_shape_audit.py \
        "held_out-arm=verification/p7-q3-4b/waves/held_out-arm/predictions.jsonl" \
        --out "$OUT/shape-audit-held_out.json"
fi

# ---- drift and realized fidelity (CPU) ------------------------------------
log "DRIFT control ln-PPL base vs arm -> drift.json"
uv run python "$OUT/drift_probe.py" --base "$BASE" --arm "$ARM" \
    --binary "$PERPLEXITY" --out "$OUT/drift.json"

log "FIDELITY realized cos per edited tensor -> fidelity.json"
uv run python "$OUT/fidelity_probe.py" --gguf "$ARM" --axes "$AXES" \
    --scale 1.5 --out "$OUT/fidelity.json"

log "P7 wave run complete; every server this script started is stopped"
