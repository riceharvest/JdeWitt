#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# P7 Spark-X2.6-1.7B Q4_K_M: build the edited arm on the served artifact and
# take every certification wave at ONE pinned serve config.
#
# This is the wave card's whole job: it is launched detached under the b70 lock
# and it stops every server it starts before it exits.
#
#   bash verification/p7-spark/run_waves.sh            # run / resume
#   bash verification/p7-spark/run_waves.sh --dry      # print the argv list, touch nothing
#
# Order (each step is skipped when its output is already on disk and matches):
#   1. operands   sha256 of the base, the f16 twin, the records, suites, corpora, binaries
#   2. axes       spanfit axes --shape shared (refit; deterministic given the records)
#   3. codec      probe q6_k / q8_0 / f16 on EVERY edited tensor of the pristine artifact
#   4. merge      edit-quantized at the chosen type, scale stays 1.0
#   5. fidelity   compare_edit over every edited tensor of the merged bytes
#   6. drift      prefill-only control PPL base vs arm (zero generated tokens)
#   7. arm server  removal-40 (cap 4096), held-out 120 (4096), held-out 24 (4096),
#                  retention 241 (1024)  -- one argv per axis, -c 4096 --parallel 4
#   8. base server removal-base (4096), retention-base (1024) -- the baseline pair
#   9. shape      refusal shape on the removal and held-out waves, bound to the arm digest
#  10. spec       verification/p7-spark/spec.json (+ the manifest's release path)
#
# The port, device, caps and paths are variables below; nothing is guessed from
# a previous run's scratch files.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT="$ROOT/verification/p7-spark"
PY="$ROOT/.venv/bin/python"
HELPER="$ROOT/scripts/p7_spark_waves.py"
SERVE=${P7_SERVE:-/mnt/ssd/b70-prep/llama.cpp-spark/build-vulkan/bin/llama-server}
PPL=${P7_PPL:-/mnt/ssd/dev workspace/llama.cpp-dspark/build-perp/bin/llama-perplexity}
BASE=${P7_BASE:-/mnt/ssd/models/Spark-X2.6-1.7B-Q4_K_M.gguf}
ARM=${P7_ARM:-/mnt/ssd/models/Spark-X2.6-1.7B-Q4_K_M.p7-arm.gguf}
RECORDS=${P7_RECORDS:-/mnt/ssd/dev workspace/JdeWitt/runs/p4/records.pt}
AXES="$OUT/axes-shared.pt"
PORT=${P7_PORT:-8141}
FLOOR=${P7_FLOOR:-0.90}
F16=${P7_F16:-/mnt/ssd/models/Spark-X2.6-1.7B-f16.gguf}

ADV=verification/p7-spark/suites/harmful-40.jsonl
HOLDOUT=verification/screens/q3-holdout-wider.jsonl
HOLDOUT24=verification/p7-spark/suites/q3-holdout-24.jsonl
RETENTION=verification/capability/capability-run-en.jsonl

DRY=0
DEV=${P7_VK_DEVICE:-}
SERVER_PID=
SERVER_LABEL=

usage() {
  sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry) DRY=1; shift ;;
    --port) PORT=$2; shift 2 ;;
    --floor) FLOOR=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1 (use --dry, --port N, --floor X, --help)" >&2; exit 64 ;;
  esac
done

mkdir -p "$OUT/waves" "$OUT/steps" "$OUT/servers" "$OUT/shape"
cd "$ROOT" || exit 1
PROGRESS="$OUT/progress.txt"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$PROGRESS"
}

die() {
  log "FAILED: $*"
  stop_server
  exit 1
}

# --- commands -----------------------------------------------------------------
# must() is the one place a step runs. Under --dry it prints the exact argv it
# would run and returns success, so the dry output IS the plan.
must() {
  if [ "$DRY" = 1 ]; then
    local arg
    printf '  +'
    for arg in "$@"; do printf ' %q' "$arg"; done
    printf '\n'
    return 0
  fi
  log "run: $*"
  "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "FAILED (rc=$rc): $*"
    stop_server
    exit $rc
  fi
  return 0
}

json_get() {  # file key
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1]));
v=d
for part in sys.argv[2].split("."):
    v = v.get(part) if isinstance(v, dict) else None
print("" if v is None else v)' "$1" "$2"
}

step_done() { [ -f "$OUT/steps/$1.ok" ]; }

mark_step() {  # name  detail
  [ "$DRY" = 1 ] && return 0
  printf '%s %s\n' "$(date -Is)" "${2:-}" > "$OUT/steps/$1.ok"
  log "step $1 done: ${2:-}"
}

serve_config() {  # artifact cap
  "$PY" "$HELPER" argv --artifact "$1" --cap "$2" --port "$PORT" --device "$DEV"
}

resolve_device() {
  if [ "$DRY" = 1 ]; then
    DEV=${P7_VK_DEVICE:-Vulkan1}
    return 0
  fi
  if [ -n "${P7_VK_DEVICE:-}" ]; then
    DEV=$P7_VK_DEVICE
  else
    local listing line
    listing=$("$SERVE" --list-devices 2>&1) || die "cannot list devices: $listing"
    line=$(printf '%s\n' "$listing" | grep -E 'Vulkan[0-9]+: .*(BMG|B70|Arc)' | head -1)
    [ -n "$line" ] || die "no b70 Vulkan device in: $(printf '%s' "$listing" | tr '\n' ' ')"
    DEV=$(printf '%s' "$line" | sed -E 's/^[[:space:]]*(Vulkan[0-9]+):.*/\1/')
  fi
  local named
  named=$("$SERVE" --list-devices 2>&1 | grep -F "$DEV:")
  case "$named" in
    *BMG*|*B70*|*Arc*) log "serving on $DEV (${named#*: })" ;;
    *) die "$DEV is '${named:-not a device}'; refusing to serve on anything but the b70 (the RTX 3060 is banned)" ;;
  esac
}

# --- server lifecycle ---------------------------------------------------------
kill_stale() {
  local f pid cmd
  for f in "$OUT"/servers/*.pid; do
    [ -f "$f" ] || continue
    pid=$(cat "$f" 2>/dev/null)
    [ -n "$pid" ] || { rm -f "$f"; continue; }
    if kill -0 "$pid" 2>/dev/null; then
      cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
      case "$cmd" in
        *llama-server*"--port $PORT"*) log "killing stale server pid $pid from $f"; kill -TERM "$pid" 2>/dev/null ;;
        *) log "leaving pid $pid ($f): it is no longer a llama-server on port $PORT" ;;
      esac
    fi
    rm -f "$f"
  done
}

stop_server() {
  local pid=$SERVER_PID
  [ -n "$pid" ] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "stopping server pid $pid (${SERVER_LABEL:-?})"
    kill -TERM "$pid" 2>/dev/null
    local i
    for i in $(seq 1 60); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$pid" 2>/dev/null && { log "pid $pid ignored SIGTERM; SIGKILL"; kill -KILL "$pid" 2>/dev/null; }
  fi
  rm -f "$OUT/servers/${SERVER_LABEL:-server}.pid"
  SERVER_PID=
  SERVER_LABEL=
}

wait_ready() {  # artifact -> 0 ready, 2 another model on the port, 1 not ready
  local art=$1 rc=1 deadline=$((SECONDS + 900))
  while [ $SECONDS -lt $deadline ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      return 1
    fi
    "$PY" - "$PORT" "$art" <<'PY'
import json, os, sys, urllib.request
port, artifact = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/props', timeout=5) as response:
        payload = json.loads(response.read())
except Exception:
    sys.exit(1)
served = payload.get('model_path') or payload.get('path') or ''
if not served:
    sys.exit(1)
sys.exit(0 if os.path.realpath(served) == os.path.realpath(artifact) else 2)
PY
    rc=$?
    [ $rc -eq 0 ] && return 0
    [ $rc -eq 2 ] && return 2
    sleep 2
  done
  return 1
}

start_server() {  # artifact label
  local art=$1 label=$2 logf
  SERVER_LABEL=$label
  logf="$OUT/servers/$label.log"
  kill_stale
  if [ "$DRY" = 1 ]; then
    printf '  + LD_LIBRARY_PATH=%s %s -m %s --host 127.0.0.1 --port %s -c 4096 --parallel 4 -dev %s -ngl 999 --split-mode none --jinja --no-webui --temp 0  > %s 2>&1 &\n' \
      "$(dirname "$SERVE")" "$SERVE" "$art" "$PORT" "$DEV" "$logf"
    return 0
  fi
  LD_LIBRARY_PATH="$(dirname "$SERVE"):${LD_LIBRARY_PATH:-}" \
    "$SERVE" -m "$art" --host 127.0.0.1 --port "$PORT" -c 4096 --parallel 4 \
    -dev "$DEV" -ngl 999 --split-mode none --jinja --no-webui --temp 0 \
    >"$logf" 2>&1 &
  SERVER_PID=$!
  printf '%s\n' "$SERVER_PID" > "$OUT/servers/$label.pid"
  log "server $label pid $SERVER_PID on $DEV: $SERVE -m $art -c 4096 --parallel 4 -dev $DEV --port $PORT"
  wait_ready "$art"
  local rc=$?
  if [ $rc -eq 2 ]; then
    die "port $PORT is serving a different artifact than $art (see $logf); stop that server first"
  fi
  [ $rc -eq 0 ] || die "server $label did not answer /props with $art within 900s (see $logf)"
  log "server $label ready"
}

# --- waves -------------------------------------------------------------------
wave_done() {  # label
  local f="$OUT/waves/$1/wave.json"
  [ -f "$f" ] || return 1
  "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("measured") is True else 1)' "$f"
}

run_wave() {  # label artifact suite cap kinds
  local label=$1 art=$2 suite=$3 cap=$4 kinds=$5 dir="$OUT/waves/$1" cfg
  if wave_done "$label"; then
    log "skip wave $label (measured wave.json already on disk)"
    return 0
  fi
  if [ -e "$dir/predictions.jsonl" ]; then
    mv "$dir/predictions.jsonl" "$dir/predictions.jsonl.partial-$(date +%s)"
    log "moved an unfinished predictions.jsonl aside in $dir"
  fi
  mkdir -p "$dir"
  cfg=$(serve_config "$art" "$cap")
  must "$PY" "$ROOT/scripts/capability_wave.py" "$suite" "$dir" \
    --label "$label" --port "$PORT" --max-tokens "$cap" --parallel 4 \
    --kinds "$kinds" --serve-config "$cfg" --preview 5 \
    || die "wave $label failed"
  [ "$DRY" = 1 ] && return 0
  wave_done "$label" || die "wave $label wrote no measured wave.json"
  log "wave $label done: $(json_get "$dir/wave.json" rows) rows, $(json_get "$dir/wave.json" completion_tokens_total) tokens"
}

# --- plan --------------------------------------------------------------------
plan() {
  echo "P7 Spark-X2.6-1.7B Q4_K_M — wave plan (dry run; nothing is executed)"
  echo "  repo       $ROOT"
  echo "  serve      $SERVE"
  echo "  ppl        $PPL"
  echo "  base       $BASE"
  echo "  arm        $ARM"
  echo "  records    $RECORDS"
  echo "  port       $PORT      device $DEV   pinned -c 4096 --parallel 4"
  echo "  floor      $FLOOR"
  echo
  echo "1. operands"
  must "$PY" "$HELPER" operands --manifest "$OUT/operands.json"
  echo "2. axes (refit)"
  must "$PY" -m jdewitt.spanfit axes --records "$RECORDS" --out "$AXES" --shape shared
  echo "3. codec table (preview_edit q6_k, q8_0, f16 over every edited tensor)"
  must "$PY" "$HELPER" probe --base "$BASE" --axes "$AXES" --out "$OUT/codec-table.json" \
    --floor "$FLOOR" --types q6_k,q8_0,f16
  echo "4. merge (edit-quantized at the chosen type, scale 1.0)"
  must "$PY" "$HELPER" merge --base "$BASE" --out "$ARM" --records "$RECORDS" \
    --type '<codec-table.chosen_type>' --state "$OUT/steps/merge.json" --scale 1.0 --floor "$FLOOR"
  echo "5. fidelity (compare_edit over every edited tensor)"
  must "$PY" "$HELPER" fidelity --base "$BASE" --arm "$ARM" --axes "$AXES" \
    --out "$OUT/fidelity-full.json" --certify-out "$OUT/fidelity.json" --floor "$FLOOR"
  echo "6. drift (prefill-only control PPL, zero generated tokens)"
  must "$PY" "$HELPER" drift --base "$BASE" --arm "$ARM" --corpora verification/ppl-probe \
    --perplexity "$PPL" --out "$OUT/drift.json" --budget 0.02
  echo "7. arm waves (server: $ARM)"
  start_server "$ARM" arm
  must "$PY" "$ROOT/scripts/capability_wave.py" "$ADV" "$OUT/waves/removal-arm" \
    --label removal-arm --port "$PORT" --max-tokens 4096 --parallel 4 --kinds refusal \
    --serve-config "$(serve_config "$ARM" 4096)" --preview 5
  must "$PY" "$ROOT/scripts/capability_wave.py" "$HOLDOUT" "$OUT/waves/heldout-120-arm" \
    --label heldout-120-arm --port "$PORT" --max-tokens 4096 --parallel 4 --kinds refusal \
    --serve-config "$(serve_config "$ARM" 4096)" --preview 5
  must "$PY" "$ROOT/scripts/capability_wave.py" "$HOLDOUT24" "$OUT/waves/heldout-24-arm" \
    --label heldout-24-arm --port "$PORT" --max-tokens 4096 --parallel 4 --kinds refusal \
    --serve-config "$(serve_config "$ARM" 4096)" --preview 5
  must "$PY" "$ROOT/scripts/capability_wave.py" "$RETENTION" "$OUT/waves/retention-arm" \
    --label retention-arm --port "$PORT" --max-tokens 1024 --parallel 4 --kinds math,code \
    --serve-config "$(serve_config "$ARM" 1024)" --preview 5
  echo "   (stop arm server)"
  echo "8. base waves (server: $BASE)"
  start_server "$BASE" base
  must "$PY" "$ROOT/scripts/capability_wave.py" "$ADV" "$OUT/waves/removal-base" \
    --label removal-base --port "$PORT" --max-tokens 4096 --parallel 4 --kinds refusal \
    --serve-config "$(serve_config "$BASE" 4096)" --preview 5
  must "$PY" "$ROOT/scripts/capability_wave.py" "$RETENTION" "$OUT/waves/retention-base" \
    --label retention-base --port "$PORT" --max-tokens 1024 --parallel 4 --kinds math,code \
    --serve-config "$(serve_config "$BASE" 1024)" --preview 5
  echo "   (stop base server)"
  echo "9. shape audits + spec"
  must "$PY" "$HELPER" shape --pair "removal-arm=$OUT/waves/removal-arm/predictions.jsonl" \
    --artifact "$ARM" --base "$BASE" --recipe '<steps/merge.json.command>' \
    --edit '<steps/merge.json.edit>' --raw "$OUT/shape/removal.raw.json" --out "$OUT/shape/removal.json"
  must "$PY" "$HELPER" shape --pair "heldout-120-arm=$OUT/waves/heldout-120-arm/predictions.jsonl" \
    --artifact "$ARM" --base "$BASE" --recipe '<steps/merge.json.command>' \
    --edit '<steps/merge.json.edit>' --raw "$OUT/shape/heldout.raw.json" --out "$OUT/shape/heldout.json"
  must "$PY" "$HELPER" spec --arm "$ARM" --base "$BASE" --recipe '<steps/merge.json.command>' \
    --edit '<steps/merge.json.edit>' --label spark-1.7b-q4km-s1.0 \
    --removal-wave "$OUT/waves/removal-arm" --held-out-wave "$OUT/waves/heldout-120-arm" \
    --retention-wave "$OUT/waves/retention-arm" --retention-base-wave "$OUT/waves/retention-base" \
    --removal-suite "$ADV" --held-out-suite "$HOLDOUT" --retention-suite "$RETENTION" \
    --shape-removal "$OUT/shape/removal.json" --shape-held-out "$OUT/shape/heldout.json" \
    --drift "$OUT/drift.json" --fidelity "$OUT/fidelity.json" \
    --codec-table "$OUT/codec-table.json" --port "$PORT" --device "$DEV" \
    --out "$OUT/spec.json" --copy-out verification/releases/spark-1.7b/spec.json
  must "$PY" "$HELPER" summary --out "$OUT/waves.json"
  echo
  echo "next (analysis card): see verification/p7-spark/NEXT.md"
}

trap 'stop_server' EXIT
trap 'log "interrupted by signal"; exit 130' INT TERM HUP

# --- run ---------------------------------------------------------------------
log "P7 Spark-X2.6-1.7B start (dry=$DRY port=$PORT floor=$FLOOR)"
resolve_device
if [ "$DRY" = 1 ]; then
  plan
  log "dry run: plan printed, nothing executed"
  exit 0
fi

# 1. operands
log "step 1: operands"
must "$PY" "$HELPER" operands --manifest "$OUT/operands.json" \
  || die "operands do not match the manifest"
mark_step operands "sha256 verified"

# 2. axes (refit; deterministic given the records)
if [ -f "$AXES" ]; then
  log "step 2: axes present at $AXES (refit; tensor is identical to a fresh fit from the same records)"
else
  log "step 2: refitting the shared axis from $RECORDS"
  must "$PY" -m jdewitt.spanfit axes --records "$RECORDS" --out "$AXES" --shape shared \
    || die "spanfit axes failed"
fi
mark_step axes "$(sha256sum "$AXES" | cut -d' ' -f1)"

# 3. codec table
if [ -f "$OUT/codec-table.json" ]; then
  if [ "$(json_get "$OUT/codec-table.json" base.sha256)" != "$(sha256sum "$BASE" | cut -d' ' -f1)" ] || \
     [ "$(json_get "$OUT/codec-table.json" axes.tensor_sha256)" != "$("$PY" "$HELPER" axes-sha --axes "$AXES")" ]; then
    log "codec table was measured on other bytes; re-probing"
    rm -f "$OUT/codec-table.json"
  fi
fi
if [ ! -f "$OUT/codec-table.json" ]; then
  log "step 3: codec table"
  must "$PY" "$HELPER" probe --base "$BASE" --axes "$AXES" \
    --out "$OUT/codec-table.json" --floor "$FLOOR" --types q6_k,q8_0,f16 \
    || die "codec probe failed"
fi
CHOSEN=$(json_get "$OUT/codec-table.json" chosen_type)
if [ -z "$CHOSEN" ]; then
  CHOSEN=$(json_get "$OUT/codec-table.json" best_available_type)
  log "CODEC EVENT: no candidate type clears the $FLOOR floor; building the best available type $CHOSEN so the row still has measured numbers. The row is failed(codec)."
fi
mark_step codec "$CHOSEN"
log "step 3: chosen type $CHOSEN"

# 4. merge
if [ ! -f "$OUT/steps/merge.json" ] || [ "$(json_get "$OUT/steps/merge.json" sha256)" != "$(sha256sum "$ARM" 2>/dev/null | cut -d' ' -f1)" ]; then
  log "step 4: merge at $CHOSEN"
  must "$PY" "$HELPER" merge --base "$BASE" --out "$ARM" --records "$RECORDS" \
    --type "$CHOSEN" --state "$OUT/steps/merge.json" --scale 1.0 --floor "$FLOOR" \
    || die "merge failed"
else
  log "step 4: merge already recorded for $ARM (sha matches)"
fi
MERGE_SHA=$(json_get "$OUT/steps/merge.json" sha256)
[ -n "$MERGE_SHA" ] || die "steps/merge.json has no sha256"
RECIPE=$(json_get "$OUT/steps/merge.json" command)
EDIT=$(json_get "$OUT/steps/merge.json" edit)
mark_step merge "$MERGE_SHA"

# 5. fidelity
if [ ! -f "$OUT/fidelity.json" ] || [ "$(json_get "$OUT/fidelity.json" artifact_sha256)" != "$MERGE_SHA" ]; then
  log "step 5: fidelity (compare_edit over every edited tensor)"
  must "$PY" "$HELPER" fidelity --base "$BASE" --arm "$ARM" --axes "$AXES" \
    --out "$OUT/fidelity-full.json" --certify-out "$OUT/fidelity.json" --floor "$FLOOR" \
    || die "fidelity failed"
fi
mark_step fidelity "min_cos $(json_get "$OUT/fidelity.json" min_cos) over $(json_get "$OUT/fidelity.json" sampled) tensors"

# 6. drift
if [ ! -f "$OUT/drift.json" ] || [ "$(json_get "$OUT/drift.json" source.arm.sha256)" != "$MERGE_SHA" ]; then
  log "step 6: drift (prefill-only control PPL)"
  must "$PY" "$HELPER" drift --base "$BASE" --arm "$ARM" --corpora verification/ppl-probe \
    --perplexity "$PPL" --out "$OUT/drift.json" --budget 0.02 \
    || die "drift failed"
fi
mark_step drift "control delta $(json_get "$OUT/drift.json" control_delta_ln_ppl) (budget 0.02)"

# 7. arm waves
log "step 7: arm waves"
start_server "$ARM" arm
run_wave removal-arm "$ARM" "$ADV" 4096 refusal
run_wave heldout-120-arm "$ARM" "$HOLDOUT" 4096 refusal
run_wave heldout-24-arm "$ARM" "$HOLDOUT24" 4096 refusal
run_wave retention-arm "$ARM" "$RETENTION" 1024 math,code
stop_server
mark_step arm-waves "removal, heldout-120, heldout-24, retention"

# 8. base waves
log "step 8: base waves (the paired baseline)"
start_server "$BASE" base
run_wave removal-base "$BASE" "$ADV" 4096 refusal
run_wave retention-base "$BASE" "$RETENTION" 1024 math,code
stop_server
mark_step base-waves "removal-base, retention-base"

# 9. shape audits
log "step 9: shape audits"
must "$PY" "$HELPER" shape --pair "removal-arm=$OUT/waves/removal-arm/predictions.jsonl" \
  --artifact "$ARM" --base "$BASE" --recipe "$RECIPE" --edit "$EDIT" \
  --raw "$OUT/shape/removal.raw.json" --out "$OUT/shape/removal.json" \
  || die "removal shape audit failed"
must "$PY" "$HELPER" shape --pair "heldout-120-arm=$OUT/waves/heldout-120-arm/predictions.jsonl" \
  --artifact "$ARM" --base "$BASE" --recipe "$RECIPE" --edit "$EDIT" \
  --raw "$OUT/shape/heldout.raw.json" --out "$OUT/shape/heldout.json" \
  || die "held-out shape audit failed"
mark_step shape "removal + held-out audits bound to the arm digest"

# 10. spec
log "step 10: spec"
must "$PY" "$HELPER" spec --arm "$ARM" --base "$BASE" --recipe "$RECIPE" --edit "$EDIT" \
  --label spark-1.7b-q4km-s1.0 \
  --removal-wave "$OUT/waves/removal-arm" --held-out-wave "$OUT/waves/heldout-120-arm" \
  --retention-wave "$OUT/waves/retention-arm" --retention-base-wave "$OUT/waves/retention-base" \
  --removal-suite "$ADV" --held-out-suite "$HOLDOUT" --retention-suite "$RETENTION" \
  --shape-removal "$OUT/shape/removal.json" --shape-held-out "$OUT/shape/heldout.json" \
  --drift "$OUT/drift.json" --fidelity "$OUT/fidelity.json" \
  --codec-table "$OUT/codec-table.json" --port "$PORT" --device "$DEV" \
  --out "$OUT/spec.json" --copy-out verification/releases/spark-1.7b/spec.json \
  || die "spec could not be written from the saved waves"
mark_step spec "$OUT/spec.json"

# 11. rollup
must "$PY" "$HELPER" summary --out "$OUT/waves.json" || true

log "FINISHED all waves; codec verdict $(json_get "$OUT/codec-table.json" verdict), min realized cos $(json_get "$OUT/fidelity.json" min_cos)"
log "next: uv run jdewitt release-certify --spec verification/releases/spark-1.7b/spec.json --out verification/releases/spark-1.7b/packet.json"
