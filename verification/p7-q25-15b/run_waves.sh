#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# P7 qwen2.5-1.5b: every wave the cert spec names, in order, on CPU.
#
# One CPU llama-server at a time, every artifact served at the pinned cert argv
# (-c 4096 --parallel 4, cap per axis as registered in
# verification/releases/matrix.manifest.json: removal 4096, held_out 4096,
# retention 1024). A wave is only comparable to another wave taken at the same
# argv (METHODOLOGY R9), so the three artifacts are served with the identical
# string and each wave records it.
#
# Resumable: a wave whose directory already holds a measured wave.json with the
# right row count is skipped, so a killed run continues where it stopped. The
# local steps (shape audits, drift, fidelity) skip on their own output file.
# Every server this script starts is killed on exit, interrupt or error.
#
#   bash verification/p7-q25-15b/run_waves.sh          # run every step, in order
#   bash verification/p7-q25-15b/run_waves.sh --dry    # print the argv list only
#   bash verification/p7-q25-15b/run_waves.sh --force  # redo finished waves
#
# Overrides: P7_ARTS_DIR, P7_SERVER (llama-server binary), P7_PORT, P7_THREADS,
# P7_CTX, P7_PARALLEL, P7_CAP_REMOVAL, P7_CAP_HELD_OUT, P7_CAP_RETENTION.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE"
WAVES="$OUT/waves"
PROGRESS="$OUT/progress.txt"

# The GGUF artifacts are gitignored and ~5 GB, so they live outside the tree at a
# stable path (the same convention as the q3 cert's /mnt/ssd/hermes/models). They are
# bound to the spec by sha256 in operands.json, never by this path.
ARTS_DIR="${P7_ARTS_DIR:-/mnt/ssd/models/p7-q25-15b}"
ART_BASE="$ARTS_DIR/artifacts/base-q8.gguf"
ART_PUBLISHED="$ARTS_DIR/artifacts/published-q8.gguf"
ART_ARM="$ARTS_DIR/artifacts/arm-refit-q8.gguf"
AXES="$ARTS_DIR/records/refit-axes.pt"

SUITE_HARMFUL="$OUT/suites/harmful-40.jsonl"
SUITE_HOLDOUT24="$OUT/suites/holdout-24f.jsonl"
SUITE_HOLDOUT120="$ROOT/verification/screens/q3-holdout-wider.jsonl"
SUITE_RETENTION="$ROOT/verification/capability/capability-run-en.jsonl"

CTX="${P7_CTX:-4096}"
PARALLEL="${P7_PARALLEL:-4}"
PORT="${P7_PORT:-8141}"
THREADS="${P7_THREADS:-16}"
CAP_REMOVAL="${P7_CAP_REMOVAL:-4096}"
CAP_HELD_OUT="${P7_CAP_HELD_OUT:-4096}"
CAP_RETENTION="${P7_CAP_RETENTION:-1024}"
# CPU-only build (no cuda/vulkan backend loaded): this card holds no GPU.
SERVER_BIN="${P7_SERVER:-/mnt/ssd/moth-cache/engines/llama.cpp/build-cpu/bin/llama-server}"

DRY=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --dry) DRY=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg (use --dry, --force, --help)" >&2; exit 2 ;;
  esac
done

log() {
  local line
  line="$(date -Iseconds) $*"
  if [[ "$DRY" -eq 1 ]]; then
    # --dry prints the argv list and touches nothing on disk.
    printf '%s\n' "$line"
    return 0
  fi
  printf '%s\n' "$line" | tee -a "$PROGRESS"
}

die() { log "FAILED: $*"; exit 1; }

serve_argv() {  # artifact
  printf -- '-m %s -c %s --parallel %s --jinja --no-webui --host 127.0.0.1 --port %s' \
    "$1" "$CTX" "$PARALLEL" "$PORT"
}

wave_rows() {  # suite -> number of non-empty lines
  awk 'NF' "$1" | wc -l | tr -d ' '
}

wave_done() {  # dir rows
  local dir="$1" rows="$2"
  [[ "$FORCE" -eq 1 ]] && return 1
  [[ -f "$dir/wave.json" && -f "$dir/predictions.jsonl" ]] || return 1
  uv run python - "$dir" "$rows" <<'PY'
import json, sys
from pathlib import Path
directory, rows = Path(sys.argv[1]), int(sys.argv[2])
try:
    wave = json.loads((directory / 'wave.json').read_text())
except (OSError, json.JSONDecodeError):
    sys.exit(1)
predictions = sum(1 for line in (directory / 'predictions.jsonl').read_text().splitlines() if line.strip())
ok = bool(wave.get('measured')) and int(wave.get('rows') or 0) == rows and predictions == rows
sys.exit(0 if ok else 1)
PY
}

# ---------------------------------------------------------------- server life
SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "stopping server pid $SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 100); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 0.2
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap 'stop_server' EXIT INT TERM

port_busy() { curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

start_server() {  # artifact
  local artifact="$1"
  [[ -x "$SERVER_BIN" ]] || die "server binary not executable: $SERVER_BIN"
  port_busy && die "port $PORT already answers /health — another server is there; refusing to serve the wrong artifact"
  local argv
  argv="$(serve_argv "$artifact")"
  log "server start: $SERVER_BIN $argv"
  # shellcheck disable=SC2086
  LD_LIBRARY_PATH="$(dirname "$SERVER_BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$SERVER_BIN" $argv -t "$THREADS" --log-file "$OUT/server.log" >/dev/null 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 240); do
    if port_busy; then
      local served
      served="$(curl -sf --max-time 5 "http://127.0.0.1:$PORT/props" \
        | uv run python -c 'import json,sys; print(json.load(sys.stdin).get("model_path") or "")' 2>/dev/null || true)"
      log "server ready (pid $SERVER_PID, serves $served)"
      return 0
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || die "server exited before answering /health (see $OUT/server.log)"
    sleep 1
  done
  die "server did not answer /health within 240s"
}

# ---------------------------------------------------------------- steps
run_wave() {  # name artifact suite cap kinds(optional)
  local name="$1" artifact="$2" suite="$3" cap="$4" kinds="${5:-}"
  local dir="$WAVES/$name" rows argv
  rows="$(wave_rows "$suite")"
  [[ -f "$artifact" ]] || die "artifact missing for wave $name: $artifact"
  [[ -f "$suite" ]] || die "suite missing for wave $name: $suite"
  if wave_done "$dir" "$rows"; then
    log "wave $name: already done ($rows rows) — skip"
    return 0
  fi
  argv="$(serve_argv "$artifact")"
  if [[ "$DRY" -eq 1 ]]; then
    log "DRY wave $name: server [$argv] ; suite $suite rows $rows cap $cap kinds ${kinds:-all}"
    return 0
  fi
  mkdir -p "$dir"
  start_server "$artifact"
  local extra=()
  [[ -n "$kinds" ]] && extra=(--kinds "$kinds")
  log "wave $name: $suite ($rows rows, cap $cap) -> $dir"
  uv run python "$ROOT/scripts/capability_wave.py" "$suite" "$dir" \
    --label "$name" --port "$PORT" --max-tokens "$cap" --parallel "$PARALLEL" \
    --serve-config "$argv" "${extra[@]}" >"$dir/stdout.txt" 2>&1 \
    || die "wave $name failed (see $dir/stdout.txt and $OUT/server.log)"
  stop_server
  log "wave $name: done"
}

run_shape() {  # label predictions out
  local label="$1" predictions="$2" out="$3"
  if [[ -f "$out" && "$FORCE" -eq 0 ]]; then
    log "shape audit $out: already done — skip"
    return 0
  fi
  if [[ "$DRY" -eq 1 ]]; then
    log "DRY shape audit: refusal_shape_audit.py $label=$predictions --out $out"
    return 0
  fi
  [[ -f "$predictions" ]] || die "shape audit needs predictions, missing: $predictions"
  log "shape audit $out over $predictions"
  uv run python "$ROOT/scripts/refusal_shape_audit.py" "$label=$predictions" --out "$out" \
    >>"$PROGRESS" 2>&1 || die "shape audit failed: $out"
}

local_step() {  # out description command...
  local out="$1" description="$2"
  shift 2
  if [[ -f "$out" && "$FORCE" -eq 0 ]]; then
    log "$description: already done ($out) — skip"
    return 0
  fi
  if [[ "$DRY" -eq 1 ]]; then
    log "DRY $description: $*"
    return 0
  fi
  log "$description: $*"
  "$@" >>"$PROGRESS" 2>&1 || die "$description failed (see $PROGRESS)"
}

# ---------------------------------------------------------------- preflight
[[ "$DRY" -eq 1 ]] || mkdir -p "$WAVES"
log "run_waves start: ctx $CTX parallel $PARALLEL port $PORT server $SERVER_BIN"
for artifact in "$ART_BASE" "$ART_PUBLISHED" "$ART_ARM"; do
  if [[ ! -f "$artifact" ]]; then
    die "missing operand $artifact — build the artifacts before running waves"
  fi
  [[ "$DRY" -eq 1 ]] || log "operand $(basename "$artifact") $(stat -c %s "$artifact") bytes"
done
for suite in "$SUITE_HARMFUL" "$SUITE_HOLDOUT24" "$SUITE_HOLDOUT120" "$SUITE_RETENTION"; do
  [[ -f "$suite" ]] || die "missing suite $suite"
done

# ---------------------------------------------------------------- waves, in order
# Removal: harmful-40, both wording inventories are graded from one wave
# (score_response stores refusal_check and refusal_check_wide at write time).
run_wave removal-arm       "$ART_ARM"       "$SUITE_HARMFUL"   "$CAP_REMOVAL"   refusal
run_wave removal-base      "$ART_BASE"      "$SUITE_HARMFUL"   "$CAP_REMOVAL"   refusal
run_wave removal-published "$ART_PUBLISHED" "$SUITE_HARMFUL"   "$CAP_REMOVAL"   refusal
# Held-out: the 120-row registered wider suite is the certify axis; the published
# build is the paired competitor, the base is the reference the edit moved from.
run_wave heldout120-arm       "$ART_ARM"       "$SUITE_HOLDOUT120" "$CAP_HELD_OUT" refusal
run_wave heldout120-base      "$ART_BASE"      "$SUITE_HOLDOUT120" "$CAP_HELD_OUT" refusal
run_wave heldout120-published "$ART_PUBLISHED" "$SUITE_HOLDOUT120" "$CAP_HELD_OUT" refusal
# The 24 held-out families the spec calls out: reported beside the 120, never as
# the axis (certify's floor is 120 rows).
run_wave holdout24-arm  "$ART_ARM"  "$SUITE_HOLDOUT24" "$CAP_HELD_OUT" refusal
run_wave holdout24-base "$ART_BASE" "$SUITE_HOLDOUT24" "$CAP_HELD_OUT" refusal
# Retention: the 241-row authored suite, base and published measured at the same argv.
run_wave retention-arm       "$ART_ARM"       "$SUITE_RETENTION" "$CAP_RETENTION"
run_wave retention-base      "$ART_BASE"      "$SUITE_RETENTION" "$CAP_RETENTION"
run_wave retention-published "$ART_PUBLISHED" "$SUITE_RETENTION" "$CAP_RETENTION"

stop_server

# ---------------------------------------------------------------- local evidence
run_shape removal  "$WAVES/removal-arm/predictions.jsonl"       "$OUT/shape-audit-removal.json"
run_shape heldout  "$WAVES/heldout120-arm/predictions.jsonl"    "$OUT/shape-audit-heldout.json"
local_step "$OUT/drift.json" "drift" \
  uv run python "$ROOT/scripts/p7_q25_drift.py" \
  --base "$ART_BASE" --arm "$ART_ARM" --out "$OUT/drift.json" --threads "$THREADS"
local_step "$OUT/fidelity.json" "fidelity" \
  uv run python "$ROOT/scripts/p7_q25_fidelity.py" \
  --base "$ART_BASE" --arm "$ART_ARM" --axes "$AXES" --out "$OUT/fidelity.json"

stop_server
if [[ "$DRY" -eq 1 ]]; then
  log "DRY: no server started, no file written"
else
  log "all waves done — next: bash $OUT/run_waves.sh (resume check) then the commands in $OUT/NEXT.md"
fi
