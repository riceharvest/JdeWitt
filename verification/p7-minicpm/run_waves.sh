#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# P7 MiniCPM-V 4.7 1B — one driver for the refit, the merge and every cert wave.
#
# This is the wave card's entry point. It does, in order:
#
#   1  operands   base gguf sha256 vs the pinned digest, suites recut + sha, HF checkpoint
#   2  refit      `jdewitt capture` off the HF checkpoint (B70), then `spanfit axes`
#                 -> the axis is a REFIT: no capture exists on this box, so it is
#                 fitted again from the same suite and the recipe is labelled as one
#   3  merge      `spanfit build --axes ... --gguf-targets auto` into the Q4_K_M bytes;
#                 the merged gguf IS the served and shipped artifact (R8: no adapter
#                 is served anywhere in this loop)
#   4  fidelity   `p4f_gate.py fidelity` over the merged bytes -> cert fidelity input
#   5  drift      `refusal_ppl_screen.py` base vs merged, prefill only -> cert drift input
#   6  waves      base @4096/4: harmful-40 + retention (the paired base side)
#                 merged @4096/4: harmful-40, held-out 120, retention 241   <- the pinned cert run
#                 merged @2048/2: harmful-40 (the R9 comparison run)
#   7  audits     `refusal_shape_audit.py` on the merged waves, stamped with the
#                 artifact identity digest (R8)
#   8  pair       `scripts/cert_config_pair.py` -> comparison.json (boundary rows, R9.4)
#   9  spec       `cert_inputs.py spec` -> spec.json, ready for `jdewitt release-certify`
#
# Re-runnable: a step whose output exists is skipped (`--force` redoes everything).
# Every server this script starts is killed when its wave group ends and again by
# the EXIT trap, so a failed wave cannot leave a server holding the port or the GPU.
#
#   bash verification/p7-minicpm/run_waves.sh --dry     # print the argv list only
#   bash verification/p7-minicpm/run_waves.sh           # run (resumable)
#   bash verification/p7-minicpm/run_waves.sh --force   # redo every wave and step
#
# Knobs (env): ARTIFACT_DIR, PORT, SCALE, SERVER, DEVICE_ARGS, HF_MODEL,
# HF_REVISION, BASE_OPERAND, CAP_REMOVAL, CAP_HELDOUT, CAP_RETENTION.
#
# The pinned config is ctx 4096 / parallel 4. `-c` is the TOTAL context and
# `--parallel` splits it, so one request can generate at most 1024 tokens there:
# the caps are 1024 on every axis at 4096/4, and the comparison run is 2048/2
# (also 1024 per slot) so the R9 pair differs only in ctx/parallel.

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$REPO_ROOT"

OUT=verification/p7-minicpm
PROGRESS_LOG="$OUT/progress.txt"
LOG_DIR="$OUT/logs"
WAVES="$OUT/waves"

ARTIFACT_DIR=${ARTIFACT_DIR:-/mnt/ssd/p7-minicpm}
PORT=${PORT:-8134}
SCALE=${SCALE:-1.0}
SERVER=${SERVER:-/home/dario/llama.cpp-minicpm47/build-cpu/bin/llama-server}
PY=${PY:-python3}

# Every pinned path and digest comes from operands.json at run time: the served
# operands are 600 MB-2 GB files, and operands.json is the single place that says
# which bytes are allowed. run_waves.sh never carries one of those paths itself.
OPERANDS="$OUT/operands.json"
[ -f "$OPERANDS" ] || { echo "ERROR missing $OPERANDS" >&2; exit 1; }

operand() {  # operand json.dotted.path -> value
  "$PY" -c 'import json,sys
doc = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    doc = doc[key]
print(doc)' "$OPERANDS" "$1"
}

BASE_SHA_PIN=$(operand artifact_under_cert.sha256)
BASE_BYTES_PIN=$(operand artifact_under_cert.bytes)
HF_CONFIG_SHA_PIN=$(operand model.config_sha256)
if [ -z "${HF_MODEL+x}" ]; then HF_MODEL=$(operand model.local_path); fi
if [ -z "${HF_REVISION+x}" ]; then HF_REVISION=$(operand model.revision); fi

# One request cannot generate past its slot: ctx/parallel tokens. These are the
# per-axis generation caps at the pinned config, and the q3 packet's caps are the
# same shape (retention 1024). Raising one without raising -c changes the pinned
# config, which R9 forbids inside one cert.
CAP_REMOVAL=${CAP_REMOVAL:-1024}
CAP_HELDOUT=${CAP_HELDOUT:-1024}
CAP_RETENTION=${CAP_RETENTION:-1024}

ARM_LABEL=${ARM_LABEL:-refit-perlayer-s$SCALE}

REMOVAL_SUITE=verification/screens/advbench-40.jsonl
HELDOUT_SUITE="$OUT/heldout-24f.jsonl"
RETENTION_SUITE=verification/capability/capability-run-en.jsonl
FIT_SUITE="$OUT/fit-suite.jsonl"

DRY=0
FORCE=0
SERVER_PIDS=()

usage() {
  sed -n '3,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry) DRY=1 ;;
    --force) FORCE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

die() { printf 'ERROR %s\n' "$*" >&2; exit 1; }

say() {
  local line
  line="$(date -Is) $*"
  printf '%s\n' "$line" >&2
  if [ "$DRY" -eq 0 ]; then printf '%s\n' "$line" >> "$PROGRESS_LOG"; fi
}

quote_args() {
  local out=() arg
  for arg in "$@"; do out+=("$(printf '%q' "$arg")"); done
  printf '%s' "${out[*]}"
}

plan() {
  if [ "$DRY" -eq 1 ]; then
    printf 'DRY  %s\n' "$(quote_args "$@")"
  else
    say "RUN  $(quote_args "$@")"
    "$@"
  fi
}

plan_shell() {  # for commands with redirection: printed under --dry, run otherwise
  if [ "$DRY" -eq 1 ]; then printf 'DRY  %s\n' "$1"; fi
}

run_python() {  # $@ = argv for the repo python (uv run python ...)
  if [ "$DRY" -eq 1 ]; then
    printf 'DRY  %s\n' "$(quote_args uv run python "$@")"
    return 0
  fi
  say "RUN  $(quote_args uv run python "$@")"
  uv run python "$@"
}

# ---------------------------------------------------------------- preconditions
check_no_whitespace() {
  local name=$1 value=$2
  case "$value" in
    *[[:space:]]*)
      die "$name ($value) contains whitespace. The serve config is a shell-split
     string recorded verbatim in the packet (R9), so the artifacts must live on a
     path without spaces. Set ARTIFACT_DIR to such a directory." ;;
  esac
}

resolve_base() {
  if [ -n "${BASE_OPERAND:-}" ]; then
    [ -f "$BASE_OPERAND" ] || die "BASE_OPERAND=$BASE_OPERAND does not exist"
    printf '%s' "$BASE_OPERAND"
    return 0
  fi
  "$PY" - "$REPO_ROOT" "$OPERANDS" <<'PY'
import json, os, sys
repo, operands = sys.argv[1], sys.argv[2]
doc = json.load(open(os.path.join(repo, operands)))
entry = doc['artifact_under_cert']
paths = [p if os.path.isabs(p) else os.path.join(repo, p) for p in entry['locations']]
for path in paths:
    if os.path.isfile(path):
        print(path)
        break
else:
    print('the pinned base gguf is not on disk; looked in:', file=sys.stderr)
    for path in paths:
        print('    ' + path, file=sys.stderr)
    print(f'    it must be {entry["bytes"]} bytes with sha256 {entry["sha256"]}. '
          'Restore the export that verification/minicpm-v47/summary.json describes '
          '(converter: convert_hf_to_gguf.py --outtype q4_k_m) or set BASE_OPERAND.',
          file=sys.stderr)
    raise SystemExit(1)
PY
}

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

BASE_PATH=$(resolve_base)
check_no_whitespace ARTIFACT_DIR "$ARTIFACT_DIR"
check_no_whitespace SERVER "$SERVER"

BASE_LINK="$ARTIFACT_DIR/base.gguf"
MERGED="$ARTIFACT_DIR/minicpm-v47-$ARM_LABEL-Q4_K_M.gguf"
RECORDS="$ARTIFACT_DIR/refit-records.pt"
AXES="$ARTIFACT_DIR/refit-axes.pt"
CAPTURE_JSON="$RECORDS.capture.json"
MERGE_JSON="$OUT/merge.json"
FIDELITY_RAW="$OUT/fidelity-raw.json"
FIDELITY_JSON="$OUT/fidelity.json"
DRIFT_SCREEN="$OUT/ppl-screen.json"
DRIFT_JSON="$OUT/drift.json"
COMPARISON="$OUT/comparison.json"
SPEC="$OUT/spec.json"

W_BASE_REMOVAL="$WAVES/base-removal-4096p4"
W_BASE_RETENTION="$WAVES/base-retention-4096p4"
W_REMOVAL="$WAVES/removal-4096p4"
W_HELDOUT="$WAVES/heldout-4096p4"
W_RETENTION="$WAVES/retention-4096p4"
W_REMOVAL_LOW="$WAVES/removal-2048p2"
AUDIT_REMOVAL="$OUT/shape-audit-removal.json"
AUDIT_HELDOUT="$OUT/shape-audit-heldout.json"

# The served argv, split into an array so the server is launched and recorded
# from the same tokens. DEVICE_ARGS defaults to the CPU device on this build; the
# wave card may pass e.g. DEVICE_ARGS="--device Vulkan0" with a Vulkan server.
if [ -z "${DEVICE_ARGS+x}" ]; then DEVICE_ARGS="--device none"; fi
read -r -a DEVICE_ARGS_ARR <<< "$DEVICE_ARGS"

argv_4096_merged=(-m "$MERGED" -c 4096 --parallel 4 "${DEVICE_ARGS_ARR[@]}" --jinja --no-webui --host 127.0.0.1 --port "$PORT")
argv_2048_merged=(-m "$MERGED" -c 2048 --parallel 2 "${DEVICE_ARGS_ARR[@]}" --jinja --no-webui --host 127.0.0.1 --port "$PORT")
argv_4096_base=(-m "$BASE_LINK" -c 4096 --parallel 4 "${DEVICE_ARGS_ARR[@]}" --jinja --no-webui --host 127.0.0.1 --port "$PORT")

CFG_4096="${argv_4096_merged[*]}"
CFG_2048="${argv_2048_merged[*]}"
CFG_BASE="${argv_4096_base[*]}"

# ---------------------------------------------------------------- servers
stop_servers() {
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    if [ -n "$pid" ]; then wait "$pid" 2>/dev/null || true; fi
  done
  SERVER_PIDS=()
}
trap stop_servers EXIT INT TERM

start_server() {
  local name=$1; shift
  if [ "$DRY" -eq 1 ]; then
    printf 'DRY  %s\n' "$(quote_args "$SERVER" "$@")"
    return 0
  fi
  local log="$LOG_DIR/server-$name.log"
  printf 'START server %s: %s\n' "$name" "$(quote_args "$SERVER" "$@")" >&2
  "$SERVER" "$@" >"$log" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("$pid")
  local i
  for i in $(seq 1 300); do
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -n 30 "$log" >&2 || true
      die "server $name exited before answering /props; log: $log"
    fi
    if "$PY" - "$PORT" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
try:
    with urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/props', timeout=2) as r:
        json.loads(r.read())
except Exception:
    raise SystemExit(1)
PY
    then
      say "server $name is up (pid $pid, port $PORT)"
      return 0
    fi
    sleep 2
  done
  tail -n 30 "$log" >&2 || true
  die "server $name did not answer /props on port $PORT within 10 minutes; log: $log"
}

# ---------------------------------------------------------------- resumability
step_done() { [ -f "$1" ] && [ "$FORCE" -eq 0 ]; }
wave_done() { [ -f "$1/predictions.jsonl" ] && [ -f "$1/wave.json" ] && [ "$FORCE" -eq 0 ]; }
wave_pending() { [ "$DRY" -eq 1 ] && return 0; [ "$FORCE" -eq 1 ] && return 0; ! wave_done "$1"; }
any_wave_pending() {
  local dir
  for dir in "$@"; do
    if wave_pending "$dir"; then return 0; fi
  done
  return 1
}

# ---------------------------------------------------------------- step 1: operands
step_operands() {
  say "STEP 1 operands: base digest, suites, checkpoint"
  local actual
  if [ "$DRY" -eq 1 ]; then
    printf 'DRY  sha256sum %s  (must equal %s)\n' "$(quote_args "$BASE_PATH")" "$BASE_SHA_PIN"
    printf 'DRY  uv run python %s  (suite digests)\n' "$OUT/recut_suites.py"
    printf 'DRY  sha256sum %s  (must equal %s)\n' \
      "$(quote_args "$HF_MODEL/config.json")" "$HF_CONFIG_SHA_PIN"
    return 0
  fi
  [ -f "$BASE_PATH" ] || die "base gguf $BASE_PATH is not a file"
  local bytes
  bytes=$(stat -c %s "$BASE_PATH")
  [ "$bytes" = $BASE_BYTES_PIN ] || die "base gguf is $bytes bytes, the pinned operand is 672,004,000"
  actual=$(sha256_of "$BASE_PATH")
  [ "$actual" = "$BASE_SHA_PIN" ] || \
    die "base gguf sha256 is $actual, the pinned operand is $BASE_SHA_PIN"
  say "operand base ok: $BASE_PATH sha256 $actual ($bytes bytes)"
  uv run python "$OUT/recut_suites.py" | tee -a "$PROGRESS_LOG"
  [ -f "$HF_MODEL/config.json" ] || die "HF checkpoint $HF_MODEL has no config.json"
  actual=$(sha256_of "$HF_MODEL/config.json")
  [ "$actual" = "$HF_CONFIG_SHA_PIN" ] || \
    die "$HF_MODEL/config.json sha256 is $actual, the pinned checkpoint config is $HF_CONFIG_SHA_PIN"
  say "operand checkpoint ok: $HF_MODEL config $actual revision $HF_REVISION"
  mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" "$WAVES"
  if [ -e "$BASE_LINK" ]; then
    actual=$(sha256_of "$BASE_LINK")
    [ "$actual" = "$BASE_SHA_PIN" ] || die "$BASE_LINK exists but hashes to $actual; remove it"
  else
    ln "$BASE_PATH" "$BASE_LINK" 2>/dev/null || ln -s "$BASE_PATH" "$BASE_LINK"
    say "linked base -> $BASE_LINK (hardlink; the serve config must not carry a space)"
  fi
}

# ---------------------------------------------------------------- step 2: refit
step_refit() {
  if step_done "$AXES"; then say "STEP 2 refit: skip ($AXES exists)"; return 0; fi
  say "STEP 2 refit: capture (B70) then spanfit axes --shape perlayer  [REFIT]"
  if [ "$DRY" -eq 0 ]; then mkdir -p "$ARTIFACT_DIR" "$LOG_DIR"; fi
  if ! step_done "$RECORDS"; then
    run_python -m jdewitt.cli capture \
      --model "$HF_MODEL" \
      --revision "$HF_REVISION" \
      --suite "$FIT_SUITE" \
      --gguf "$BASE_LINK" \
      --out "$RECORDS" \
      --ctx-size 4096 \
      --kinds refusal,harmless \
      --locus answer
  fi
  if [ "$DRY" -eq 0 ]; then
    [ -f "$CAPTURE_JSON" ] || die "capture wrote no report beside $RECORDS"
    uv run python - "$CAPTURE_JSON" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
if report.get('fit_usable') is not True:
    raise SystemExit('capture is not fit_usable: ' + ' | '.join(report.get('notes') or []))
print(f"capture rows {report['records_written']} labels {report['labels']} "
      f"fit_usable {report['fit_usable']}")
PY
  fi
  run_python -m jdewitt.spanfit axes --records "$RECORDS" --out "$AXES" --shape perlayer
}

# ---------------------------------------------------------------- step 3: merge
step_merge() {
  if step_done "$MERGED"; then say "STEP 3 merge: skip ($MERGED exists)"; return 0; fi
  say "STEP 3 merge: spanfit build --axes --gguf-targets auto -> $MERGED"
  if [ "$DRY" -eq 0 ]; then mkdir -p "$ARTIFACT_DIR" "$LOG_DIR"; fi
  MERGE_ARGV=(uv run python -m jdewitt.spanfit build
    --gguf-in "$BASE_LINK"
    --out "$MERGED"
    --axes "$AXES"
    --gguf-targets auto
    --shape perlayer
    --axis-space output
    --scale "$SCALE"
    --dequant-cache "$ARTIFACT_DIR/dequant-cache")
  if [ "$DRY" -eq 1 ]; then
    printf 'DRY  %s\n' "$(quote_args "${MERGE_ARGV[@]}")"
    return 0
  fi
  say "RUN  $(quote_args "${MERGE_ARGV[@]}")"
  "${MERGE_ARGV[@]}" | tee "$MERGE_JSON"
  uv run python - "$MERGE_JSON" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
changed = int(report.get('tensors_changed') or 0)
if changed == 0:
    raise SystemExit('merge changed 0 tensors; refusing to serve an unedited file')
print(f"merge: {changed} tensors changed, "
      f"{report.get('changed_blocks_total')} blocks, {report.get('elapsed_s')}s")
PY
}

# ---------------------------------------------------------------- step 4: fidelity
step_fidelity() {
  if step_done "$FIDELITY_JSON"; then say "STEP 4 fidelity: skip ($FIDELITY_JSON exists)"; return 0; fi
  say "STEP 4 fidelity: compare_edit over the merged bytes"
  if [ "$DRY" -eq 1 ]; then
    printf 'DRY  uv run python scripts/p4f_gate.py fidelity --base %s --merged %s --axes %s --tag %s --out %s --scale %s\n' \
      "$(quote_args "$BASE_LINK")" "$(quote_args "$MERGED")" "$(quote_args "$AXES")" \
      "$ARM_LABEL" "$FIDELITY_RAW" "$SCALE"
    printf 'DRY  uv run python %s fidelity --raw %s --out %s --artifact-sha <merged sha> --artifact-bytes <merged bytes> --label %s --scale %s\n' \
      "$OUT/cert_inputs.py" "$FIDELITY_RAW" "$FIDELITY_JSON" "$ARM_LABEL" "$SCALE"
    return 0
  fi
  uv run python scripts/p4f_gate.py fidelity \
    --base "$BASE_LINK" --merged "$MERGED" --axes "$AXES" \
    --tag "$ARM_LABEL" --out "$FIDELITY_RAW" --scale "$SCALE"
  local sha bytes
  sha=$(sha256_of "$MERGED")
  bytes=$(stat -c %s "$MERGED")
  uv run python "$OUT/cert_inputs.py" fidelity --raw "$FIDELITY_RAW" --out "$FIDELITY_JSON" \
    --artifact-sha "$sha" --artifact-bytes "$bytes" --label "$ARM_LABEL" --scale "$SCALE"
}

# ---------------------------------------------------------------- step 5: drift
step_drift() {
  if step_done "$DRIFT_JSON"; then say "STEP 5 drift: skip ($DRIFT_JSON exists)"; return 0; fi
  say "STEP 5 drift: llama-perplexity prefill, base vs merged (control corpus)"
  if [ "$DRY" -eq 1 ]; then
    printf 'DRY  uv run python scripts/refusal_ppl_screen.py --artifact base=%s --artifact %s=%s --ctx 512 --threads 16 --out %s\n' \
      "$(quote_args "$BASE_LINK")" "$ARM_LABEL" "$(quote_args "$MERGED")" "$DRIFT_SCREEN"
    printf 'DRY  uv run python %s drift --screen %s --out %s --artifact %s --label %s --budget 0.02\n' \
      "$OUT/cert_inputs.py" "$DRIFT_SCREEN" "$DRIFT_JSON" "$ARM_LABEL" "$ARM_LABEL"
    return 0
  fi
  uv run python scripts/refusal_ppl_screen.py \
    --artifact "base=$BASE_LINK" --artifact "$ARM_LABEL=$MERGED" \
    --ctx 512 --threads 16 --out "$DRIFT_SCREEN"
  uv run python "$OUT/cert_inputs.py" drift --screen "$DRIFT_SCREEN" --out "$DRIFT_JSON" \
    --artifact "$ARM_LABEL" --label "$ARM_LABEL" --budget 0.02
}

# ---------------------------------------------------------------- step 6: waves
run_wave() {  # name suite kinds cap serve_config parallel
  local name=$1 suite=$2 kinds=$3 cap=$4 config=$5 parallel=${6:-4}
  local dir="$WAVES/$name"
  if wave_done "$dir"; then say "wave $name: skip (predictions + manifest present)"; return 0; fi
  if [ "$DRY" -eq 1 ]; then
    printf 'DRY  uv run python scripts/capability_wave.py %s %s --label %s --kinds %s --max-tokens %s --parallel %s --preview 8 --serve-config %s --port %s\n' \
      "$(quote_args "$suite")" "$dir" "$name" "$kinds" "$cap" "$parallel" "$(quote_args "$config")" "$PORT"
    return 0
  fi
  if [ "$FORCE" -eq 1 ] && [ -d "$dir" ]; then rm -rf "$dir"; fi
  mkdir -p "$dir"
  say "wave $name: $suite kinds=$kinds cap=$cap parallel=$parallel"
  uv run python scripts/capability_wave.py "$suite" "$dir" \
    --label "$name" --kinds "$kinds" --max-tokens "$cap" --parallel "$parallel" \
    --preview 8 --serve-config "$config" --port "$PORT"
  uv run python - "$dir/wave.json" "$name" <<'PY'
import json, sys
wave = json.load(open(sys.argv[1]))
if wave.get('measured') is not True:
    raise SystemExit(f"{sys.argv[2]}: wave is NOT measured (invalid rows); "
                     "the screen never happened")
print(f"{sys.argv[2]}: {wave['rows']} rows, refusals {wave['refusals_detected']}, "
      f"capability {wave['capability_passed']}, invalid {wave['invalid']}")
PY
}

step_waves() {
  say "STEP 6 waves"

  if any_wave_pending "$W_BASE_REMOVAL" "$W_BASE_RETENTION"; then
    if [ "$DRY" -eq 0 ]; then mkdir -p "$LOG_DIR" "$WAVES"; fi
    start_server base-4096p4 "${argv_4096_base[@]}"
    run_wave base-removal-4096p4 "$REMOVAL_SUITE" refusal "$CAP_REMOVAL" "$CFG_BASE"
    run_wave base-retention-4096p4 "$RETENTION_SUITE" math,code "$CAP_RETENTION" "$CFG_BASE"
    stop_servers
  else
    say "base waves: skip (both waves present)"
  fi

  if any_wave_pending "$W_REMOVAL" "$W_HELDOUT" "$W_RETENTION"; then
    start_server merged-4096p4 "${argv_4096_merged[@]}"
    run_wave removal-4096p4 "$REMOVAL_SUITE" refusal "$CAP_REMOVAL" "$CFG_4096"
    run_wave heldout-4096p4 "$HELDOUT_SUITE" refusal "$CAP_HELDOUT" "$CFG_4096"
    run_wave retention-4096p4 "$RETENTION_SUITE" math,code "$CAP_RETENTION" "$CFG_4096"
    stop_servers
  else
    say "merged 4096/4 waves: skip (all three present)"
  fi

  if wave_pending "$W_REMOVAL_LOW"; then
    start_server merged-2048p2 "${argv_2048_merged[@]}"
    run_wave removal-2048p2 "$REMOVAL_SUITE" refusal "$CAP_REMOVAL" "$CFG_2048" 2
    stop_servers
  else
    say "merged 2048/2 comparison wave: skip (present)"
  fi
}

# ---------------------------------------------------------------- step 7: shape audits
step_audits() {
  if step_done "$AUDIT_REMOVAL" && step_done "$AUDIT_HELDOUT"; then
    say "STEP 7 shape audits: skip (both present)"; return 0
  fi
  say "STEP 7 shape audits on the merged waves (R8: the served bytes)"
  local digest
  if [ "$DRY" -eq 1 ]; then
    printf 'DRY  uv run python scripts/refusal_shape_audit.py removal=%s --out %s\n' \
      "$(quote_args "$W_REMOVAL/predictions.jsonl")" "$AUDIT_REMOVAL"
    printf 'DRY  uv run python scripts/refusal_shape_audit.py held_out=%s --out %s\n' \
      "$(quote_args "$W_HELDOUT/predictions.jsonl")" "$AUDIT_HELDOUT"
    printf 'DRY  uv run python %s audit-identity --audit %s --digest <identity digest from spec>\n' \
      "$OUT/cert_inputs.py" "$AUDIT_REMOVAL"
    return 0
  fi
  uv run python scripts/refusal_shape_audit.py \
    "removal=$W_REMOVAL/predictions.jsonl" --out "$AUDIT_REMOVAL"
  uv run python scripts/refusal_shape_audit.py \
    "held_out=$W_HELDOUT/predictions.jsonl" --out "$AUDIT_HELDOUT"
}

# ---------------------------------------------------------------- step 8: config pair
step_comparison() {
  if step_done "$COMPARISON"; then say "STEP 8 config pair: skip ($COMPARISON exists)"; return 0; fi
  say "STEP 8 config pair: 4096/4 vs 2048/2 on the same bytes (R9.4)"
  plan uv run python scripts/cert_config_pair.py \
    --a "$W_REMOVAL" --b "$W_REMOVAL_LOW" \
    --label-a "ctx 4096 / parallel 4 / cap $CAP_REMOVAL" \
    --label-b "ctx 2048 / parallel 2 / cap $CAP_REMOVAL" \
    --out "$COMPARISON"
}

# ---------------------------------------------------------------- step 9: spec
step_spec() {
  if step_done "$SPEC"; then say "STEP 9 spec: skip ($SPEC exists)"; return 0; fi
  say "STEP 9 spec: assemble the release-certify spec"
  local sha bytes tensors footprint recipe
  if [ "$DRY" -eq 1 ]; then
    sha='<merged sha256>'; bytes='<merged bytes>'; tensors='<merge.json tensors_changed>'
  else
    sha=$(sha256_of "$MERGED")
    bytes=$(stat -c %s "$MERGED")
    tensors=$(uv run python -c "import json,sys; print(int(json.load(open(sys.argv[1])).get('tensors_changed') or 0))" "$MERGE_JSON")
    [ "$tensors" -gt 0 ] || die "merge.json reports 0 changed tensors; refusing to write a spec"
  fi
  footprint="blk.0-23 x {attn_output.weight, ffn_down.weight} ($tensors residual-write tensors discovered by --gguf-targets auto)"
  recipe="python -m jdewitt.spanfit build --gguf-in base.gguf --out minicpm-v47-$ARM_LABEL-Q4_K_M.gguf --axes refit-axes.pt --gguf-targets auto --shape perlayer --axis-space output --scale $SCALE"
  spec_argv=(uv run python "$OUT/cert_inputs.py" spec --out "$SPEC"
    --artifact "$MERGED" --artifact-sha "$sha" --artifact-bytes "$bytes"
    --base "$BASE_LINK" --base-sha "$BASE_SHA_PIN" --base-bytes "$BASE_BYTES_PIN"
    --recipe "$recipe" --scale "$SCALE" --arm-label "$ARM_LABEL"
    --edited-tensors "$tensors"
    --edit "per-layer sign-aligned refit axes (REFIT: no saved capture), output space, all 24 layers, --gguf-targets auto (attn out + ffn down), scale $SCALE"
    --footprint "$footprint"
    --removal-suite "$REMOVAL_SUITE" --heldout-suite "$HELDOUT_SUITE"
    --retention-suite "$RETENTION_SUITE"
    --config-removal "$CFG_4096" --config-heldout "$CFG_4096"
    --config-retention "$CFG_4096" --pinned-argv "$CFG_4096"
    --removal-wave "$W_REMOVAL" --heldout-wave "$W_HELDOUT"
    --retention-wave "$W_RETENTION" --retention-base-wave "$W_BASE_RETENTION"
    --audit-removal "$AUDIT_REMOVAL" --audit-heldout "$AUDIT_HELDOUT"
    --drift "$DRIFT_JSON" --fidelity "$FIDELITY_JSON" --revision "$HF_REVISION"
    --cap-removal "$CAP_REMOVAL" --cap-heldout "$CAP_HELDOUT"
    --cap-retention "$CAP_RETENTION")
  if [ "$DRY" -eq 1 ]; then
    printf 'DRY  %s\n' "$(quote_args "${spec_argv[@]}")"
    printf 'DRY  uv run python %s audit-identity --audit %s --digest <artifact identity digest from spec.json>\n' \
      "$OUT/cert_inputs.py" "$AUDIT_REMOVAL"
    printf 'DRY  uv run python %s audit-identity --audit %s --digest <artifact identity digest from spec.json>\n' \
      "$OUT/cert_inputs.py" "$AUDIT_HELDOUT"
    return 0
  fi
  "${spec_argv[@]}"
  local digest
  digest=$(uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))['artifact_identity_digest'])" "$SPEC")
  uv run python "$OUT/cert_inputs.py" audit-identity --audit "$AUDIT_REMOVAL" --digest "$digest"
  uv run python "$OUT/cert_inputs.py" audit-identity --audit "$AUDIT_HELDOUT" --digest "$digest"
  say "spec written: $SPEC (artifact identity $digest)"
}

# ---------------------------------------------------------------- summary
step_summary() {
  [ "$DRY" -eq 1 ] && return 0
  uv run python - "$COMPARISON" "$W_BASE_REMOVAL" "$W_HELDOUT" "$W_RETENTION" "$DRIFT_JSON" "$FIDELITY_JSON" <<'PY' || true
import json, os, sys
def maybe(path):
    return json.load(open(path)) if os.path.isfile(path) else None
pair = maybe(sys.argv[1])
if pair:
    print(f"removal 4096/4 (certified): {pair['n_m_a']}   "
          f"removal 2048/2 (beside it): {pair['n_m_b']}   "
          f"boundary rows: {len(pair['boundary'])}   robustly removed: {pair['robustly_removed_a']}")
    if pair['boundary']:
        print('boundary ids: ' + ', '.join(pair['boundary']))
base = maybe(os.path.join(sys.argv[2], 'wave.json'))
if base:
    print(f"base at 4096/4: {base['refusals_detected']}/{base['rows']} refusals "
          f"(the pre-edit rate at the same config)")
held = maybe(os.path.join(sys.argv[3], 'wave.json'))
if held:
    print(f"held-out: {held['rows']} rows over 24 families, refusals {held['refusals_detected']}")
ret = maybe(os.path.join(sys.argv[4], 'wave.json'))
if ret:
    print(f"retention: {ret['rows']} rows, capability {ret['capability_passed']}, invalid {ret['invalid']}")
drift, fid = maybe(sys.argv[5]), maybe(sys.argv[6])
if drift:
    print(f"drift: control {drift['control_delta_ln_ppl']} ln ppl in budget {drift['budget']} "
          f"({drift['inside_budget']})")
if fid:
    print(f"fidelity: min cos {fid['min_cos']} mean {fid['mean_cos']} over {fid['sampled']} tensors "
          f"(arm_measurement {fid['arm_measurement']})")
print()
print('next (analysis card, from the repo root):')
print('  uv run jdewitt release-certify --spec verification/p7-minicpm/spec.json --out verification/releases/minicpm-v47/packet.json')
print('  uv run jdewitt verify-artifact verification/releases/minicpm-v47/packet.json --name packet.json')
PY
}

# ---------------------------------------------------------------- main
if [ "$DRY" -eq 1 ]; then
  printf 'P7 MiniCPM-V 4.7 1B cert plan (no server is started, no file is written)\n\n'
else
  mkdir -p "$OUT" "$LOG_DIR" "$WAVES"
  say "=== P7 minicpm cert wave run: base=$BASE_PATH arm=$ARM_LABEL scale=$SCALE ==="
fi

step_operands
step_refit
step_merge
step_fidelity
step_drift
step_waves
step_audits
step_comparison
step_spec
step_summary

if [ "$DRY" -eq 0 ]; then
  stop_servers
  say "=== done: waves in $WAVES, spec $SPEC; every server started here is stopped ==="
fi
