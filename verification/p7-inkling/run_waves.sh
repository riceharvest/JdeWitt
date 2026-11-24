#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# P7 cert: Inkling-2-Mini IQ4_NL -- every step the Phase 7 packet needs, in order.
#
# ONE script, resumable, launched by the GPU card as
#     bash verification/p7-inkling/run_waves.sh
# and by the analysis card only through the saved artifacts it leaves behind.
#
# What it does, in order, each step skipped when its output already exists:
#   0. operands   sha256-check every pinned operand (operands.json); restore the
#                 registered removal suite from its in-clone twin when the host
#                 cache lost it, and check the restored bytes against the pin.
#   1. capture    capture residual records on the SOURCE weights
#                 (`jdewitt capture`, locus first-answer per METHODOLOGY R4). If the
#                 checkpoint's architecture cannot be loaded by this stack it writes
#                 capture-blocked.json with the measured error and refuses
#                 CAPTURE_UNSUPPORTED (exit 77) -- it never invents an axis and never
#                 serves the parent IQ4_NL as if it were the arm.
#   2. axis       `python -m jdewitt.spanfit axes --shape shared` -> shared-axes.pt
#   3. arm        merge the shared axis into the IQ4_NL bytes with a PER-TENSOR type
#                 choice: each edited tensor keeps the artifact's own IQ4_NL grid
#                 unless the delivered cos falls below 0.90, and only then climbs
#                 q6_k -> q8_0 -> f16. Every edited tensor is re-measured on the
#                 SHIPPED bytes (ggufio.compare_edit) -> codec-table.json.
#   4. waves      serve one artifact at a time at the pinned cert argv
#                 (-c 4096 --parallel 4, generation cap 1024): base retention (241),
#                 then arm harmful-40 (both wording inventories are stored by the
#                 shipped scorer), arm held-out (registered 120-family set), arm
#                 retention (241). The server this script started is stopped before
#                 the next artifact is served and after the last wave.
#   5. audits     refusal shape audit (R6) over both saved refusal waves; the audit
#                 documents are stamped with the artifact identity digest (R8).
#   6. drift      base vs arm on the ppl-probe control corpus (llama-perplexity).
#   7. spec       verification/releases/inkling-2-mini/spec.json, assembled from what
#                 was measured, plus the exact next command for the analysis card.
#
# Usage:
#   bash verification/p7-inkling/run_waves.sh            # run / resume
#   bash verification/p7-inkling/run_waves.sh --dry      # print every argv, execute nothing
#   bash verification/p7-inkling/run_waves.sh --check    # operands + preflight, no writes
#
# Environment (defaults first):
#   P7_PYTHON  repo python              .venv/bin/python
#   P7_MODELS  GGUF directory           /mnt/ssd/models/Inkling-2-Mini-GGUF
#   P7_SRC     source checkpoint        /mnt/ssd/models/Inkling-2-Mini
#   P7_SERVE   llama-server binary      first executable of the candidate list below
#   P7_PPL     llama-perplexity binary  first executable of the candidate list below
#   P7_NGL     -ngl                     999
#   P7_PORT    server port              8134
#   P7_SCALE   merge strength           1.5  (inside the registered grid 1.0-2.0)
#   P7_LAYERS  footprint for the merge  unset = every discovered residual-write layer
#   P7_RECORDS capture records .pt      verification/p7-inkling/capture-records.pt
#
# Nothing here downloads weights, edits an operand in place, or grades a row by hand.
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUT="$ROOT/verification/p7-inkling"
RELEASE="$ROOT/verification/releases/inkling-2-mini"
PROGRESS="$OUT/progress.txt"
RUNLOG="$OUT/run.log"

PY=${P7_PYTHON:-$ROOT/.venv/bin/python}
MODELS=${P7_MODELS:-/mnt/ssd/models/Inkling-2-Mini-GGUF}
SRC=${P7_SRC:-/mnt/ssd/models/Inkling-2-Mini}
BASE_ART="$MODELS/Inkling-2-Mini-IQ4_NL.gguf"
ARM_ART="$MODELS/Inkling-2-Mini-IQ4_NL-edited.gguf"
RECORDS=${P7_RECORDS:-$OUT/capture-records.pt}
AXES="$OUT/shared-axes.pt"
PORT=${P7_PORT:-8134}
NGL=${P7_NGL:-999}
SCALE=${P7_SCALE:-1.5}
LAYERS_ARG=${P7_LAYERS:-}
CTX=4096
PARALLEL=4
CAP=1024                 # 4096 ctx / 4 slots: a larger per-row cap cannot fit the slot
DRIFT_BUDGET=0.02
CODEC_FLOOR=0.90

# Registered suites (verification/releases/matrix.manifest.json -> registered.suites).
SUITE_REMOVAL=/mnt/ssd/hermes/cache/scratch/probes/harmful-40.jsonl
SUITE_REMOVAL_TWIN="$ROOT/verification/screens/advbench-40.jsonl"
SUITE_HELDOUT="$ROOT/verification/screens/q3-holdout-wider.jsonl"
SUITE_RETENTION="$ROOT/verification/capability/capability-run-en.jsonl"
SUITE_CAPTURE="$OUT/capture-suite.jsonl"

OPERANDS="$OUT/operands.json"
WAVE_REMOVAL="$OUT/harmful40-arm"
WAVE_HELDOUT="$OUT/holdout-arm"
WAVE_ARM_RET="$OUT/retention-arm"
WAVE_BASE_RET="$OUT/retention-base"
AUDIT_REMOVAL="$OUT/shape-audit-harmful40.json"
AUDIT_HELDOUT="$OUT/shape-audit-holdout.json"
PPL_SCREEN="$OUT/ppl-screen.json"
DRIFT="$OUT/drift.json"
CODEC="$OUT/codec-table.json"
FIDELITY="$OUT/fidelity.json"
BUILD_REPORT="$OUT/build.json"
SPEC="$RELEASE/spec.json"

SERVE_BIN=""
SERVER_PID=""
SERVED_ARGV=""

say()  { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$RUNLOG" >&2; }
mark() { printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$PROGRESS"; }
die()  { local code=$1; shift; say "REFUSED($code): $*"; mark REFUSED "$code: $*"; exit "$code"; }
finished() { [[ -s $1 ]]; }

# --------------------------------------------------------------- server argv/state
# One array is the single source of truth for the pinned cert argv: the same tokens
# launch the server and are recorded in wave.json (R9).
serve_argv() {  # ARTIFACT [NGL] -> array in REPLY_ARGV, string in REPLY
  local art=$1 ngl=${2:-$NGL}
  REPLY_ARGV=(-m "$art" -c "$CTX" --parallel "$PARALLEL" --jinja --no-webui
              --host 127.0.0.1 --port "$PORT" -ngl "$ngl" --temp 0)
  REPLY="${REPLY_ARGV[*]}"
}

resolve_bin() {  # NAME candidates... -> REPLY
  local name=$1; shift
  local var
  var=$(printf 'P7_%s' "$name" | tr '[:lower:]' '[:upper:]')
  if [[ -n ${!var:-} ]]; then REPLY=${!var}; return 0; fi
  local cand
  for cand in "$@"; do
    if [[ -x $cand ]]; then REPLY=$cand; return 0; fi
  done
  REPLY=""
  return 1
}

wait_health() {  # ARTIFACT PID PID-DESCRIPTION
  "$PY" - "$PORT" "$1" "$2" "$3" <<'PY'
import json, os, sys, time, urllib.request
port, wanted, pid, where = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
deadline = time.time() + 2700          # a 17 GiB MoE read from disk is slow, not hung
last = 'no response'
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5) as r:
            if r.status == 200:
                break
    except Exception as error:
        last = f'{type(error).__name__}: {error}'
    # A server that died (unknown architecture, OOM, bad argv) never becomes healthy:
    # waiting the full load budget on it burns a GPU reservation for nothing.
    try:
        os.kill(pid, 0)
    except OSError:
        print(f'{where}: pid {pid} exited before /health answered ({last})', file=sys.stderr)
        raise SystemExit(2)
    time.sleep(5)
else:
    print(f'health timeout on {where}: {last}', file=sys.stderr)
    raise SystemExit(1)
try:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/props', timeout=10) as r:
        doc = json.loads(r.read() or b'{}')
except Exception as error:
    print(f'/props did not answer on {where}: {error}', file=sys.stderr)
    raise SystemExit(1)
served = doc.get('model_path') or doc.get('path') or ''
if served and os.path.realpath(served) != os.path.realpath(wanted):
    print(f'{where} serves {served}, not {wanted} -- refusing to attribute a wave to it',
          file=sys.stderr)
    raise SystemExit(1)
print(f'healthy: {where} is serving {served or wanted}')
PY
}

stop_server() {
  [[ -n $SERVER_PID ]] || return 0
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    say "stopping llama-server pid $SERVER_PID"
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      say "pid $SERVER_PID ignored SIGTERM; sending SIGKILL"
      kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
  SERVED_ARGV=""
  rm -f "$OUT/serve/server.pid"
}

served_model() {
  "$PY" - "$PORT" <<'PY' 2>/dev/null || true
import json, sys, urllib.request
try:
    with urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/props', timeout=10) as r:
        print(json.loads(r.read() or b'{}').get('model_path') or '')
except Exception:
    print('')
PY
}

start_server() {  # ARTIFACT
  local art=$1 up logfile
  if [[ -n $SERVER_PID ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    up=$(served_model)
    if [[ -n $up && $up -ef $art ]]; then return 0; fi
    stop_server
  fi
  [[ -n $SERVE_BIN ]] || die 72 SERVE_NOT_FOUND "no llama-server binary; set P7_SERVE. Measured on this host: the Inkling fork builds (/mnt/ssd/dev/inkling-rt{,-t57174e1b,-fadec}/llama.cpp/build-sycl/bin) carry llama-cli/llama-bench/llama-perplexity and libllama-server-impl.so, but no llama-server executable, and a stock llama.cpp does not know inkling2_loop. Build that one target in the fork before serving; do not swap in a build that cannot load the architecture."
  [[ -s $art ]] || die 78 ARTIFACT_MISSING "$art does not exist"
  mkdir -p "$OUT/serve"
  logfile="$OUT/serve/$(basename "${art%.gguf}").server.log"
  serve_argv "$art"
  SERVED_ARGV=$REPLY
  say "serving $(basename "$art"): $SERVE_BIN $SERVED_ARGV"
  "$SERVE_BIN" "${REPLY_ARGV[@]}" >"$logfile" 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" > "$OUT/serve/server.pid"
  printf '%s\n' "$SERVED_ARGV" > "$OUT/serve/$(basename "${art%.gguf}").argv"
  local health
  health=0
  wait_health "$art" "$SERVER_PID" "llama-server pid $SERVER_PID" || health=$?
  if [[ $health -eq 0 ]]; then
    mark serve "ok $(basename "$art") pid=$SERVER_PID"
    return 0
  fi
  if [[ $health -eq 2 ]]; then
    # It exited: that is not a slow load, it is a server that cannot run this argv.
    tail -n 25 "$logfile" >&2 || true
    stop_server
    die 74 SERVE_EXITED "llama-server exited while loading $art (see $logfile); a server that cannot load the artifact is not a slow load"
  fi
  # One documented fallback: -ngl 999 may not fit the reservation. Retry once at
  # -ngl 0 and RECORD that argv, because the wave's serve_config has to name the
  # config it really ran under (R9).
  say "load failed at -ngl $NGL; retrying once at -ngl 0 (recorded as a fallback)"
  tail -n 20 "$logfile" >&2 || true
  stop_server
  serve_argv "$art" 0
  SERVED_ARGV=$REPLY
  "$SERVE_BIN" "${REPLY_ARGV[@]}" >>"$logfile" 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" > "$OUT/serve/server.pid"
  printf '%s\n' "$SERVED_ARGV" > "$OUT/serve/$(basename "${art%.gguf}").argv"
  wait_health "$art" "$SERVER_PID" "llama-server pid $SERVER_PID (ngl 0 fallback)" \
    || die 74 SERVE_UNHEALTHY "llama-server never became healthy at -ngl 0; see $logfile"
  mark serve "ok-fallback-ngl0 $(basename "$art") pid=$SERVER_PID"
}

# ------------------------------------------------------------------------ 0 operands
step_operands() {
  finished "$OPERANDS" || die 73 OPERANDS_UNPINNED "$OPERANDS is missing; the pins are this card's output, not a guess"
  if [[ ! -s $SUITE_REMOVAL ]]; then
    [[ -s $SUITE_REMOVAL_TWIN ]] || die 78 SUITE_MISSING "$SUITE_REMOVAL and its twin $SUITE_REMOVAL_TWIN are both missing"
    mkdir -p "$(dirname "$SUITE_REMOVAL")"
    cp -- "$SUITE_REMOVAL_TWIN" "$SUITE_REMOVAL"
    say "restored $SUITE_REMOVAL from the in-clone twin $(basename "$SUITE_REMOVAL_TWIN")"
    mark restore "removal suite <- $(basename "$SUITE_REMOVAL_TWIN")"
  fi
  "$PY" - "$OPERANDS" <<'PY'
import hashlib, json, sys
from pathlib import Path
doc = json.loads(Path(sys.argv[1]).read_text())
bad = []
for item in doc['operands']:
    path = Path(item['path'])
    if not path.exists():
        if item.get('required', True):
            bad.append(f"missing: {path}")
        continue
    if item.get('bytes') and path.stat().st_size != item['bytes']:
        bad.append(f"{path}: {path.stat().st_size} bytes, pinned {item['bytes']}")
        continue
    if item.get('sha256'):
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                digest.update(chunk)
        if digest.hexdigest() != item['sha256']:
            bad.append(f"{path}: sha256 {digest.hexdigest()}, pinned {item['sha256']}")
if bad:
    raise SystemExit('OPERAND_MISMATCH\n  ' + '\n  '.join(bad))
print(f"operands ok: {len(doc['operands'])} checked")
PY
  mark operands "sha256 verified"
}

# ------------------------------------------------------------------------- 1 capture
step_capture() {
  if finished "$RECORDS"; then
    say "skip capture: $RECORDS exists"
    return 0
  fi
  mkdir -p "$OUT"
  if [[ ! -s $SUITE_CAPTURE ]]; then
    # Deterministic mixed suite: the registered 40 harmful rows give the 'bad' label
    # and 32 harmless rows give the 'good' one, so the axis is a contrast and not a
    # topic list. 64 rows keeps the capture inside the GPU reservation.
    "$PY" - "$SUITE_REMOVAL" "$ROOT/verification/harsh-suites/selection.jsonl" "$SUITE_CAPTURE" <<'PY'
import json, sys
from pathlib import Path
removal, selection, out = (Path(p) for p in sys.argv[1:4])
rows = [json.loads(line) for line in removal.read_text().splitlines() if line.strip()]
harmless = [json.loads(line) for line in selection.read_text().splitlines()
            if line.strip() and json.loads(line).get('kind') == 'harmless'][:32]
picked = rows + harmless
out.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in picked))
print(f'capture suite: {len(rows)} refusal + {len(harmless)} harmless -> {out}')
PY
    mark capture-suite "written $SUITE_CAPTURE"
  fi
  # Architecture probe BEFORE spending a GPU. A checkpoint this stack cannot build
  # has no capture and therefore no axis: that is CAPTURE_UNSUPPORTED with the
  # measured reason, never a zero and never a parent served as an arm.
  if ! "$PY" - "$SRC" "$OUT" <<'PY'
import json, sys, traceback
from pathlib import Path
source, out = sys.argv[1], Path(sys.argv[2])
report = {'schema_version': 1, 'source': source, 'measured': True}
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    report['config_class'] = type(config).__name__
    report['model_type'] = getattr(config, 'model_type', None)
    report['supported'] = True
except Exception as error:
    report['supported'] = False
    report['error'] = f'{type(error).__name__}: {error}'
    report['traceback_tail'] = traceback.format_exc().splitlines()[-3:]
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        report['config_mapping_inkling_keys'] = sorted(k for k in CONFIG_MAPPING if 'inkling' in k)
    except Exception:
        pass
    cfg = Path(source) / 'config.json'
    if cfg.is_file():
        doc = json.loads(cfg.read_text())
        report['checkpoint'] = {'model_type': doc.get('model_type'),
                                'architectures': doc.get('architectures'),
                                'text_model_type': (doc.get('text_config') or {}).get('model_type'),
                                'quantization_config': doc.get('quantization_config'),
                                'has_auto_map': bool(doc.get('auto_map'))}
    (out / 'capture-blocked.json').write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(json.dumps(report, indent=1))
    raise SystemExit(1)
print(json.dumps(report))
PY
  then
    die 77 CAPTURE_UNSUPPORTED "this stack cannot load $(basename "$SRC") (measured; see capture-blocked.json): no residual records -> no fitted axis -> no arm. Capture is the first axis to fail; the analysis card records failed(capture) with these numbers."
  fi
  say "capturing residual records on the source weights: $SRC"
  "$PY" -m jdewitt capture \
    --model "$SRC" --suite "$SUITE_CAPTURE" --out "$RECORDS" --gguf "$BASE_ART" \
    --kinds refusal,harmless --locus first-answer --max-tokens 96 --ctx-size 4096 \
    --limit 64 2>&1 | tee -a "$RUNLOG"
  finished "$RECORDS" || die 77 CAPTURE_FAILED "capture wrote no records"
  mark capture "records $(basename "$RECORDS")"
}

# ---------------------------------------------------------------------------- 2 axis
step_axes() {
  if finished "$AXES"; then
    say "skip axis fit: $AXES exists"
    return 0
  fi
  finished "$RECORDS" || die 77 CAPTURE_UNSUPPORTED "no capture records; there is no axis to fit"
  "$PY" -m jdewitt.spanfit axes --records "$RECORDS" --out "$AXES" --shape shared \
    2>&1 | tee -a "$RUNLOG"
  finished "$AXES" || die 79 AXIS_FAILED "spanfit axes wrote no axes tensor"
  mark axis "shared axis $(basename "$AXES") sha=$(sha256sum "$AXES" | cut -c1-16)"
}

# ----------------------------------------------------------------------------- 3 arm
step_arm() {
  if finished "$ARM_ART" && finished "$CODEC" && finished "$FIDELITY" && finished "$BUILD_REPORT"; then
    say "skip arm: $ARM_ART and its codec table exist"
    return 0
  fi
  [[ -s $BASE_ART ]] || die 78 ARTIFACT_MISSING "base artifact $BASE_ART does not exist"
  finished "$AXES" || die 79 AXIS_FAILED "no shared axis at $AXES; refusing to merge an unfitted direction"
  "$PY" - "$ROOT" "$BASE_ART" "$ARM_ART" "$AXES" "$SCALE" "$CODEC_FLOOR" \
          "${LAYERS_ARG}" "$RECORDS" "$BUILD_REPORT" "$CODEC" "$FIDELITY" <<'PY'
"""Merge one shared axis into the IQ4_NL bytes, per-tensor type from measured cos.

IQ4_NL is the type the user already serves, so the default answer is to re-encode
the edited tensors at that same grid. A tensor climbs q6_k -> q8_0 -> f16 only when
the IQ grid cannot hold the delta (delivered cos < floor). The choice is made from a
codec round-trip measurement of the exact array that will be written, and the FINAL
number recorded is re-measured on the shipped bytes with ggufio.compare_edit.
"""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
import torch

(root, base, arm, axes_p, scale, floor, layers_arg, records_p, build_p, codec_p, fid_p) = sys.argv[1:12]
scale, floor = float(scale), float(floor)
sys.path.insert(0, root)
from jdewitt.gguf_adapters import discover_writeback_targets, gguf_weight_shapes, inferred_hidden_width
from jdewitt.ggufio import axis_delta, compare_edit, decode_values
from jdewitt.quants import CODECS, type_name_for_id
from jdewitt.rewrite import rewrite_with_edits
from jdewitt.ggufwrite import parse_container, tensor_nbytes
from jdewitt.spanfit import parse_layer_ids


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0


axes = np.asarray(torch.load(axes_p, map_location='cpu', weights_only=False)['axes'], np.float32)
width = int(axes.shape[-1])
shapes = gguf_weight_shapes(base)
discovered = discover_writeback_targets(shapes, width)
layers = sorted(discovered['targets'])
if layers_arg:
    wanted = set(parse_layer_ids(layers_arg))
    layers = [layer for layer in layers if layer in wanted]
if not layers:
    raise SystemExit(json.dumps({'error': {
        'code': 'NO_VERIFIED_ADAPTER',
        'message': (f'{base} carries no residual-write projection at width {width} '
                    f'(its own writeback names put the hidden width at '
                    f'{inferred_hidden_width(shapes)}); the records and the artifact do '
                    'not describe the same model'),
        'artifact_hidden_width': inferred_hidden_width(shapes), 'records_width': width}}))

names = [name for layer in layers[:int(axes.shape[0])] for name in discovered['targets'][layer]]
edits = {name: (axes[int(name.split('.')[1])], 'output') for name in names}
header = parse_container(base)
by_name = {entry['name']: entry for entry in header['tensors']}
missing = [name for name in names if name not in by_name]
if missing:
    raise SystemExit(json.dumps({'error': {
        'code': 'NO_VERIFIED_ADAPTER',
        'message': f'the axis footprint names tensors this artifact does not carry: {missing[:4]}'}}))

# Decode each edited tensor once, probe the codec round-trip for every candidate.
candidates = ['IQ4_NL', 'Q6_K', 'Q8_0', 'F16']
probe, chosen = [], {}
with open(base, 'rb') as handle:
    for name in names:
        tensor = by_name[name]
        native = type_name_for_id(tensor['qtype'])
        handle.seek(header['data_base'] + tensor['offset'])
        raw = np.frombuffer(handle.read(tensor_nbytes(tensor['dims'], tensor['qtype'])), np.uint8)
        matrix = decode_values(raw, native).astype(np.float32).reshape(
            tuple(reversed(tensor['dims'])))
        intended = axis_delta(edits[name][0], matrix, scale, 'output')
        order = [native] + [kind for kind in candidates if kind != native]
        row = {'tensor': name, 'source_type': native, 'probe': {}}
        pick = None
        for kind in order:
            codec = CODECS[kind]
            block, quantize, dequantize = codec[1], codec[2], codec[3]
            flat = np.ascontiguousarray((matrix + intended).reshape(-1), dtype=np.float32)
            pad = (-flat.size) % block
            if pad:
                flat = np.concatenate([flat, np.zeros(pad, np.float32)])
            back = np.asarray(dequantize(np.asarray(quantize(flat), np.uint8)), np.float32)
            realized = back[:matrix.size].reshape(matrix.shape) - matrix
            value = round(cos(realized, intended), 4)
            row['probe'][kind] = value
            if pick is None and value >= floor:
                pick = (kind, value)
        if pick is None:
            pick = ('F16', row['probe'].get('F16'))
        row['chosen_type'], row['probe_cos'] = pick
        chosen[name] = pick[0]
        probe.append(row)

# One pass per distinct type: each pass copies every other tensor byte for byte, so a
# chained pass never rewrites a tensor an earlier pass already wrote.
groups = {}
for name, kind in chosen.items():
    groups.setdefault(kind, []).append(name)
ordered = sorted(groups, key=lambda kind: candidates.index(kind) if kind in candidates else -1)
tmp_dir = Path(arm).with_suffix('.parts')
tmp_dir.mkdir(parents=True, exist_ok=True)
current = Path(base)
steps = []
for index, kind in enumerate(ordered):
    target = Path(arm) if index == len(ordered) - 1 else tmp_dir / f'pass{index}-{kind.lower()}.gguf'
    group = {name: edits[name] for name in groups[kind]}
    report = rewrite_with_edits(str(current), str(target), group, scale=scale,
                                axis_space='output', edit_type=kind.lower(),
                                verify_untouched=True)
    steps.append({'pass': index, 'edit_type': kind, 'tensors': len(group),
                  'bytes_out': report['bytes_out'],
                  'untouched_verified': report['tensors'] - report['tensors_edited']})
    current = target
if Path(arm) != current:
    shutil.move(str(current), arm)
shutil.rmtree(tmp_dir, ignore_errors=True)

# Re-measure on the SHIPPED bytes: the probe chose the type, this is what shipped.
delivered = compare_edit(base, arm, edits, scale=scale)
stored = {entry['name']: entry for entry in parse_container(arm)['tensors']}
rows, below = [], []
for row in delivered['tensors']:
    entry = stored[row['name']]
    item = {'tensor': row['name'],
            'chosen_type': type_name_for_id(entry['qtype']),
            'realized_cos': row['cos'],
            'byte_cost': tensor_nbytes(entry['dims'], entry['qtype']),
            'bytes_before': tensor_nbytes(by_name[row['name']]['dims'], by_name[row['name']]['qtype']),
            'snr': row['snr'], 'norm_ratio': row['norm_ratio'],
            'probe': probe[names.index(row['name'])]['probe']}
    rows.append(item)
    if item['realized_cos'] < floor:
        below.append(row['name'])
cosines = [row['realized_cos'] for row in rows]
footprint = (f'blk.{min(layers)}-{max(layers)} x '
             + ', '.join(sorted({name.split('.', 2)[-1] for name in names})))
table = {'schema_version': 1, 'lineage': 'inkling-2-mini', 'label': 'iq4nl-shared',
         'artifact': arm, 'artifact_bytes': Path(arm).stat().st_size,
         'edited_tensors': len(rows), 'footprint': footprint,
         'chosen_type': ', '.join(sorted({row['chosen_type'] for row in rows})),
         'scale': scale, 'floor': floor, 'grid': 'IQ4_NL first (the served grid)',
         'min_cos': min(cosines) if cosines else None,
         'mean_cos': round(sum(cosines) / len(cosines), 4) if cosines else None,
         'below_floor': below, 'codec_event': bool(below), 'tensors': rows,
         'note': ('realized cos is dequant(arm) - dequant(base) against the intended '
                  'projection, measured on the shipped bytes; "probe" is the codec '
                  'round-trip that chose the type. A tensor below the floor is a codec '
                  'event named here, never rounded away.')}
Path(codec_p).write_text(json.dumps(table, indent=1, sort_keys=True) + '\n')
arm_sha = sha256(arm)
Path(fid_p).write_text(json.dumps({
    'schema_version': 1, 'lineage': 'inkling-2-mini', 'label': 'iq4nl-shared',
    'measured': True, 'arm_measurement': True,
    'method': 'ggufio.compare_edit over every edited tensor of the served IQ4_NL arm',
    'artifact_sha256': arm_sha,
    'tensors': [{'name': row['tensor'], 'qtype': row['chosen_type'],
                 'cos': row['realized_cos'], 'snr': row['snr'],
                 'norm_ratio': row['norm_ratio']} for row in rows],
    'sampled': len(rows), 'min_cos': table['min_cos'], 'mean_cos': table['mean_cos'],
    'scope': 'every tensor the merge re-encoded, measured on the arm bytes that ship',
    'note': f'a low cos here means the codec, not the axis, decides what ships; floor {floor}',
}, indent=1, sort_keys=True) + '\n')
Path(build_p).write_text(json.dumps({
    'schema_version': 1, 'lineage': 'inkling-2-mini', 'base': base, 'arm': arm,
    'axes': axes_p, 'axes_sha256': sha256(axes_p), 'axes_shape': list(axes.shape),
    'scale': scale, 'floor': floor, 'layers': layers, 'tensors': names, 'passes': steps,
    'probe': probe, 'codec_table_sha256': sha256(codec_p), 'artifact_sha256': arm_sha,
    'records': records_p,
    'recipe': ('python -m jdewitt.spanfit axes --records RECORDS --out AXES --shape shared; '
               'then one rewrite_with_edits pass per chosen type (IQ4_NL first, then '
               'q6_k/q8_0/f16 only where the IQ grid could not hold the delta)'),
}, indent=1, sort_keys=True) + '\n')
print(json.dumps({'edited_tensors': len(rows), 'min_cos': table['min_cos'],
                  'mean_cos': table['mean_cos'], 'types': sorted({r['chosen_type'] for r in rows}),
                  'codec_event': table['codec_event'], 'below_floor': below[:8]}, indent=1))
PY
  finished "$ARM_ART" || die 79 ARM_BUILD_FAILED "the merge wrote no artifact"
  mark arm "built $(basename "$ARM_ART") $(stat -c%s "$ARM_ART") bytes"
}

# --------------------------------------------------------------------------- 4 waves
wave() {  # NAME ARTIFACT SUITE KINDS OUT-DIR
  local name=$1 art=$2 suite=$3 kinds=$4 out=$5
  local wave_json="$out/wave.json" preds="$out/predictions.jsonl" rows
  if finished "$wave_json" && finished "$preds"; then
    say "skip $name: $wave_json exists"
    return 0
  fi
  [[ -f $suite ]] || die 78 SUITE_MISSING "$name: $suite does not exist"
  if [[ -e $out ]]; then
    say "$name: $out exists without a complete wave; removing the partial directory"
    rm -rf -- "$out"
    mark delete "$name partial wave dir"
  fi
  start_server "$art"
  mkdir -p "$out"
  if ! "$PY" "$ROOT/scripts/capability_wave.py" "$suite" "$out" \
        --label "$name" --port "$PORT" --max-tokens "$CAP" --parallel "$PARALLEL" \
        --kinds "$kinds" --temperature 0 --preview 3 --serve-config "$SERVED_ARGV" \
        2>&1 | tee -a "$RUNLOG"; then
    die 76 WAVE_FAILED "$name returned non-zero; a wave that did not measure is UNMEASURED, not a score"
  fi
  finished "$wave_json" || die 76 WAVE_UNWRITTEN "$name wrote no wave.json"
  rows=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"])' "$wave_json")
  mark wave "$name rows=$rows kinds=$kinds cap=$CAP"
}

step_waves() {
  # Base first: retention needs a paired base wave, and two 17 GiB artifacts never
  # fit one card at once. The server is stopped between artifacts (R5: one at a time).
  wave retention-base "$BASE_ART" "$SUITE_RETENTION" math,code "$WAVE_BASE_RET"
  stop_server
  wave harmful40-arm "$ARM_ART" "$SUITE_REMOVAL" refusal "$WAVE_REMOVAL"
  wave holdout-arm "$ARM_ART" "$SUITE_HELDOUT" refusal "$WAVE_HELDOUT"
  wave retention-arm "$ARM_ART" "$SUITE_RETENTION" math,code "$WAVE_ARM_RET"
  stop_server
  # A refusal row without BOTH stored inventories cannot carry a two-inventory count
  # (release-certify refuses CERTIFY_ONE_INVENTORY). Fail here, where the wave is,
  # instead of in the analysis card.
  "$PY" - "$WAVE_REMOVAL/predictions.jsonl" "$WAVE_HELDOUT/predictions.jsonl" <<'PY'
import json, sys
for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    short = [row['id'] for row in rows
             if not isinstance(row.get('refusal_check_wide'), dict)
             or not isinstance(row.get('refusal_check'), dict)]
    if short:
        raise SystemExit(f'{path}: {len(short)} rows carry no stored shipped+wide refusal '
                         f'inventory ({short[:6]}); a degenerate row cannot be re-graded '
                         'after the fact, so this wave cannot carry a removal count')
print('both refusal waves carry shipped + wide inventories on every row')
PY
}

# -------------------------------------------------------------------------- 5 audits
step_audits() {
  if finished "$AUDIT_REMOVAL" && finished "$AUDIT_HELDOUT"; then
    say "skip shape audits: both exist"
    return 0
  fi
  finished "$WAVE_REMOVAL/predictions.jsonl" || die 76 WAVE_MISSING "$WAVE_REMOVAL/predictions.jsonl"
  finished "$WAVE_HELDOUT/predictions.jsonl" || die 76 WAVE_MISSING "$WAVE_HELDOUT/predictions.jsonl"
  "$PY" "$ROOT/scripts/refusal_shape_audit.py" \
    "harmful40-arm=$WAVE_REMOVAL/predictions.jsonl" --out "$AUDIT_REMOVAL" 2>&1 | tee -a "$RUNLOG"
  "$PY" "$ROOT/scripts/refusal_shape_audit.py" \
    "holdout-arm=$WAVE_HELDOUT/predictions.jsonl" --out "$AUDIT_HELDOUT" 2>&1 | tee -a "$RUNLOG"
  mark audit "harmful40 + holdout shape audits written"
}

# --------------------------------------------------------------------------- 6 drift
step_drift() {
  if finished "$DRIFT"; then say "skip drift: $DRIFT exists"; return 0; fi
  local ppl=${P7_PPL:-}
  if [[ -z $ppl ]]; then
    resolve_bin PPL \
      /mnt/ssd/dev/inkling-rt/llama.cpp/build-sycl/bin/llama-perplexity \
      "/mnt/ssd/dev workspace/llama.cpp-dspark/build-perp/bin/llama-perplexity" || true
    ppl=$REPLY
  fi
  [[ -n $ppl && -x $ppl ]] || die 72 PPL_NOT_FOUND "no llama-perplexity binary; set P7_PPL"
  if ! finished "$PPL_SCREEN"; then
    "$PY" "$ROOT/scripts/refusal_ppl_screen.py" --binary "$ppl" \
      --corpus-dir "$ROOT/verification/ppl-probe" \
      --artifact "base=$BASE_ART" --artifact "arm=$ARM_ART" --out "$PPL_SCREEN" 2>&1 | tee -a "$RUNLOG"
    mark drift "ppl screen taken with $ppl"
  fi
  "$PY" - "$PPL_SCREEN" "$DRIFT" "$DRIFT_BUDGET" <<'PY'
import json, sys
from pathlib import Path
screen, out, budget = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
doc = json.loads(screen.read_text())
delta = (doc['artifacts'].get('arm') or {}).get('delta_control')
if delta is None:
    raise SystemExit('REFUSED(DRIFT_UNMEASURED): the screen produced no control delta; '
                     'a drift reading without a control corpus is not a reading')
out.write_text(json.dumps({
    'schema_version': 1, 'lineage': 'inkling-2-mini', 'label': 'iq4nl-shared',
    'measured': True, 'instrument': 'llama-perplexity on the ppl-probe control corpus, base vs arm',
    'control_delta_ln_ppl': delta, 'budget': budget,
    'inside_budget': abs(float(delta)) <= budget,
    'source': screen.name, 'source_pointer': 'artifacts.arm.delta_control',
}, indent=1, sort_keys=True) + '\n')
print(f'drift: control {delta:+.4f} ln-ppl (budget {budget})')
PY
  mark drift "written $DRIFT"
}

# ---------------------------------------------------------------------------- 7 spec
step_spec() {
  [[ -s $ARM_ART ]] || die 78 ARM_NOT_BUILT "no arm artifact to describe"
  mkdir -p "$RELEASE"
  "$PY" - "$ROOT" "$RELEASE" "$BUILD_REPORT" "$CODEC" "$FIDELITY" "$AXES" \
          "$WAVE_REMOVAL" "$WAVE_HELDOUT" "$WAVE_ARM_RET" "$WAVE_BASE_RET" \
          "$SUITE_REMOVAL" "$SUITE_HELDOUT" "$SUITE_RETENTION" \
          "$DRIFT" "$BASE_ART" "$ARM_ART" "$AUDIT_REMOVAL" "$AUDIT_HELDOUT" \
          "$CTX" "$PARALLEL" "$CAP" "$SCALE" <<'PY'
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from jdewitt.certify import _identity_digest, inventory_digests

(root, release, build_p, codec_p, fid_p, axes_p, w_rem, w_hold, w_ret, w_base,
 suite_rem, suite_hold, suite_ret, drift_p, base_art, arm_art, audit_rem, audit_hold,
 ctx, parallel, cap, scale) = sys.argv[1:23]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    return sum(1 for line in Path(path).read_text().splitlines() if line.strip())


def serve_string(artifact, ngl=999):
    return (f'-m {artifact} -c {ctx} --parallel {parallel} --jinja --no-webui '
            f'--host 127.0.0.1 --port 8134 -ngl {ngl} --temp 0')


codec = json.loads(Path(codec_p).read_text())
build = json.loads(Path(build_p).read_text())
manifest = json.loads((Path(root) / 'verification/releases/matrix.manifest.json').read_text())
entry = next(item for item in manifest['lineages'] if item['lineage'] == 'inkling-2-mini')
arm_sha = sha256(arm_art)
# R9: the config each wave ran under is the one recorded beside the server; the served
# argv is copied verbatim into the spec so a -ngl fallback cannot hide in the prose.
recorded = Path(root, 'verification/p7-inkling/serve', f'{Path(arm_art).stem}.argv')
argv = recorded.read_text().strip() if recorded.is_file() else serve_string(arm_art)
pinned = (f'-m {arm_art} -c {ctx} --parallel {parallel} --jinja --no-webui '
          f'--host 127.0.0.1 --port 8134 --temp 0')
spec = {
    'schema_version': 1, 'lineage': 'inkling-2-mini',
    'arm': {
        'label': 'iq4nl-shared',
        'edited_tensors': int(codec['edited_tensors']),
        'edited_tensor_footprint': codec.get('footprint'),
        'chosen_type': codec.get('chosen_type'),
        'realized_cos': codec.get('min_cos'),
        'realized_cos_kind': 'arm-measurement (compare_edit on the served bytes)',
        'byte_cost': Path(arm_art).stat().st_size,
        'scale': float(scale),
        'codec_table': codec_p,
    },
    'artifact': {
        'path': arm_art, 'sha256': arm_sha, 'sha256_prefix_16': arm_sha[:16],
        'bytes': Path(arm_art).stat().st_size,
        'recipe': {
            'command': build.get('recipe'),
            'base_sha256_prefix_16': entry['base']['sha256_prefix_16'],
            'edit': ('shared sign-aligned axis fitted on captures of the source weights '
                     '(spanfit axes --shape shared, locus first-answer), merged into the '
                     'IQ4_NL bytes; each edited tensor keeps the IQ4_NL grid unless the '
                     'codec cannot hold the delta, then q6_k -> q8_0 -> f16'),
            'note': ('REFIT: the axis was fitted on this host from capture records; no prior '
                     'Inkling axis exists in this clone, and the recipe says so rather than '
                     'borrowing another lineage\u2019s'),
            'axes': {'path': axes_p, 'sha256': sha256(axes_p)},
            'records': {'path': build.get('records')} if build.get('records') else None,
        },
    },
    'base': {'path': base_art, 'sha256': entry['base']['sha256'],
             'sha256_prefix_16': entry['base']['sha256_prefix_16'],
             'bytes': Path(base_art).stat().st_size},
    'suites': {
        'removal': {'path': suite_rem, 'in_clone': False, 'sha256': sha256(suite_rem),
                    'rows': rows(suite_rem),
                    'note': ('the registered harmful-40 path, restored on this host from the '
                             'in-clone twin verification/screens/advbench-40.jsonl; the 40 ids '
                             'and prompts are the ones the qwen removal wave recorded')},
        'held_out': {'path': suite_hold, 'in_clone': True, 'sha256': sha256(suite_hold),
                     'rows': rows(suite_hold),
                     'note': 'the registered held-out set: 120 rows, 120 families'},
        'retention': {'path': suite_ret, 'in_clone': True, 'sha256': sha256(suite_ret),
                      'rows': rows(suite_ret), 'note': '180 maths + 61 code'},
    },
    'configs': {axis: {'argv': argv} for axis in ('removal', 'retention', 'held_out')},
    'pinned_cert_argv': pinned,
    'inputs': {
        'removal': {'wave': w_rem, 'suite': suite_rem},
        'held_out': {'wave': w_hold, 'suite': suite_hold},
        'retention': {'wave': w_ret, 'base_wave': w_base, 'suite': suite_ret},
        'shape_audit': [],
        'wording_inventory': inventory_digests(),
        'drift': drift_p, 'fidelity': fid_p,
    },
    'expected': {},
    'provenance': {
        'written_by': 'verification/p7-inkling/run_waves.sh',
        'spec_written_from': 'the measured waves, the arm codec table and the drift screen',
        'generation_cap': int(cap), 'ctx': int(ctx), 'parallel': int(parallel),
        'records_origin': 'capture on the source weights by this script',
    },
}
digest = _identity_digest(spec['artifact'])
# R8: an audit that does not say which artifact it read is not evidence.
for path, axis in ((Path(audit_rem), 'removal'), (Path(audit_hold), 'held_out')):
    doc = json.loads(path.read_text())
    doc['artifact_identity_digest'] = digest
    doc['artifact'] = {'path': arm_art, 'identity_digest': digest}
    for row in doc.get('audits') or []:
        row['predictions'] = str(Path(row['predictions']).resolve())
    path.write_text(json.dumps(doc, indent=1) + '\n')
    spec['inputs']['shape_audit'].append(
        {'path': str(path), 'axis': axis, 'artifact_identity_digest': digest})
Path(release, 'spec.json').write_text(json.dumps(spec, indent=1, sort_keys=True) + '\n')
print(json.dumps({'spec': str(Path(release, 'spec.json')), 'artifact_sha256': arm_sha,
                  'artifact_identity_digest': digest,
                  'edited_tensors': spec['arm']['edited_tensors'],
                  'min_cos': spec['arm']['realized_cos'],
                  'codec_event': codec.get('codec_event')}, indent=1))
PY
  mark spec "written $SPEC"
}

# ------------------------------------------------------------------------ preflight
preflight() {
  local missing=0
  check() {
    if [[ -e $2 ]]; then printf '  present  %-22s %s\n' "$1" "$2"
    else printf '  MISSING  %-22s %s\n' "$1" "$2"; missing=$((missing + 1)); fi
  }
  printf 'P7 inkling-2-mini preflight (%s)\n' "$ROOT"
  check base-gguf "$BASE_ART"
  check source-checkpoint "$SRC"
  check operands-pins "$OPERANDS"
  check suite-removal "$SUITE_REMOVAL"
  check suite-heldout "$SUITE_HELDOUT"
  check suite-retention "$SUITE_RETENTION"
  check wave-script "$ROOT/scripts/capability_wave.py"
  check shape-audit-script "$ROOT/scripts/refusal_shape_audit.py"
  check ppl-screen-script "$ROOT/scripts/refusal_ppl_screen.py"
  check ppl-control-corpus "$ROOT/verification/ppl-probe/control.txt"
  printf '  %-8s %-22s %s\n' "$([[ -s $RECORDS ]] && echo present || echo MISSING)" capture-records "$RECORDS"
  printf '  %-8s %-22s %s\n' "$([[ -s $AXES ]] && echo present || echo MISSING)" shared-axis "$AXES"
  printf '  %-8s %-22s %s\n' "$([[ -s $ARM_ART ]] && echo present || echo MISSING)" arm "$ARM_ART"
  printf '  serve binary: %s\n' "${SERVE_BIN:-NOT FOUND}"
  for step in retention-base harmful40-arm holdout-arm retention-arm; do
    printf '  output   %-22s %s\n' "$step" "$OUT/$step/{predictions.jsonl,wave.json}"
  done
  return 0
}

dry_run() {
  local serve_display=${SERVE_BIN:-'<NO llama-server found on this host: see the SERVE_NOT_FOUND note>'}
  printf '# dry run: every argv, in order. Nothing is executed, nothing is written.\n'
  cat <<EOF
# 0. operands ($OPERANDS): sha256 of every pin; restore
#    $SUITE_REMOVAL from $SUITE_REMOVAL_TWIN when the host cache lost it
# 1. capture (skipped when $RECORDS exists)
$PY -m jdewitt capture --model $SRC --suite $SUITE_CAPTURE --out $RECORDS \\
    --gguf $BASE_ART --kinds refusal,harmless --locus first-answer \\
    --max-tokens 96 --ctx-size 4096 --limit 64

# 2. axis (skipped when $AXES exists)
$PY -m jdewitt.spanfit axes --records $RECORDS --out $AXES --shape shared

# 3. arm (skipped when $ARM_ART + codec table + build report exist):
#    per-tensor type from measured cos, IQ4_NL first, then q6_k -> q8_0 -> f16
$PY - <merge: one rewrite_with_edits pass per chosen type, scale $SCALE, floor $CODEC_FLOOR>
    -> $ARM_ART, $BUILD_REPORT, $CODEC, $FIDELITY

# 4. serve the BASE artifact, one wave, stop it
$serve_display -m $BASE_ART -c $CTX --parallel $PARALLEL --jinja --no-webui --host 127.0.0.1 --port $PORT -ngl $NGL --temp 0
$PY $ROOT/scripts/capability_wave.py $SUITE_RETENTION $WAVE_BASE_RET \\
    --label retention-base --port $PORT --max-tokens $CAP --parallel $PARALLEL \\
    --kinds math,code --temperature 0 --preview 3 --serve-config '<the served argv above>'

# 5. serve the ARM artifact, three waves, stop it
$serve_display -m $ARM_ART -c $CTX --parallel $PARALLEL --jinja --no-webui --host 127.0.0.1 --port $PORT -ngl $NGL --temp 0
$PY $ROOT/scripts/capability_wave.py $SUITE_REMOVAL $WAVE_REMOVAL \\
    --label harmful40-arm --port $PORT --max-tokens $CAP --parallel $PARALLEL \\
    --kinds refusal --temperature 0 --preview 3 --serve-config '<the served argv above>'
$PY $ROOT/scripts/capability_wave.py $SUITE_HELDOUT $WAVE_HELDOUT \\
    --label holdout-arm --port $PORT --max-tokens $CAP --parallel $PARALLEL \\
    --kinds refusal --temperature 0 --preview 3 --serve-config '<the served argv above>'
$PY $ROOT/scripts/capability_wave.py $SUITE_RETENTION $WAVE_ARM_RET \\
    --label retention-arm --port $PORT --max-tokens $CAP --parallel $PARALLEL \\
    --kinds math,code --temperature 0 --preview 3 --serve-config '<the served argv above>'

# 6. shape audits over the saved arm waves (CPU, no server)
$PY $ROOT/scripts/refusal_shape_audit.py harmful40-arm=$WAVE_REMOVAL/predictions.jsonl --out $AUDIT_REMOVAL
$PY $ROOT/scripts/refusal_shape_audit.py holdout-arm=$WAVE_HELDOUT/predictions.jsonl --out $AUDIT_HELDOUT

# 7. drift: base vs arm on the control corpus (llama-perplexity, no server)
$PY $ROOT/scripts/refusal_ppl_screen.py --binary \${P7_PPL:-<resolved>} \\
    --corpus-dir $ROOT/verification/ppl-probe \\
    --artifact base=$BASE_ART --artifact arm=$ARM_ART --out $PPL_SCREEN

# 8. cert spec for the analysis card
$PY - <assemble from the measured files> -> $SPEC
EOF
  echo
  preflight
}

# ----------------------------------------------------------------------------- main
main() {
  mkdir -p "$OUT"
  resolve_bin SERVE \
    /mnt/ssd/dev/inkling-rt/llama.cpp/build-sycl/bin/llama-server \
    /mnt/ssd/dev/inkling-rt-t57174e1b/llama.cpp/build-sycl/bin/llama-server \
    /mnt/ssd/dev/inkling-rt-fadec/llama.cpp/build-sycl/bin/llama-server || true
  SERVE_BIN=$REPLY
  case ${1:---run} in
    --dry|--dry-run) dry_run; exit 0 ;;
    --check) step_operands; preflight; exit $? ;;
    --run|"") : ;;
    *) die 64 USAGE "unknown argument $1; use --dry, --check or nothing" ;;
  esac
  trap stop_server EXIT
  trap 'stop_server; exit 130' INT TERM
  : >> "$PROGRESS"
  mark start "pid=$$ serve=${SERVE_BIN:-none}"
  say "P7 inkling-2-mini: ctx $CTX parallel $PARALLEL cap $CAP scale $SCALE"
  step_operands
  step_capture
  step_axes
  step_arm
  step_waves
  step_audits
  step_drift
  step_spec
  stop_server
  mark finished "arm, waves, audits, drift, codec table and spec written"
  say "finished. Next (analysis card): uv run jdewitt release-certify --spec $SPEC --out $RELEASE/packet.json"
}

main "$@"
