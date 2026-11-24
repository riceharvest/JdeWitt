#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# The GPU half of the P5 merge+cert card: three waves, one server per wave, resumable.
#
# Run it *inside* the GPU lock, never before it:
#   lab-gpu b70 --need-gib 12 -- scripts/p5_cert_phase.sh
#
# Each wave names its own model and the /props gate asserts the served path is that
# model before a row is generated. That is not decoration: an earlier revision of this
# script held one -m "$ART" for every wave, so the run labelled "hold-base" would have
# been a second reading of the ARM (a base wave that never saw the base). A wave counts
# as a base wave only when the server says it is serving the base.
#
# Two serve argvs, deliberately, and both are recorded verbatim in each wave's
# serve_config:
#   content    --parallel 2  because the 24-family wave generates up to 4096 tokens
#                            per row and llama.cpp splits the 8192 ctx across slots
#                            (4096/slot); this mirrors the base holdout wave's argv.
#   retention  --parallel 4  because every existing retention arm (q3base, f15s15,
#                            ...) was taken at 8192/4 and the paired comparison is
#                            only meaningful against the same slot count.
#
# A wave whose outdir already holds a complete wave.json (rows == expected, one
# prediction line per row) is skipped, so a killed phase resumes without re-spending
# GPU time on what already landed.
set -euo pipefail

ART=${ART:-/mnt/ssd/models/q3-4b-pair-f15s15-l935.gguf}
BASE=${BASE:-/mnt/ssd/models/q3-4b-base-q8.gguf}
SRV=${SRV:-/mnt/ssd/b70-code/llama.cpp-flint/build-vk/bin/llama-server}
PORT=${PORT:-8141}
OUT=${OUT:-verification/p5}
LOG=${LOG:-verification/p5/server.log}
PY=${PY:-/mnt/ssd/dev workspace/JdeWitt/.venv/bin/python}

export LD_LIBRARY_PATH=/mnt/ssd/b70-code/llama.cpp-flint/build-vk/bin:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$(cd "$(dirname "$0")/.." && pwd)

common=(--host 127.0.0.1 --port "$PORT" --ctx-size 8192 -ngl 999 \
        --device Vulkan1 --split-mode none --jinja --no-webui --no-warmup -lv 4)

start_server() {  # $1 = model, $2 = parallel
  local model=$1 par=$2
  : > "$LOG"
  "$SRV" -m "$model" "${common[@]}" --parallel "$par" >>"$LOG" 2>&1 &
  echo $!
}

wait_props() {  # $1 = the model that must be the one being served
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/props" -o /tmp/p5-props.json; then
      "$PY" - "$1" <<'EOF'
import json,sys
from pathlib import Path
p=json.load(open('/tmp/p5-props.json'))
served=p.get('model_path') or p.get('path') or ''
print('props:', served, '| n_ctx', p.get('default_generation_settings',{}).get('n_ctx'))
assert Path(served).resolve()==Path(sys.argv[1]).resolve(), f'serving {served} not {sys.argv[1]}'
EOF
      return 0
    fi
    sleep 2
  done
  echo "server did not answer /props; last log:" >&2
  tail -30 "$LOG" >&2
  return 1
}

stop_server() {  # $1 = pid
  kill "$1" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$1" 2>/dev/null || return 0; sleep 1; done
  kill -9 "$1" 2>/dev/null || true
}

wave_done() {  # $1 = outdir, $2 = expected rows
  [ -f "$1/wave.json" ] && [ -f "$1/predictions.jsonl" ] || return 1
  "$PY" - "$1" "$2" <<'EOF'
import json,sys
from pathlib import Path
d=Path(sys.argv[1]); want=int(sys.argv[2])
w=json.load(open(d/'wave.json'))
rows=sum(1 for _ in open(d/'predictions.jsonl'))
sys.exit(0 if w.get('rows')==want and rows==want else 1)
EOF
}

serve_wave() {  # $1 model $2 label $3 suite $4 outdir $5 max_tokens $6 parallel $7 kinds
  local model=$1 label=$2 suite=$3 outdir=$4 mt=$5 par=$6 kinds=$7
  local argv="-m $model --host 127.0.0.1 --port $PORT --ctx-size 8192 --parallel $par -ngl 999 --device Vulkan1 --split-mode none --jinja --no-webui --no-warmup"
  local pid
  pid=$(start_server "$model" "$par")
  trap 'stop_server '"$pid"' || true' EXIT
  wait_props "$model"
  echo "=== wave $label on $model (pid $pid, max_tokens=$mt parallel=$par kinds=$kinds)"
  "$PY" scripts/capability_wave.py "$suite" "$outdir" --label "$label" --port "$PORT" \
      --max-tokens "$mt" --parallel "$par" --kinds "$kinds" --serve-config "$argv" \
      --preview 3
  stop_server "$pid"
  trap - EXIT
}

mkdir -p "$OUT"

# 1. The arm's 24-family content wave (4096 tok, parallel 2).
if wave_done "$OUT/arm-holdout-24" 24; then
  echo "=== skip pair-f15s15-hold24 (complete)"
else
  serve_wave "$ART" pair-f15s15-hold24 verification/screens/q3-holdout-24.jsonl \
      "$OUT/arm-holdout-24" 4096 2 refusal
fi

# 2. The same-argv base wave on the BASE artifact: the paired McNemar base is taken on
#    this server at this slot count, not borrowed from another run.
if wave_done "$OUT/base-holdout-24" 24; then
  echo "=== skip hold-base-hold24 (complete)"
else
  serve_wave "$BASE" hold-base-hold24 verification/screens/q3-holdout-24.jsonl \
      "$OUT/base-holdout-24" 4096 2 refusal
fi

# 3. The 241-row retention wave on the arm (math,code at 1024 tok, parallel 4).
if wave_done "$OUT/arm-retention" 241; then
  echo "=== skip pair-f15s15-retention (complete)"
else
  serve_wave "$ART" pair-f15s15-retention verification/capability/capability-run-en.jsonl \
      "$OUT/arm-retention" 1024 4 math,code
fi

echo "PHASE DONE"
ls -l "$OUT"/arm-holdout-24 "$OUT"/base-holdout-24 "$OUT"/arm-retention
