#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# The GPU half of the P5 merge+cert card: two waves against one artifact on the B70.
#
# Run it *inside* the GPU lock, never before it:
#   lab-gpu b70 --need-gib 12 -- scripts/p5_cert_phase.sh
#
# Two serve argvs, deliberately, and both are recorded verbatim in each wave's
# serve_config:
#   content    --parallel 2  because the 24-family wave generates up to 4096 tokens
#                            per row and llama.cpp splits the 8192 ctx across slots
#                            (4096/slot); this mirrors the base holdout wave's argv.
#   retention  --parallel 4  because every existing retention arm (q3base, f15s15,
#                            ...) was taken at 8192/4 and the paired comparison is
#                            only meaningful against the same slot count.
set -euo pipefail

ART=${ART:-/mnt/ssd/models/q3-4b-pair-f15s15-l935.gguf}
SRV=${SRV:-/mnt/ssd/b70-code/llama.cpp-flint/build-vk/bin/llama-server}
PORT=${PORT:-8141}
OUT=${OUT:-verification/p5}
LOG=${LOG:-/mnt/ssd/hermes/profiles/ab-run-2560/cache/scratch/p5-cert-server.log}
PY=${PY:-/mnt/ssd/dev workspace/JdeWitt/.venv/bin/python}

export LD_LIBRARY_PATH=/mnt/ssd/b70-code/llama.cpp-flint/build-vk/bin:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$(cd "$(dirname "$0")/.." && pwd)

common=(-m "$ART" --host 127.0.0.1 --port "$PORT" --ctx-size 8192 -ngl 999 \
        --device Vulkan1 --split-mode none --jinja --no-webui --no-warmup -lv 4)

start_server() {  # $1 = parallel
  local par=$1
  : > "$LOG"
  "$SRV" "${common[@]}" --parallel "$par" >>"$LOG" 2>&1 &
  echo $!
}

wait_props() {  # poll /props until the server answers, prove the artifact identity
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/props" -o /tmp/p5-props.json; then
      "$PY" - "$ART" <<'EOF'
import json,sys
from pathlib import Path
p=json.load(open('/tmp/p5-props.json'))
served=p.get('model_path') or p.get('path') or ''
print('props:', p.get('model_path'), '| n_ctx', p.get('default_generation_settings',{}).get('n_ctx'))
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

wave() {  # $1 label $2 suite $3 outdir $4 max_tokens $5 parallel $6 kinds
  local label=$1 suite=$2 outdir=$3 mt=$4 par=$5 kinds=$6
  local argv="-m $ART --host 127.0.0.1 --port $PORT --ctx-size 8192 --parallel $par -ngl 999 --device Vulkan1 --split-mode none --jinja --no-webui --no-warmup"
  echo "=== wave $label (max_tokens=$mt parallel=$par kinds=$kinds)"
  "$PY" scripts/capability_wave.py "$suite" "$outdir" --label "$label" --port "$PORT" \
      --max-tokens "$mt" --parallel "$par" --kinds "$kinds" --serve-config "$argv" \
      --preview 3
}

mkdir -p "$OUT"

pid=$(start_server 2)
trap 'stop_server "$pid"' EXIT
wait_props
echo "content wave on server pid $pid (parallel 2)"
wave pair-f15s15-hold24 verification/screens/q3-holdout-24.jsonl "$OUT/arm-holdout-24" 4096 2 refusal
# The same-argv base wave, so the paired McNemar has a base taken on this server,
# not one borrowed from a previous run at another slot count.
wave hold-base-hold24 verification/screens/q3-holdout-24.jsonl "$OUT/base-holdout-24" 4096 2 refusal
stop_server "$pid"
trap - EXIT

pid=$(start_server 4)
trap 'stop_server "$pid"' EXIT
wait_props
echo "retention wave on server pid $pid (parallel 4)"
wave pair-f15s15-retention verification/capability/capability-run-en.jsonl "$OUT/arm-retention" 1024 4 math,code
stop_server "$pid"
trap - EXIT
echo "PHASE DONE"
ls -l "$OUT"/arm-holdout-24 "$OUT"/arm-retention
