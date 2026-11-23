#!/usr/bin/env bash
# Detach the B70 cert phase from this agent run: queue for the lock for up to four
# hours, then run the two waves. A fresh agent reads $LOG when this pid exits.
#
#   setsid nohup scripts/p5_cert_launch.sh > /path/phase.log 2>&1 < /dev/null &
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE" || exit 1
# The waiter key must be a pid that outlives the launcher shell: lab-gpu queues on
# $LAB_GPU_WAITER_PID, and `exec` keeps this pid alive as lab-gpu itself.
export LAB_GPU_WAITER_PID=$$
export LAB_GPU_WAIT_MIN=${LAB_GPU_WAIT_MIN:-240}
exec lab-gpu b70 --need-gib 12 -- bash "$HERE/scripts/p5_cert_phase.sh"
