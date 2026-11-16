# SPDX-License-Identifier: AGPL-3.0-or-later
"""Append-only run lifecycle event log. Hot path is one small O_APPEND write."""
import json
import os
from pathlib import Path

from .clock import now_iso

MAX_BYTES = 8 * 1024 * 1024


def emit(path, kind, **fields):
    """Append one JSON line and rotate to .1 once past 8 MiB (best effort).

    Single small O_APPEND writes are atomic across processes on Linux, so several
    b70-ablator jobs may share one run log safely. A logging failure never kills
    the run; the durable artifacts stay the source of truth.
    """
    path = Path(path)
    try:
        if path.is_file() and path.stat().st_size > MAX_BYTES:
            path.rename(path.with_suffix('.jsonl.1'))
    except OSError:
        pass
    line = json.dumps({'at': now_iso(), 'kind': kind, **fields},
                      separators=(',', ':'), ensure_ascii=False, default=str) + '\n'
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(descriptor, line.encode())
        finally:
            os.close(descriptor)
    except OSError:
        pass


def read_events(path, limit=2000):
    """Tail the event log plus its rotated predecessor; malformed lines are skipped."""
    lines = []
    for candidate in (Path(str(path).removesuffix('.jsonl') + '.jsonl.1'), Path(path)):
        if candidate.is_file():
            lines.extend(candidate.read_text(errors='replace').splitlines())
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out
