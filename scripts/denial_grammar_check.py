# SPDX-License-Identifier: AGPL-3.0-or-later
"""Static run of the denial-free grammar experiment.

Thin wrapper around ``jdewitt.denial_grammar_run.compile_run``; the same
implementation is reached through ``jdewitt grammar --compile``.  No model, no
server, no GPU: table, GGUF tokenizer header, text and automata only.

Run: ``python scripts/denial_grammar_check.py --out verification/denial-grammar``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jdewitt.denial_grammar import MIN_ANSWER_CHARS  # noqa: E402
from jdewitt.denial_grammar_run import compile_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='verification/denial-grammar')
    parser.add_argument('--predictions', default='verification/**/predictions.jsonl')
    parser.add_argument('--artifact', default=None,
                        help='GGUF whose tokenizer is read for the token view (header only)')
    parser.add_argument('--table', default=None)
    parser.add_argument('--min-chars', type=int, default=MIN_ANSWER_CHARS)
    parser.add_argument('--skip-probe', action='store_true')
    parser.add_argument('--mine', action='store_true',
                        help='grow the table from the saved answers first and emit it')
    args = parser.parse_args()
    _report, summary = compile_run(args.out, args.predictions, args.artifact,
                                   skip_probe=args.skip_probe, min_chars=args.min_chars,
                                   table=args.table, mine=args.mine)
    print(json.dumps(summary, indent=2, default=str, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
