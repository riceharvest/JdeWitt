# SPDX-License-Identifier: AGPL-3.0-or-later
"""Retention falsifier for the decode-side denial-word remap.

One authored capability suite, both arms, one cap: ``arm`` decodes the rows with
the same client-stepped loop the decode driver documents (``jdewitt vocab-remap``)
and grades every answer with the shipped ``jdewitt.evaluation.score_response``.
``report`` pairs the arms by row id and writes the falsifier: pass counts, the
discordant pairs and the exact McNemar p, median answer chars, invalid rows, and
a per-rule collateral table naming the table entry that fired on each failing
row. ``narrow`` derives a table override that scopes or drops a named entry so a
rule that costs capability rows can be re-run once.

    python scripts/vocab_remap_retention.py arm --suite SUITE.jsonl \
        --artifact ARTIFACT.gguf --serve PATH/llama-server \
        --out verification/vocab-remap/retention/off --remap off --max-tokens 512

    python scripts/vocab_remap_retention.py report \
        --off verification/vocab-remap/retention/off \
        --on verification/vocab-remap/retention/on \
        --out verification/vocab-remap/retention.json

The off arm is the control: it runs the identical loop with no ``logit_bias``, and
the decode card already showed that loop reproduces a plain greedy wave
token-for-token. A row is a pass only when ``score_response`` says ``status=ok``
and ``passed`` is true; a truncated or empty answer never counts as a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jdewitt.data import load_suite  # noqa: E402
from jdewitt.evaluation import score_response  # noqa: E402
from jdewitt.vocabremap import (  # noqa: E402
    OPENER_WINDOW_TOKENS,
    _merge_ranks,
    apply_chat_template,
    decode_steps,
    detokenize,
    entry_sequences,
    load_table,
    read_tokenizer,
    served_tokenize,
    start_server,
    stop_server,
    verify_artifact_identity,
    verify_served_tokenizer,
)

ARM_SCHEMA = 'jdewitt.vocab-remap-retention-arm/1'
REPORT_SCHEMA = 'jdewitt.vocab-remap-retention/1'
ALPHA = 0.05
# The graded answer excludes these markers; they are still in the raw token path.
SPECIAL_MARKERS = ('<|im_end|>', '<|im_start|>', '<|endoftext|>', '<|eot_id|>', '</s>')


class _DeterministicJudge:
    """Capability rows never consult a judge; score_response only reads scoring_mode here."""

    scoring_mode = 'refusal-check'

    def check(self, answer, language):  # pragma: no cover - capability suites never call this
        raise RuntimeError('retention suite carries capability rows only; no refusal check is run')

    def classify(self, prompt, answer, language):  # pragma: no cover - never called
        raise RuntimeError('retention suite carries capability rows only; no judge is called')


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _portable(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return Path(path).name


def _json_line(record: dict) -> str:
    return json.dumps(record, sort_keys=True, allow_nan=False)


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f'{path}:{number}: invalid JSON: {error.msg}') from error
    return rows


def _append_row(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(_json_line(record) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def _rewrite_valid_rows(path: Path) -> list[dict]:
    """Drop a half-written trailing line left by a killed process."""
    if not path.exists():
        return []
    kept, changed = [], False
    for line in path.read_text().splitlines():
        if not line.strip():
            changed = True
            continue
        try:
            kept.append(json.loads(line))
        except json.JSONDecodeError:
            changed = True
    if changed:
        path.write_text(''.join(_json_line(row) + '\n' for row in kept))
    return kept


def _passes(row: dict) -> bool:
    return row.get('status') == 'ok' and row.get('passed') is True


def _suppression_key(event: dict) -> tuple:
    return (event.get('entry_id'), event.get('step'), event.get('suppressed_text'))


def _meta(artifact: Path, suite: Path, table: Path, remap: str, max_tokens: int,
          served: dict, tokenizer: dict) -> dict:
    document, raw_table = load_table(table)
    return {
        'schema': ARM_SCHEMA,
        'remap': remap,
        'scope': ('One arm of the retention falsifier: the authored 241-row capability suite decoded with the '
                  'documented client-stepped loop, graded by jdewitt.evaluation.score_response. A row is a '
                  'pass only when status=ok and passed=true.'),
        'artifact': {'path': _portable(artifact), 'sha256': _sha256_file(artifact),
                     'bytes': artifact.stat().st_size, 'tokenizer': tokenizer['identity']},
        'suite': {'path': _portable(suite), 'sha256': _sha256_file(suite), 'rows': len(load_suite(suite))},
        'table': {'path': _portable(table), 'sha256': hashlib.sha256(raw_table).hexdigest(),
                  'entries': len(document['entries']),
                  'derivation': document.get('derivation')},
        'server': served,
        'config': {'max_tokens': max_tokens, 'temperature': 0.0, 'cache_prompt': True,
                   'n_probs': 1, 'opener_window_tokens': OPENER_WINDOW_TOKENS,
                   'template': 'chat', 'grader': 'jdewitt.evaluation.score_response'},
    }


def run_arm(args) -> int:
    artifact, suite_dir, out = Path(args.artifact), Path(args.suite), Path(args.out)
    if not artifact.is_file():
        raise SystemExit(f'artifact does not exist: {artifact}')
    if not suite_dir.is_file():
        raise SystemExit(f'suite does not exist: {suite_dir}')
    if args.remap not in ('on', 'off'):
        raise SystemExit('--remap must be on or off')
    if args.max_tokens < 1:
        raise SystemExit('--max-tokens must be positive')
    if args.serve is None and args.port is None:
        raise SystemExit('arm needs --serve PATH (start a CPU llama-server) or --port N (one already running)')
    suite = load_suite(suite_dir)
    entries_document, _raw = load_table(args.table)
    entries = entries_document['entries']
    tokenizer = read_tokenizer(artifact)
    tokens = tokenizer['tokens']
    vocab = {token: token_id for token_id, token in enumerate(tokens)}
    ranks = _merge_ranks(tokenizer['merges'])

    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / 'rows.jsonl'
    done = []
    if rows_path.exists():
        if not args.resume:
            raise SystemExit(f'{rows_path} exists; pass --resume to continue that arm or pick a fresh --out')
        done = _rewrite_valid_rows(rows_path)
        if any(row.get('remap') != args.remap for row in done):
            raise SystemExit(f'{rows_path} holds rows from another arm; refusing to mix them')
    done_ids = {row['id'] for row in done}
    if args.limit:
        suite = suite[:args.limit]
        done_ids &= {row['id'] for row in suite}
    pending = [row for row in suite if row['id'] not in done_ids]
    meta_path = out / 'meta.json'
    table_path = args.table or Path('jdewitt/assets/vocab-remap-denial.json')
    started = None
    began = time.monotonic()
    try:
        port = args.port
        if args.serve is not None:
            from jdewitt.vocabremap import _free_port
            port = port or _free_port()
            started = start_server(args.serve, artifact, port, out / 'server.log')
        if not isinstance(port, int) or not 0 < port < 65536:
            raise SystemExit(f'--port must be an integer in 1..65535, got {port!r}')
        props = verify_artifact_identity(port, artifact)
        served_check = verify_served_tokenizer(entries, port, tokens, vocab, ranks)
        if not served_check['agrees']:
            raise SystemExit(f'the server on port {port} tokenizes table forms differently from the artifact '
                             f'header ({len(served_check["mismatches"])} of '
                             f'{served_check["contexts_checked"]} contexts); refusing to record rows')
        served = {'argv': (Path(args.serve).name if args.serve else ''),
                  'started_by_runner': started is not None, 'port': port,
                  **props, 'tokenizer_check': {'contexts_checked': served_check['contexts_checked'],
                                               'agrees': served_check['agrees']}}
        meta = _meta(artifact, suite_dir, table_path, args.remap, args.max_tokens, served, tokenizer)
        if args.limit:
            meta['config']['limit'] = args.limit
        if meta_path.exists():
            previous = json.loads(meta_path.read_text())
            if previous.get('artifact', {}).get('sha256') != meta['artifact']['sha256']:
                raise SystemExit(f'{meta_path} was produced on another artifact; use a fresh --out')
            if previous.get('suite', {}).get('sha256') != meta['suite']['sha256']:
                raise SystemExit(f'{meta_path} was produced on another suite; use a fresh --out')
            if previous.get('table', {}).get('sha256') != meta['table']['sha256']:
                raise SystemExit(f'{meta_path} was produced by another table; use a fresh --out')
            if previous.get('config') != meta.get('config'):
                raise SystemExit(f'{meta_path} was produced with another cap/limit; use a fresh --out')
        else:
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
        entry_items = entry_sequences(entries, port, tokens)
        judge = _DeterministicJudge()
        print(f'[{args.remap}] {len(pending)} of {len(suite)} rows to decode '
              f'(cap {args.max_tokens}, build {props.get("build_info")})', flush=True)
        for number, task in enumerate(pending, 1):
            row_began = time.monotonic()
            prompt = apply_chat_template(port, task['prompt'])
            prompt_ids = served_tokenize(port, prompt, parse_special=True)
            loop = decode_steps(port, tokens, entry_items, prompt, prompt_ids, args.max_tokens,
                                args.remap == 'on')
            text = detokenize(loop['ids'], tokens)
            finish_reason = 'stop' if loop['stop'] is not None else 'length'
            scored = score_response(task, {'content': text, 'finish_reason': finish_reason,
                                           'completion_tokens': len(loop['ids'])}, judge)
            record = {
                'id': task['id'], 'kind': task['kind'], 'language': task['language'],
                'family_id': task.get('family_id'), 'category': task.get('category'),
                'remap': args.remap, 'max_tokens': args.max_tokens,
                'prompt_sha256': hashlib.sha256(task['prompt'].encode()).hexdigest(),
                'prompt_tokens': len(prompt_ids), 'stop': loop['stop'],
                'finish_reason': finish_reason, 'n_tokens': len(loop['ids']),
                'text': text,
                'answer': scored.get('answer'), 'status': scored.get('status'),
                'passed': scored.get('passed'), 'degenerate': scored.get('degenerate'),
                'grade': (scored.get('grade') or {}).get('reason') if task['kind'] == 'code' else None,
                'suppressions': [{'entry_id': event['entry_id'], 'step': event['step'],
                                  'suppressed_id': event['suppressed_id'],
                                  'suppressed_text': event['suppressed_text'],
                                  'prev_token': event.get('prev_token')} for event in loop['suppressions']],
                'completion_requests': loop['completion_requests'],
                'resample_limit_steps': len(loop['resample_limit_steps']),
                'retokenized_steps': len(loop['retokenized_steps']),
                'seconds': round(time.monotonic() - row_began, 3),
            }
            _append_row(rows_path, record)
            done.append(record)
            if number % 10 == 0 or number == len(pending):
                elapsed = time.monotonic() - began
                tokens_so_far = sum(item['n_tokens'] for item in done)
                print(f'[{args.remap}] {number}/{len(pending)} rows  {tokens_so_far} tokens  '
                      f'{tokens_so_far / max(elapsed, 1e-9):.1f} tok/s  '
                      f'last={task["id"]} ({record["n_tokens"]} tok, {record["status"]})', flush=True)
    finally:
        if started is not None:
            stop_server(started)

    rows = done
    generated = sum(row['n_tokens'] for row in rows)
    decode_seconds = round(sum(row['seconds'] for row in rows), 3)
    by_kind = {}
    for kind in sorted({row['kind'] for row in rows}):
        subset = [row for row in rows if row['kind'] == kind]
        by_kind[kind] = {'rows': len(subset), 'passed': sum(1 for row in subset if _passes(row)),
                         'invalid': sum(1 for row in subset if row['status'] != 'ok')}
    report = {
        'schema': ARM_SCHEMA,
        'remap': args.remap,
        'scope': meta['scope'],
        'artifact': meta['artifact'], 'suite': meta['suite'], 'table': meta['table'],
        'server': meta['server'], 'config': meta['config'],
        'rows_file': {'path': 'rows.jsonl', 'sha256': _sha256_file(rows_path), 'rows': len(rows)},
        'metrics': {
            'generated_tokens': generated,
            'decode_seconds': decode_seconds,
            'wall_seconds': round(time.monotonic() - began, 3),
            'tokens_per_second': round(generated / decode_seconds, 2) if decode_seconds else None,
            'completion_requests': sum(row['completion_requests'] for row in rows),
            'suppression_events': sum(len(row['suppressions']) for row in rows),
            'rows_with_suppressions': sum(1 for row in rows if row['suppressions']),
            'resample_limit_rows': sum(1 for row in rows if row['resample_limit_steps']),
        },
        'scores': {'rows': len(rows), 'passed': sum(1 for row in rows if _passes(row)),
                   'invalid': sum(1 for row in rows if row['status'] != 'ok'),
                   'by_kind': by_kind},
        'resumed_rows': len(done_ids),
        'verdict': {'complete': len(rows) == len(suite),
                    'note': 'an arm is complete only when it holds exactly the suite rows it names'},
    }
    (out / 'run.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'arm': args.remap, 'rows': len(rows), 'passed': report['scores']['passed'],
                      'invalid': report['scores']['invalid'], 'seconds': report['metrics']['wall_seconds'],
                      'complete': report['verdict']['complete']}, indent=1))
    return 0 if report['verdict']['complete'] else 1


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p over the discordant pairs b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _median(values: list[int]) -> float | None:
    return round(statistics.median(values), 1) if values else None


def _binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def loss_rate_upper(k: int, n: int, alpha: float = ALPHA) -> float | None:
    """One-sided Clopper-Pearson upper bound on the per-row loss probability.

    With k=0 losses out of n this is the exact ``1 - alpha**(1/n)`` bound; the
    paired test says nothing about rows it never flipped, so the bound is the
    honest ceiling the suite can put on a rule that costs a row.
    """
    if n == 0:
        return None
    if k >= n:
        return 1.0
    low, high = k / n, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        if _binomial_cdf(k, n, middle) > alpha:
            low = middle
        else:
            high = middle
    return round(high, 6)


def _arm_stats(off: dict[str, dict], on: dict[str, dict]) -> dict:
    ids = sorted(off)
    both = [rid for rid in ids if _passes(off[rid]) and _passes(on[rid])]
    lost = [rid for rid in ids if _passes(off[rid]) and not _passes(on[rid])]
    gained = [rid for rid in ids if not _passes(off[rid]) and _passes(on[rid])]
    neither = [rid for rid in ids if not _passes(off[rid]) and not _passes(on[rid])]
    off_chars = [len(off[rid].get('answer') or '') for rid in ids]
    on_chars = [len(on[rid].get('answer') or '') for rid in ids]
    return {
        'rows': len(ids),
        'off_passed': sum(1 for rid in ids if _passes(off[rid])),
        'on_passed': sum(1 for rid in ids if _passes(on[rid])),
        'off_invalid': sum(1 for rid in ids if off[rid]['status'] != 'ok'),
        'on_invalid': sum(1 for rid in ids if on[rid]['status'] != 'ok'),
        'median_answer_chars': {'off': _median(off_chars), 'on': _median(on_chars)},
        'paired': {'both_pass': len(both), 'off_pass_on_fail': len(lost), 'off_fail_on_pass': len(gained),
                   'both_fail': len(neither), 'discordant': len(lost) + len(gained),
                   'mcnemar_exact_p': mcnemar_exact(len(lost), len(gained)),
                   'alpha': ALPHA,
                   'row_loss_rate_upper95': loss_rate_upper(len(lost), len(ids)),
                   'row_loss_rate_upper95_note': ('one-sided 95% Clopper-Pearson bound on the probability '
                                                  'that a capability row which passes without the remap '
                                                  'fails with it; rows the suite never flipped are not '
                                                  'evidence of absence beyond this bound'),
                   'method': 'exact two-sided binomial test over the discordant pairs (McNemar)'},
        'off_pass_on_fail_rows': lost,
        'off_fail_on_pass_rows': gained,
    }


def _rule_table(off: dict[str, dict], on: dict[str, dict], entry_ids: list[str] | None = None) -> dict:
    """Per table entry: where it fired, on which failing rows, and what it cost.

    Every entry in the table gets a row, including the ones that never fired: a
    zero is part of the falsifier, it says that entry had no opportunity on this
    suite rather than that it was measured to be harmless.
    """
    rules: dict[str, dict] = {}
    for entry_id in entry_ids or []:
        rules[entry_id] = {
            'entry_id': entry_id, 'events': 0, 'rows_fired': [],
            'rows_fired_off_passed_upstream': [], 'collateral_rows': [], 'recovered_rows': [],
            'failing_row_firings': [], 'fired_after_opener_window': 0,
        }
    for rid in sorted(on):
        row = on[rid]
        for event in row.get('suppressions') or []:
            entry = rules.setdefault(event['entry_id'], {
                'entry_id': event['entry_id'], 'events': 0, 'rows_fired': [],
                'rows_fired_off_passed_upstream': [], 'collateral_rows': [], 'recovered_rows': [],
                'failing_row_firings': [], 'fired_after_opener_window': 0,
            })
            entry['events'] += 1
            if rid not in entry['rows_fired']:
                entry['rows_fired'].append(rid)
            if event['step'] >= OPENER_WINDOW_TOKENS:
                entry['fired_after_opener_window'] += 1
            if not _passes(row):
                entry['failing_row_firings'].append({'row': rid, 'kind': row['kind'],
                                                     'step': event['step'],
                                                     'suppressed_text': event['suppressed_text'],
                                                     'off_passed': _passes(off[rid]),
                                                     'status': row['status']})
            if _passes(off[rid]) and not _passes(row) and rid not in entry['collateral_rows']:
                entry['collateral_rows'].append(rid)
            if not _passes(off[rid]) and _passes(row) and rid not in entry['recovered_rows']:
                entry['recovered_rows'].append(rid)
            if _passes(off[rid]) and rid not in entry['rows_fired_off_passed_upstream']:
                entry['rows_fired_off_passed_upstream'].append(rid)
    for value in rules.values():
        value['collateral_count'] = len(value['collateral_rows'])
        value['recovered_count'] = len(value['recovered_rows'])
        value['rows_fired_count'] = len(value['rows_fired'])
    return dict(sorted(rules.items()))


def _failing_attribution(on: dict[str, dict]) -> list[dict]:
    attributed = []
    for rid in sorted(on):
        row = on[rid]
        if _passes(row):
            continue
        attributed.append({
            'row': rid, 'kind': row['kind'], 'status': row['status'],
            'n_tokens': row['n_tokens'], 'finish_reason': row['finish_reason'],
            'fired': [{'entry_id': event['entry_id'], 'step': event['step'],
                       'suppressed_text': event['suppressed_text']}
                      for event in row.get('suppressions') or []],
            'note': ('the row did not pass; this is the attributed rule firing, not proof that the rule '
                     'caused the failure on its own' if row.get('suppressions') else
                     'the row did not pass and no table entry fired on it'),
        })
    return attributed


def _verdict(stats: dict, rules: dict) -> dict:
    p = stats['paired']['mcnemar_exact_p']
    losses = stats['paired']['off_pass_on_fail']
    gains = stats['paired']['off_fail_on_pass']
    costly = sorted(name for name, rule in rules.items() if rule['collateral_count'])
    if p < ALPHA and losses > gains:
        outcome = 'capability_cost'
        statement = (f'the remap arm lost {losses} capability rows against {gains} gained; the paired exact '
                     f'McNemar p={p:.6g} rejects "no difference" at alpha={ALPHA}. '
                     f'Rule(s) attributed on lost rows: {", ".join(costly) or "none"}')
    elif p < ALPHA and gains > losses:
        outcome = 'capability_gain'
        statement = (f'the remap arm gained {gains} capability rows against {losses} lost; the paired exact '
                     f'McNemar p={p:.6g} rejects "no difference" at alpha={ALPHA}')
    elif losses or gains:
        outcome = 'no_significant_difference'
        statement = (f'{losses} row(s) lost and {gains} gained; the paired exact McNemar p={p:.6g} does not '
                     f'reject "no difference" at alpha={ALPHA}. With {losses + gains} discordant pairs the '
                     'test has little power, so this is not evidence of no loss; it is a bound set by the '
                     'number of pairs the suite produced')
    else:
        outcome = 'no_discordant_pairs'
        statement = (f'both arms pass and fail the same {stats["rows"]} rows (0 discordant pairs, McNemar '
                     f'p={p:.6g}); this suite and decode found no capability difference. That is a measured '
                     'absence on 241 rows, not a general no-loss claim')
    return {'outcome': outcome, 'statement': statement, 'costly_rules': costly}


def report(args) -> int:
    off_rows = {row['id']: row for row in _load_rows(Path(args.off) / 'rows.jsonl')}
    arms = {'asset': Path(args.on)}
    for item in args.arm or []:
        label, _, directory = item.partition('=')
        if not label or not directory:
            raise SystemExit('--arm wants LABEL=DIR')
        arms[label] = Path(directory)
    if not off_rows:
        raise SystemExit(f'{args.off}/rows.jsonl is empty')
    off_meta = json.loads((Path(args.off) / 'meta.json').read_text())
    comparisons = {}
    for label, directory in arms.items():
        on_rows = {row['id']: row for row in _load_rows(directory / 'rows.jsonl')}
        missing = sorted(set(off_rows) - set(on_rows))
        extra = sorted(set(on_rows) - set(off_rows))
        if missing or extra:
            raise SystemExit(f'arm {label} does not pair with the off arm: missing {missing[:5]}, extra {extra[:5]}')
        on_meta = json.loads((directory / 'meta.json').read_text())
        for key, value in (('artifact', off_meta['artifact']['sha256']), ('suite', off_meta['suite']['sha256'])):
            got = on_meta[key]['sha256']
            if got != value:
                raise SystemExit(f'arm {label} was not decoded on the same {key} as the off arm: {got} != {value}')
        if on_meta['config']['max_tokens'] != off_meta['config']['max_tokens']:
            raise SystemExit(f'arm {label} used cap {on_meta["config"]["max_tokens"]}, the off arm '
                             f'{off_meta["config"]["max_tokens"]}; the arms must share the cap')
        stats = _arm_stats(off_rows, on_rows)
        table_path = Path(on_meta['table']['path'])
        table_entries, _raw = load_table(table_path if table_path.is_absolute() else REPO / table_path)
        entry_ids = [entry['id'] for entry in table_entries['entries']]
        rules = _rule_table(off_rows, on_rows, entry_ids)
        fired = [name for name, rule in rules.items() if rule['events']]
        verdict = _verdict(stats, rules)
        verdict['exposure'] = {
            'rows_with_firing': sum(1 for row in on_rows.values() if row.get('suppressions')),
            'rows': len(on_rows),
            'entries_never_fired': [name for name in entry_ids if name not in fired],
            'note': ('the verdict is bounded by exposure: a suite where the table rarely fires cannot '
                     'measure the cost of rules it never fires'),
        }
        comparisons[label] = {
            'directory': _portable(directory),
            'table': on_meta['table'], 'config': on_meta['config'], 'scores': on_meta.get('scores'),
            'stats': stats, 'per_rule': rules,
            'exposure': {
                'table_entries': len(entry_ids),
                'entries_fired': len(fired),
                'entries_never_fired': [name for name in entry_ids if name not in fired],
                'rows_with_any_firing': sum(1 for row in on_rows.values() if row.get('suppressions')),
                'rows': len(on_rows),
                'note': ('Entries that never fired had no opportunity on this suite; their zeros are '
                         'non-exposure, not evidence that the rule is harmless.'),
            },
            'failing_rows': _failing_attribution(on_rows),
            'verdict': verdict,
        }
    final_label = args.final or ('narrowed' if 'narrowed' in comparisons else 'asset')
    final = comparisons.get(final_label)
    if final is None:
        raise SystemExit(f'--final {final_label!r} is not one of {sorted(comparisons)}')
    off_run_path = Path(args.off) / 'run.json'
    off_run = json.loads(off_run_path.read_text()) if off_run_path.exists() else None
    document = {
        'schema': REPORT_SCHEMA,
        'scope': ('Retention falsifier: the authored 241-row capability suite decoded through the documented '
                  'client-stepped remap loop, both arms at one cap, graded by the shipped score_response. '
                  'The off arm carries no logit_bias. Paired by row id; the arms are never averaged.'),
        'suite': off_meta['suite'], 'artifact': off_meta['artifact'], 'cap': off_meta['config']['max_tokens'],
        'off': {'directory': _portable(Path(args.off)), 'table': off_meta['table'],
                'scores': off_run['scores'] if off_run else None,
                'run': off_run},
        'arms': comparisons,
        'final_arm': final_label,
        'verdict': final['verdict'],
        'limits': ('A deterministic greedy pass on one quantized artifact at one cap, with the client-stepped '
                   'loop as the only decoder under test. The suite is the authored 241-row wave; a pass is the '
                   'shipped grader\'s word, not a second opinion. Per-rule attribution lists the entry that '
                   'fired on a failing row; the bias is the only difference between the arms, but a divergence '
                   'can compound after the triggering step, so the table is evidence of attribution, not a '
                   'per-token causal proof.'),
    }
    Path(args.out).write_text(json.dumps(document, indent=2, sort_keys=True) + '\n')
    _print_report(document)
    return 0


def _print_report(document: dict) -> None:
    print(f'RETENTION FALSIFIER  suite {document["suite"]["path"]}  rows {document["suite"]["rows"]}  '
          f'cap {document["cap"]}')
    print(f'artifact {document["artifact"]["path"]}  sha256 {document["artifact"]["sha256"][:12]}...')
    for label, arm in document['arms'].items():
        stats = arm['stats']
        print()
        print(f'arm {label}  table {arm["table"]["path"]} sha256 {arm["table"]["sha256"][:12]}...')
        print(f'  passed {stats["on_passed"]}/{stats["rows"]} (off {stats["off_passed"]}/{stats["rows"]})  '
              f'invalid {stats["on_invalid"]} (off {stats["off_invalid"]})  '
              f'median chars {stats["median_answer_chars"]["on"]} (off {stats["median_answer_chars"]["off"]})')
        paired = stats['paired']
        print(f'  paired both={paired["both_pass"]} lost={paired["off_pass_on_fail"]} '
              f'gained={paired["off_fail_on_pass"]} both_fail={paired["both_fail"]}  '
              f'exact McNemar p={paired["mcnemar_exact_p"]:.6g}  '
              f'loss-rate 95% upper {paired["row_loss_rate_upper95"]:.4g}')
        exposure = arm.get('exposure') or {}
        if exposure:
            print(f'  exposure {exposure["rows_with_any_firing"]}/{exposure["rows"]} rows, '
                  f'{exposure["entries_fired"]}/{exposure["table_entries"]} table entries fired')
        for name, rule in arm['per_rule'].items():
            if rule['events']:
                print(f'  rule {name}: {rule["events"]} events on {rule["rows_fired_count"]} rows, '
                      f'{len(rule["failing_row_firings"])} firing(s) on failing rows, '
                      f'collateral {rule["collateral_count"]}, recovered {rule["recovered_count"]}')
        print(f'  verdict {arm["verdict"]["outcome"]}: {arm["verdict"]["statement"]}')


def narrow(args) -> int:
    """Derive a table override: scope named entries to the opener window or drop them."""
    document, raw = load_table(args.table)
    entries = document['entries']
    wanted = set(args.scope_opener_window or []) | set(args.drop or [])
    known = {entry['id'] for entry in entries}
    unknown = sorted(wanted - known)
    if unknown:
        raise SystemExit(f'--scope-opener-window/--drop name entries not in the table: {unknown}')
    changed = []
    kept = []
    for entry in entries:
        if entry['id'] in (args.drop or []):
            changed.append({'entry_id': entry['id'], 'change': 'dropped'})
            continue
        if entry['id'] in (args.scope_opener_window or []):
            if entry['scope'] == 'opener_window':
                raise SystemExit(f'{entry["id"]} is already scoped to the opener window')
            entry = dict(entry, scope='opener_window')
            changed.append({'entry_id': entry['id'], 'change': 'scope=any -> scope=opener_window'})
        kept.append(entry)
    if not changed:
        raise SystemExit('narrow changed nothing; name at least one entry')
    derived = dict(document)
    derived['entries'] = kept
    derived['derivation'] = {
        'derived_from_sha256': hashlib.sha256(raw).hexdigest(),
        'changes': changed,
        'reason': args.reason,
        'note': ('Iteration 2 table for the retention falsifier: only the named entry/scope changes; every '
                 'other entry is byte-identical to the packaged table.'),
    }
    Path(args.out).write_text(json.dumps(derived, indent=2) + '\n')
    print(json.dumps(derived['derivation'], indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)

    arm = commands.add_parser('arm', help='decode and grade one arm of the suite')
    arm.add_argument('--suite', type=Path, default=REPO / 'verification/capability/capability-run-en.jsonl')
    arm.add_argument('--artifact', type=Path, required=True)
    arm.add_argument('--serve', type=Path, help='llama-server binary to start (CPU, -ngl 0)')
    arm.add_argument('--port', type=int, help='port of an already-running server instead of --serve')
    arm.add_argument('--out', type=Path, required=True)
    arm.add_argument('--remap', choices=('on', 'off'), required=True)
    arm.add_argument('--table', type=Path, help='table override; default is the packaged asset')
    arm.add_argument('--max-tokens', type=int, default=512)
    arm.add_argument('--limit', type=int, default=0, help='decode only the first N rows (timing probe)')
    arm.add_argument('--resume', action='store_true', help='continue an interrupted arm in --out')
    arm.set_defaults(func=run_arm)

    rep = commands.add_parser('report', help='pair the arms and write retention.json')
    rep.add_argument('--off', type=Path, required=True)
    rep.add_argument('--on', type=Path, required=True)
    rep.add_argument('--arm', action='append', metavar='LABEL=DIR',
                     help='additional remap arm to compare against the same off arm (repeatable)')
    rep.add_argument('--final', help='arm label the verdict is taken from (default: narrowed, else asset)')
    rep.add_argument('--out', type=Path, default=REPO / 'verification/vocab-remap/retention.json')
    rep.set_defaults(func=report)

    nar = commands.add_parser('narrow', help='derive a table override scoping or dropping entries')
    nar.add_argument('--table', type=Path, help='source table; default is the packaged asset')
    nar.add_argument('--out', type=Path, required=True)
    nar.add_argument('--scope-opener-window', action='append', metavar='ENTRY_ID', default=[])
    nar.add_argument('--drop', action='append', metavar='ENTRY_ID', default=[])
    nar.add_argument('--reason', required=True)
    nar.set_defaults(func=narrow)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
