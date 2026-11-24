# SPDX-License-Identifier: AGPL-3.0-or-later
"""Licence and access gate for the Space, before any download (SPACE-ROADMAP §7.2).

The gate reads Hub metadata only: ``HfApi().model_info(repo, revision=...)`` with
``files_metadata=False``. It never calls a download function.

Two refusals, in this order:

- ``GATED_REPO`` — the Hub says the repository is gated. A Space serves anonymous
  visitors, and an anonymous visitor cannot authenticate, so the file is refused
  even when its licence is acceptable.
- ``LICENCE_UNKNOWN`` — the model card tags/cardData carry no licence, or the
  licence they carry is not in the bank's allow-list. The refusal names what was
  found.

The allow-list is a bank artifact (``spaces/bank/licences.json``), not a table in
this module: coverage and licence move together. The gate returns the licence
string exactly as the card spells it. It is copied into the output card verbatim
and is never replaced by a friendlier label such as "permissive".

Hub facts are cached per ``(repo, revision)`` for the life of the process.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ALLOW_LIST_SCHEMA = 'jdewitt-space-bank-licences-v1'
ALLOW_LIST_NAME = 'licences.json'
DEFAULT_BANK_DIR = Path(__file__).resolve().parents[1] / 'spaces' / 'bank'

DECISION_ALLOWED = 'allowed'
DECISION_REFUSED = 'refused'
GATED_REPO = 'GATED_REPO'
LICENCE_UNKNOWN = 'LICENCE_UNKNOWN'
LICENCE_ACCEPTED = 'LICENCE_ACCEPTED'

_FACTS_CACHE = {}


def clear_cache():
    """Drop cached Hub facts; tests use this, a running Space never needs to."""
    _FACTS_CACHE.clear()


def _is_gated(value):
    """Hub 2.2 reports ``gated`` as a string ("manual"/"auto") and older ones as bool."""
    if not value:
        return False
    return str(value).strip().casefold() not in ('false', 'none', 'no', '0')


def _card_licences(card_data):
    if card_data is None:
        return []
    raw = card_data.get('license') if isinstance(card_data, dict) else getattr(card_data, 'license', None)
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    return [value.strip() for value in values if isinstance(value, str) and value.strip()]


def _tag_licences(tags):
    found = []
    for tag in tags or ():
        text = str(tag)
        prefix, separator, value = text.partition(':')
        if separator and prefix.strip().casefold() == 'license' and value.strip() and value.strip() not in found:
            found.append(value.strip())
    return found


def _choose_licence(card_values, tag_values):
    """Return (verbatim licence or None, source, what-was-found phrase)."""
    if len(set(card_values)) == 1:
        return card_values[0], 'cardData.license', f"licence {card_values[0]!r}"
    if len(set(card_values)) > 1:
        return None, 'cardData.license', 'cardData.license lists ' + ', '.join(repr(v) for v in card_values)
    if len(tag_values) == 1:
        return tag_values[0], 'tags', f"licence tag {tag_values[0]!r}"
    if len(tag_values) > 1:
        return None, 'tags', 'licence tags ' + ', '.join(repr(v) for v in tag_values)
    return None, 'missing', 'no licence in cardData or tags'


def _hub_facts(repo, revision, api=None):
    key = (repo, revision)
    cached = _FACTS_CACHE.get(key)
    if cached is not None:
        return cached
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    info = api.model_info(repo, revision=revision, files_metadata=False)
    card_data = getattr(info, 'card_data', None)
    if card_data is None:
        card_data = getattr(info, 'cardData', None)
    licence, source, found = _choose_licence(_card_licences(card_data), _tag_licences(getattr(info, 'tags', None)))
    facts = {
        'gated': _is_gated(getattr(info, 'gated', False)),
        'licence': licence,
        'licence_source': source,
        'found': found,
        'sha': getattr(info, 'sha', None),
    }
    _FACTS_CACHE[key] = facts
    return facts


def _verify_allow_list_digest(bank_dir, raw, path):
    """When the bank ships SHA256SUMS, the allow-list must match its line.

    A digest line is only worth having if a mismatch refuses: an edited allow-list
    that was not re-digested is a bank that no longer says what it shipped.
    """
    sums = Path(bank_dir) / 'SHA256SUMS'
    if not sums.exists():
        return
    digest = hashlib.sha256(raw).hexdigest()
    for line in sums.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) >= 2 and Path(parts[1].lstrip('*')).name == path.name:
            if parts[0].strip().casefold() != digest:
                raise ValueError(f'{path} does not match its digest line in {sums}; re-run the bank digest')
            return


def load_allow_list(bank_dir=None):
    """Return (folded-key -> allow-list key, sha256 of the file, path)."""
    bank_dir = Path(bank_dir) if bank_dir is not None else DEFAULT_BANK_DIR
    path = bank_dir / ALLOW_LIST_NAME
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise ValueError(f'space bank allow-list is missing: {path}') from None
    try:
        payload = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f'space bank allow-list is not valid JSON: {path}: {error}') from None
    if not isinstance(payload, dict) or payload.get('schema') != ALLOW_LIST_SCHEMA:
        raise ValueError(f'space bank allow-list must declare schema {ALLOW_LIST_SCHEMA!r}: {path}')
    allowed = payload.get('allow')
    if not isinstance(allowed, dict) or not allowed:
        raise ValueError(f'space bank allow-list must carry a non-empty "allow" object: {path}')
    table = {}
    for key, meta in allowed.items():
        folded = str(key).strip().casefold()
        if not folded:
            raise ValueError(f'space bank allow-list has an empty licence id: {path}')
        if folded in table:
            raise ValueError(f'space bank allow-list repeats licence id {key!r} ignoring case: {path}')
        if not isinstance(meta, dict):
            raise ValueError(f'space bank allow-list entry {key!r} must be an object: {path}')
        table[folded] = str(key).strip()
    _verify_allow_list_digest(bank_dir, raw, path)
    return table, hashlib.sha256(raw).hexdigest(), path


def licence_gate(repo, revision=None, *, bank_dir=None, api=None):
    """Decide whether an anonymous Space may edit ``repo`` at ``revision``.

    Returns one JSON-ready dict with ``decision``, ``reason`` and the Hub's licence
    string verbatim (``None`` when the card states none). No file is downloaded.
    """
    if not isinstance(repo, str) or not repo.strip():
        raise ValueError('repo must be a nonempty Hub repository id')
    repo = repo.strip()
    facts = _hub_facts(repo, revision, api=api)
    result = {
        'repo': repo,
        'revision': revision,
        'resolved_revision': facts['sha'],
        'decision': DECISION_REFUSED,
        'reason': None,
        'licence': facts['licence'],
        'licence_source': facts['licence_source'],
        'detail': None,
    }
    if facts['gated']:
        result['reason'] = GATED_REPO
        result['detail'] = (f'{repo} is gated on the Hub, and an anonymous Space cannot authenticate, '
                            f'so the file is refused before any download.')
        return result

    table, digest, _ = load_allow_list(bank_dir)
    result['allow_list_sha256'] = digest
    licence = facts['licence']
    allowed = ', '.join(table[key] for key in sorted(table))
    if licence is None:
        result['reason'] = LICENCE_UNKNOWN
        result['detail'] = (f'{repo} has {facts["found"]}, so it is not on the bank\'s allow-list '
                            f'(allowed: {allowed}).')
        return result
    if licence.strip().casefold() not in table:
        result['reason'] = LICENCE_UNKNOWN
        result['detail'] = (f'{repo} declares licence {licence!r}, which is not on the bank\'s allow-list '
                            f'(allowed: {allowed}).')
        return result
    result['decision'] = DECISION_ALLOWED
    result['reason'] = LICENCE_ACCEPTED
    result['detail'] = f'{repo} declares licence {licence!r}, which is on the bank\'s allow-list.'
    return result


def main(argv=None):
    """``python -m jdewitt.bank_licence REPO [--revision REV] [--bank DIR]`` -> one JSON."""
    import argparse

    parser = argparse.ArgumentParser(description='Licence and access gate from Hub metadata; never downloads.')
    parser.add_argument('repo', help='Hub repository id, e.g. Qwen/Qwen3-4B-Instruct-2507')
    parser.add_argument('--revision', default=None, help='pinned revision/sha; default is the Hub default branch')
    parser.add_argument('--bank', default=None, help='bank directory holding licences.json')
    args = parser.parse_args(argv)
    print(json.dumps(licence_gate(args.repo, args.revision, bank_dir=args.bank), indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
