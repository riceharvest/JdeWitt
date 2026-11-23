# SPDX-License-Identifier: AGPL-3.0-or-later
"""A §5.1 per-model `entry.json` and the loader's `bank.json` must never drift.

`docs/SPACE-ROADMAP.md` §5.1 calls one entry + one bundle the bank's unit, so the
refit card writes `spaces/bank/<model>/entry.json`. The loader card ships
`spaces/bank/bank.json`, which is the document `bank-check`/`bank-apply` read. Two
copies of one object is a drift hazard, so this test is the thing that keeps them
the same object: every per-model file must appear in the loader document with equal
content, and its bundle must sit beside `bank.json` at the file name it pins.

It runs over whatever entries the bank holds (zero is fine), so adding the next
model extends the test without editing it.
"""
import json
from pathlib import Path

BANK = Path(__file__).resolve().parents[1] / 'spaces' / 'bank'


def _entries():
    document = json.loads((BANK / 'bank.json').read_text())
    assert document['schema'] == 'jdewitt-space-bank-v1'
    return {entry['key']: entry for entry in document['entries']}


def _per_model_files():
    return sorted(BANK.glob('*/entry.json'))


def test_the_bank_holds_at_least_the_first_entry():
    """The refit card seeds exactly one entry; an empty document would pass the
    drift test vacuously and hide a bank that was never seeded."""
    assert _entries(), 'spaces/bank/bank.json carries no entries'


def test_every_per_model_entry_is_the_loader_document_entry():
    entries = _entries()
    for path in _per_model_files():
        entry = json.loads(path.read_text())
        key = entry['key']
        assert path.parent.name in key, f'{path} sits under a directory that is not the entry key'
        assert key in entries, f'{path} carries {key!r}, which bank.json does not'
        assert entries[key] == entry, f'{path} and bank.json disagree on {key!r}'


def test_every_entry_bundle_sits_beside_the_document_at_the_pinned_digest():
    import hashlib

    for key, entry in _entries().items():
        name = entry['bundle']['file']
        bundle = BANK / name
        assert bundle.is_file(), f'{key!r} pins {name!r}, which is not beside bank.json'
        digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
        assert digest == entry['bundle']['sha256'], f'{key!r} pins a different digest'
