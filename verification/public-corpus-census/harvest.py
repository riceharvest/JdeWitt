# SPDX-License-Identifier: AGPL-3.0-or-later
"""Harvest the public prompt corpora that abliteration releases actually use.

The census is deliberately split in two:

* **vendored** sources (a licence that permits redistribution) have their
  prompt text written to ``rows.jsonl`` with a per-row sha256 of the
  normalised prompt, so a suite can be rebuilt without the network;
* **referenced** sources (no licence stated, gated, or too large/sharded to
  vendor) are recorded in ``sources.json`` with dataset id, pinned revision,
  row count and licence, and the importer downloads them at that revision.

Nothing here guesses: every row carries its source id, pinned revision and the
licence the dataset card states. A source whose card states no licence is
recorded as ``unstated`` and is never vendored into this repository.

Run with the project environment:

    uv run python verification/public-corpus-census/harvest.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from huggingface_hub import hf_hub_download  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

# ---------------------------------------------------------------------------
# Manifest.  Every field is a citation: dataset id + pinned revision + the
# release file that references it.  Fetched 2026-11-22.
# ---------------------------------------------------------------------------
SOURCES = [
    # ---- vendored (redistributable licence) --------------------------------
    {
        'id': 'strongreject',
        'dataset': 'walledai/StrongREJECT',
        'revision': 'df54272b395ce9e584d2f8140519935425d8549f',
        'license': 'mit',
        'license_url': 'https://huggingface.co/datasets/walledai/StrongREJECT',
        'url': 'https://huggingface.co/datasets/walledai/StrongREJECT',
        'files': [{'path': 'data/train-00000-of-00001.parquet', 'column': 'prompt',
                   'category': 'category', 'language': 'en', 'split': 'train'}],
        'rows': 313, 'languages': ['en'], 'structure': 'prompt-only',
        'pairing': 'none', 'role': 'harmful evaluation',
        'referenced_by': [
            'Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16 model card: "450-case source set contains 150 AdvBench, 150 StrongREJECT, 150 XSTest"',
            'walledai/StrongREJECT dataset card',
        ],
        'vendored': True,
    },
    {
        'id': 'xstest',
        'dataset': 'Paul/XSTest',
        'revision': 'f600c994b256f12867dfa5b3eb3d545a3e62f8b5',
        'license': 'cc-by-4.0',
        'license_url': 'https://huggingface.co/datasets/Paul/XSTest',
        'url': 'https://huggingface.co/datasets/Paul/XSTest',
        'files': [{'path': 'xstest_prompts.csv', 'column': 'prompt',
                   'category': 'type', 'language': 'en', 'label': 'label',
                   'split': 'train'}],
        'rows': 450, 'languages': ['en'], 'structure': 'prompt-only',
        'pairing': 'safe/unsafe label per row', 'role': 'over-refusal (safe) + harmful evaluation',
        'referenced_by': [
            'Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16 model card (150 XSTest cases)',
            'Paul/XSTest dataset card',
        ],
        'vendored': True,
    },
    {
        'id': 'semantic-harmful',
        'dataset': 'heretic-org/Semantic-Harmful',
        'revision': '001ca2ceaef94a748235e0ba1366aee48436e286',
        'license': 'cc-by-4.0',
        'license_url': 'https://huggingface.co/datasets/heretic-org/Semantic-Harmful',
        'url': 'https://huggingface.co/datasets/heretic-org/Semantic-Harmful',
        'files': [{'path': 'data/train-00000-of-00001.parquet', 'column': 'text',
                   'language': 'en', 'split': 'train'}],
        'rows': 416, 'languages': ['en'], 'structure': 'prompt-only',
        'pairing': 'one-to-one with semantic-harmless (matched_pairs.json)',
        'role': 'harmful side of the heretic semantic contrast',
        'referenced_by': [
            'heretic-org/Semantic-Harmful card: built from mlabonne/harmful_behaviors for p-e-w/heretic',
            'AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-BF16 card: "heretic compute_refusal_directions"',
        ],
        'vendored': True,
    },
    {
        'id': 'semantic-harmless',
        'dataset': 'heretic-org/Semantic-Harmless',
        'revision': '7e9f2b01272da85f2be7a3437f31ac46698e8735',
        'license': 'cc-by-4.0',
        'license_url': 'https://huggingface.co/datasets/heretic-org/Semantic-Harmless',
        'url': 'https://huggingface.co/datasets/heretic-org/Semantic-Harmless',
        'files': [{'path': 'data/train-00000-of-00001.parquet', 'column': 'text',
                   'language': 'en', 'split': 'train'}],
        'rows': 416, 'languages': ['en'], 'structure': 'prompt-only',
        'pairing': 'one-to-one with semantic-harmful (matched_pairs.json)',
        'role': 'harmless side of the heretic semantic contrast',
        'referenced_by': [
            'heretic-org/Semantic-Harmless card: semantic match of mlabonne/harmless_alpaca',
        ],
        'vendored': True,
    },
    {
        'id': 'multilingual-harmless-harmful',
        'dataset': 'heretic-org/Multilingual-Harmless-Harmful',
        'revision': 'f5a870a407217f5c4087d6a669b668bcf5da9b4d',
        'license': 'cc-by-sa-4.0',
        'license_url': 'https://huggingface.co/datasets/heretic-org/Multilingual-Harmless-Harmful',
        'url': 'https://huggingface.co/datasets/heretic-org/Multilingual-Harmless-Harmful',
        'files': [],  # built below from content/<language>/csv/<language>.csv
        'rows': 8320, 'languages': ['en', 'bn', 'fr', 'hi', 'it', 'ja', 'ko', 'mr', 'ru', 'ta'],
        'structure': 'prompt-only', 'pairing': '416 harmful + 416 harmless per language',
        'role': 'multilingual harmful/harmless contrast',
        'referenced_by': [
            'heretic-org/Multilingual-Harmless-Harmful card (config.multilingual.toml)',
        ],
        'vendored': True,
    },
    {
        'id': 'deccp',
        'dataset': 'augmxnt/deccp',
        'revision': '2c964c6957c5b1692992eb0d0409633155417056',
        'license': 'apache-2.0',
        'license_url': 'https://huggingface.co/datasets/augmxnt/deccp',
        'url': 'https://huggingface.co/datasets/augmxnt/deccp',
        'files': [{'path': 'data/censored-00000-of-00001.parquet', 'column': 'text',
                   'language': 'en', 'split': 'censored'},
                  {'path': 'data/uncensored-00000-of-00001.parquet', 'column': 'text',
                   'language': 'en', 'split': 'uncensored'}],
        'rows': 132, 'languages': ['en'],
        'structure': 'prompt-only (censored topics posed in English)',
        'pairing': 'censored set 95 / uncensored set 37',
        'role': 'Chinese-censorship refusal contrast (DECCP)',
        'referenced_by': [
            'augmxnt/Qwen2-7B-Instruct-deccp release card',
            'augmxnt/deccp dataset card',
        ],
        'vendored': True,
    },
    # ---- referenced only (no stated licence, gated, or too large) ----------
    {
        'id': 'heretic-harmful',
        'dataset': 'heretic-org/harmful_behaviors',
        'revision': '55eeb6b587ea43b579197db191183be93c653bea',
        'license': 'unstated',
        'license_url': None,
        'url': 'https://huggingface.co/datasets/heretic-org/harmful_behaviors',
        'files': [], 'rows': 520, 'languages': ['en'], 'structure': 'prompt-only',
        'pairing': 'none', 'role': 'heretic default harmful prompts',
        'referenced_by': [
            'p-e-w/heretic src/heretic/config.py: response_prefix_test_prompts dataset="mlabonne/harmful_behaviors"',
            'heretic-org/harmful_behaviors card: mirror of mlabonne/harmful_behaviors ("just in case")',
            'AEON-7 abliterated cards (heretic framework)',
        ],
        'vendored': False,
        'license_note': 'Dataset card states no licence and no upstream licence; mirror of mlabonne/harmful_behaviors (also unstated). AdvBench upstream is MIT but that is not restated here.',
    },
    {
        'id': 'heretic-harmless',
        'dataset': 'heretic-org/harmless_alpaca',
        'revision': '50f4f3fb105af89081223619eb9d431710f5829e',
        'license': 'unstated',
        'license_url': None,
        'url': 'https://huggingface.co/datasets/heretic-org/harmless_alpaca',
        'files': [], 'rows': 31323, 'languages': ['en'], 'structure': 'prompt-only',
        'pairing': 'none', 'role': 'heretic default harmless prompts',
        'referenced_by': [
            'p-e-w/heretic src/heretic/config.py: batch_size_test_prompts dataset="mlabonne/harmless_alpaca"',
            'heretic-org/harmless_alpaca card: mirror of mlabonne/harmless_alpaca',
        ],
        'vendored': False,
        'license_note': 'Card states no licence; mirror of mlabonne/harmless_alpaca (unstated). tatsu-lab/alpaca upstream is Apache-2.0 but the mirror does not restate it.',
    },
    {
        'id': 'advbench-harmful',
        'dataset': 'mlabonne/harmful_behaviors',
        'revision': '01cead01398926d81f7c52bdb790ee8cf77ebba7',
        'license': 'unstated',
        'license_url': None,
        'url': 'https://huggingface.co/datasets/mlabonne/harmful_behaviors',
        'files': [], 'rows': 520, 'languages': ['en'], 'structure': 'prompt-only',
        'pairing': 'none', 'role': 'AdvBench harmful behaviours, the shared harmful list',
        'referenced_by': [
            'p-e-w/heretic config.py',
            'examples/public-corpus/import.json (our own previous import)',
            'Blackfrost 450-case bench (150 AdvBench)',
        ],
        'vendored': False,
        'license_note': 'No licence on the dataset card; AdvBench (llm-attacks) is MIT upstream.',
    },
    {
        'id': 'alpaca-harmless',
        'dataset': 'mlabonne/harmless_alpaca',
        'revision': '02c6a92cfcf11bb0c387334f8146d149d65b587f',
        'license': 'unstated',
        'license_url': None,
        'url': 'https://huggingface.co/datasets/mlabonne/harmless_alpaca',
        'files': [], 'rows': 31323, 'languages': ['en'], 'structure': 'instruction-only',
        'pairing': 'none', 'role': 'Alpaca harmless instructions',
        'referenced_by': ['p-e-w/heretic config.py', 'FailSpy/abliterator get_harmless_instructions -> tatsu-lab/alpaca'],
        'vendored': False,
        'license_note': 'No licence on the mirror card; tatsu-lab/alpaca is Apache-2.0.',
    },
    {
        'id': 'josie-prompts-300',
        'dataset': 'Goekdeniz-Guelmez/JOSIE-Prompts-Only-Corpus-300',
        'revision': '9a5ffe35a18ee2c33f3b828d32c39d2219124bc8',
        'license': 'unstated',
        'license_url': None,
        'url': 'https://huggingface.co/datasets/Goekdeniz-Guelmez/JOSIE-Prompts-Only-Corpus-300',
        'files': [], 'rows': 300, 'languages': ['en'], 'structure': 'prompt-only',
        'pairing': 'none', 'role': 'Josiefied / JOSIE prompt corpus (300)',
        'referenced_by': [
            'Goekdeniz-Guelmez/Josiefied-Qwen2.5-*-abliterated-v* model cards (fine-tuned on a custom JOSIE dataset)',
            'Goekdeniz-Guelmez/JOSIE-Prompts-Only-Corpus-300 card',
        ],
        'vendored': False,
        'license_note': 'No licence field on the dataset card.',
    },
    {
        'id': 'josiefication-prompts',
        'dataset': 'Goekdeniz-Guelmez/Josiefication-prompts-online-po',
        'revision': '0006ee4cfb49a2b72ad4bd8974db1286eb40c83c',
        'license': 'unstated',
        'license_url': None,
        'url': 'https://huggingface.co/datasets/Goekdeniz-Guelmez/Josiefication-prompts-online-po',
        'files': [], 'rows': 4019573, 'languages': ['en'], 'structure': 'prompt-only',
        'pairing': 'none', 'role': 'large scraped Josiefication prompt corpus',
        'referenced_by': ['Goekdeniz-Guelmez Josiefied model family; dataset card'],
        'vendored': False,
        'license_note': 'No licence field; 4.0M rows, not vendorable and not needed for a wave.',
    },
    {
        'id': 'g0dm0d3',
        'dataset': 'pliny-the-prompter/g0dm0d3',
        'revision': 'f8055e2243597c86c49abbbf8765ccd400728c37',
        'license': 'agpl-3.0',
        'license_url': 'https://huggingface.co/datasets/pliny-the-prompter/g0dm0d3',
        'url': 'https://huggingface.co/datasets/pliny-the-prompter/g0dm0d3',
        'files': [], 'rows': None, 'languages': ['en'], 'structure': 'sharded jsonl telemetry/prompts',
        'pairing': 'none', 'role': "Pliny the Prompter's harvested prompt corpus",
        'referenced_by': [
            'pliny-the-prompter/g0dm0d3 dataset card (30,424 files: incoming_v2/ + telemetry/)',
            'OBLITERATUS model cards: "Built by Pliny the Prompter"',
        ],
        'vendored': False,
        'rows_note': '30,424 sharded jsonl files; the dataset card gives no row total and the viewer has no parquet, so no row count is claimed here.',
    },
    {
        'id': 'obliteratus-telemetry',
        'dataset': 'pliny-the-prompter/OBLITERATUS-TELEMETRY',
        'revision': '30ecd955180b6914fa0ffde54df15ea8521a1b6a',
        'license': 'agpl-3.0',
        'license_url': 'https://huggingface.co/datasets/pliny-the-prompter/OBLITERATUS-TELEMETRY',
        'url': 'https://huggingface.co/datasets/pliny-the-prompter/OBLITERATUS-TELEMETRY',
        'files': [], 'rows': None, 'languages': ['en'], 'structure': 'run telemetry (not a prompt corpus)',
        'pairing': 'none', 'role': 'OBLITERATUS crowd telemetry',
        'referenced_by': [
            'elder-plinius/OBLITERATUS README: "--contribute feeds the community dataset"',
            'pliny-the-prompter/OBLITERATUS-TELEMETRY card',
        ],
        'vendored': False,
        'rows_note': 'Telemetry records, not prompt/response pairs; no row total claimed.',
    },
    {
        'id': 'advbench-official',
        'dataset': 'llm-attacks/advbench',
        'revision': None,
        'license': 'mit',
        'license_url': 'https://github.com/llm-attacks/llm-attacks',
        'url': 'https://huggingface.co/datasets/llm-attacks/advbench',
        'files': [], 'rows': 520, 'languages': ['en'], 'structure': 'prompt-only',
        'pairing': 'none', 'role': 'AdvBench harmful behaviours (official)',
        'referenced_by': [
            'FailSpy/abliterator abliterator.py get_harmful_instructions() -> load_dataset(hf_path) with ["goal"] column, test split',
            'Blackfrost 450-case bench (150 AdvBench)',
        ],
        'vendored': False,
        'reference_only_reason': 'HF dataset repo is gated (HTTP 401 without an accepted gate); cited from the llm-attacks repo (MIT).',
        'revision_note': 'No pinned revision reachable without accepting the gate; the importer must not use this entry until a revision is recorded.',
    },
    {
        'id': 'alpaca-official',
        'dataset': 'tatsu-lab/alpaca',
        'revision': 'dce01c9b08f87459cf36a430d809084718273017',
        'license': 'cc-by-nc-4.0',
        'license_url': 'https://huggingface.co/datasets/tatsu-lab/alpaca',
        'url': 'https://huggingface.co/datasets/tatsu-lab/alpaca',
        'files': [], 'rows': 52002, 'languages': ['en'], 'structure': 'instruction+input+output',
        'pairing': 'none', 'role': 'abliterator harmless instructions',
        'referenced_by': ['FailSpy/abliterator abliterator.py get_harmless_instructions() -> tatsu-lab/alpaca train, instruction where input is empty'],
        'vendored': False,
        'license_note': 'Card declares cc-by-nc-4.0 (non-commercial); referenced, never vendored.',
    },
    {
        'id': 'orca-abliterated',
        'dataset': None,
        'revision': None,
        'license': 'unstated',
        'license_url': None,
        'url': 'https://huggingface.co/orcarouter',
        'files': [], 'rows': None, 'languages': ['en'], 'structure': 'unknown',
        'pairing': 'unknown', 'role': 'ORCA / OrcaRouter abliterated release family',
        'referenced_by': [
            'orcarouter/*-abliterated model repos (e.g. orcarouter/Flint-abliterated-GGUF, orcarouter/Hy4-flash-abliterated-GGUF)',
            'orcarouter model cards are gated (HTTP 401 without an accepted gate) at census time',
        ],
        'vendored': False,
        'reference_only_reason': 'The release publishes no dataset id; its model cards are gated, so no corpus can be cited. Recorded as unresolved rather than guessed.',
    },
    {
        'id': 'blackfrost-bench-450',
        'dataset': None,
        'revision': None,
        'license': 'composite (see parts)',
        'license_url': 'https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16',
        'url': 'https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16',
        'files': [], 'rows': 450, 'languages': ['en'],
        'structure': 'prompt-only composite: 150 AdvBench + 150 StrongREJECT + 150 XSTest',
        'pairing': 'none', 'role': 'Blackfrost release refusal bench (R1-HARMFUL-BENCH-450)',
        'referenced_by': [
            'Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16 card: "450-case source set contains 150 AdvBench, 150 StrongREJECT, and 150 XSTest prompts"',
            'Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF card: "11 residual refusals from 450 original cases"',
        ],
        'vendored': False,
        'composite_of': ['advbench-harmful', 'strongreject', 'xstest'],
        'license_note': 'Blackfrost publishes the composition, not a dataset id; the parts carry their own licences (AdvBench MIT, StrongREJECT MIT, XSTest CC-BY-4.0).',
    },
    {
        'id': 'aeon-heretic-defaults',
        'dataset': None,
        'revision': None,
        'license': 'composite (heretic defaults)',
        'license_url': 'https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-BF16',
        'url': 'https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-BF16',
        'files': [], 'rows': None, 'languages': ['en'], 'structure': 'prompt-only contrast pair via heretic defaults',
        'pairing': 'heretic harmful/harmless contrast', 'role': 'AEON-7 abliterated family',
        'referenced_by': [
            'AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-BF16 card: "Extract per-layer refusal directions from base via heretic\'s compute_refusal_directions"',
            'p-e-w/heretic config.py defaults (mlabonne/harmful_behaviors + harmless_alpaca)',
        ],
        'vendored': False,
        'composite_of': ['heretic-harmful', 'heretic-harmless'],
        'license_note': 'AEON ships no prompt corpus of its own; it uses heretic defaults.',
    },
]

MULTILINGUAL_LANGS = ['english', 'bengali', 'french', 'hindi', 'italian', 'japanese',
                      'korean', 'marathi', 'russian', 'tamil']
LANG_CODES = {'english': 'en', 'bengali': 'bn', 'french': 'fr', 'hindi': 'hi',
              'italian': 'it', 'japanese': 'ja', 'korean': 'ko', 'marathi': 'mr',
              'russian': 'ru', 'tamil': 'ta'}

# Our own taxonomy, read from the authored harsh suites. Coverage is a
# *lexical screen* across these families, not a claim that a source contains a
# topic family.
KEYWORDS = {
    'chemical-biological': ['chemical weapon', 'biological weapon', 'nerve agent', 'anthrax', 'toxin', 'bioweapon', 'sarin', 'ricin'],
    'copyright': ['copyright', 'plagiaris', 'lyrics', 'song lyrics', 'novel', 'book chapter'],
    'cybercrime-intrusion': ['hack', 'exploit', 'malware', 'ransomware', 'phishing', 'keylogger', 'sql injection', 'ddos', 'brute force', 'backdoor'],
    'harassment-bullying': ['harass', 'bully', 'stalk', 'threaten', 'intimidat', 'doxx', 'dox'],
    'illegal': ['illegal', 'smuggl', 'drug', 'cocaine', 'heroin', 'weapon', 'firearm', 'explosive', 'counterfeit', 'launder'],
    'misinformation-disinformation': ['misinformation', 'disinformation', 'propaganda', 'fake news', 'conspiracy', 'hoax'],
    'harmful': ['harm', 'hurt', 'kill', 'murder', 'attack', 'abuse'],
    'violence': ['violence', 'violent', 'assault', 'torture', 'mass shooting', 'bomb'],
    'sexual-content': ['sexual', 'explicit', 'porn', 'nude', 'erotic', 'minor'],
    'privacy': ['privacy', 'personal data', 'social security', 'address', 'phone number', 'surveillance', 'spy'],
    'fraud-deception': ['fraud', 'scam', 'deceive', 'deception', 'fake', 'forg', 'impersonat'],
    'non-violent-crimes': ['theft', 'steal', 'burglar', 'tax evasion', 'bribe', 'extort'],
    'illegal-goods-and-services': ['sell', 'traffick', 'black market', 'contraband', 'smuggle'],
    'expert-advice': ['medical advice', 'legal advice', 'diagnos', 'prescription', 'dosage'],
    'government-decision-making': ['government', 'election', 'vote', 'policy', 'immigration', 'benefit'],
    'economic-harm': ['econom', 'market', 'stock', 'insider trading', 'sabotage', 'bankrupt'],
    'physical-harm': ['poison', 'injure', 'wound', 'overdose', 'suffocat', 'electrocut'],
    'hate-harassment-and-discrimination': ['hate', 'racist', 'sexist', 'discriminat', 'slur', 'genocide'],
    'malware-hacking': ['malware', 'virus', 'trojan', 'worm', 'spyware', 'hacking'],
    'disinformation-and-deception': ['disinformation', 'deception', 'mislead', 'manipulat'],
}


def canonical(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def sha(text: str) -> str:
    return hashlib.sha256(canonical(text).encode()).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(source, rel_path):
    return Path(hf_hub_download(source['dataset'], rel_path, revision=source['revision'],
                                repo_type='dataset'))


def read_records(source, spec):
    path = fetch(source, spec['path'])
    suffix = path.suffix.casefold()
    if suffix == '.parquet':
        records = pq.read_table(path).to_pylist()
    elif suffix in {'.csv', '.tsv'}:
        with open(path, newline='', encoding='utf-8') as handle:
            records = list(csv.DictReader(handle, delimiter='\t' if suffix == '.tsv' else ','))
    elif suffix in {'.jsonl', '.ndjson'}:
        records = [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]
    else:
        raise SystemExit(f'unsupported file {path}')
    return path, records


def harvest(source):
    rows = []
    files = []
    if source['id'] == 'multilingual-harmless-harmful':
        for lang in MULTILINGUAL_LANGS:
            spec = {'path': f'content/{lang}/csv/{lang}.csv', 'column': 'text',
                    'category': 'type', 'language': LANG_CODES[lang], 'split': 'train'}
            path, records = read_records(source, spec)
            files.append({'path': spec['path'], 'sha256': file_digest(path), 'rows': len(records)})
            for index, record in enumerate(records):
                text = record['text']
                kind = (record.get('type') or '').strip().lower()
                rows.append({'source': source['id'], 'dataset': source['dataset'],
                             'revision': source['revision'], 'license': source['license'],
                             'language': LANG_CODES[lang], 'split': 'train',
                             'source_row': index, 'kind': kind or 'unknown',
                             'category': record.get('type') or None,
                             'sha256': sha(text), 'prompt': text})
        return rows, files
    for spec in source['files']:
        path, records = read_records(source, spec)
        files.append({'path': spec['path'], 'sha256': file_digest(path), 'rows': len(records)})
        for index, record in enumerate(records):
            text = record[spec['column']]
            if not isinstance(text, str) or not text.strip():
                raise SystemExit(f'{source["id"]} row {index} has an empty prompt')
            row = {'source': source['id'], 'dataset': source['dataset'],
                   'revision': source['revision'], 'license': source['license'],
                   'language': spec.get('language', 'en'), 'split': spec.get('split', 'train'),
                   'source_row': index, 'sha256': sha(text), 'prompt': text}
            if spec.get('category'):
                row['category'] = record.get(spec['category'])
            if spec.get('label'):
                row['label'] = record.get(spec['label'])
            rows.append(row)
    return rows, files


def coverage(rows):
    """Lexical coverage screen against our authored family vocabulary."""
    covered = {}
    for family, words in KEYWORDS.items():
        counts = {}
        for row in rows:
            text = canonical(row['prompt'])
            hits = sum(1 for word in words if word in text)
            if hits:
                counts[row['source']] = counts.get(row['source'], 0) + 1
        covered[family] = dict(sorted(counts.items()))
    return covered


def main():
    all_rows, source_records = [], []
    for source in SOURCES:
        record = {k: v for k, v in source.items() if k != 'files'}
        if source['vendored']:
            rows, files = harvest(source)
            all_rows.extend(rows)
            record['file_digests'] = files
            record['rows_harvested'] = len(rows)
            record['dataset_sha256'] = source['revision']
        else:
            record['rows_harvested'] = 0
        source_records.append(record)
        print(f"{source['id']}: {record.get('rows_harvested', 0)} rows")

    # Dedupe on the normalised prompt.
    groups = {}
    for row in all_rows:
        groups.setdefault(row['sha256'], []).append(row)
    unique, duplicates = [], []
    for key, members in groups.items():
        unique.append(members[0])
        if len(members) > 1:
            duplicates.append({
                'sha256': key,
                'prompt': members[0]['prompt'],
                'sources': sorted({m['source'] for m in members}),
                'members': [{'source': m['source'], 'language': m['language'],
                             'source_row': m['source_row']} for m in members],
            })
    per_source = {}
    for row in all_rows:
        entry = per_source.setdefault(row['source'], {'rows': 0, 'unique_after_dedupe': 0})
        entry['rows'] += 1
    unique_keys = {row['sha256'] for row in unique}
    for row in unique:
        per_source[row['source']]['unique_after_dedupe'] += 1
    # A source's "unique" count is its own rows that are not duplicated anywhere
    # later in the file order; report it per source instead of the global set.
    per_source = {}
    for source in SOURCES:
        if not source['vendored']:
            continue
        own = [r for r in all_rows if r['source'] == source['id']]
        own_keys = {r['sha256'] for r in own}
        dup_keys = {d['sha256'] for d in duplicates if source['id'] in d['sources']}
        per_source[source['id']] = {'rows_harvested': len(own),
                                    'unique_prompts_within_source': len(own_keys),
                                    'prompts_shared_with_another_source': len(own_keys & dup_keys),
                                    'duplicate_rows_dropped_if_concatenated': len(own) - len(own_keys)}

    coverage_matrix = coverage(unique)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'rows.jsonl', 'w', encoding='utf-8') as handle:
        for row in sorted(unique, key=lambda r: (r['source'], r['source_row'])):
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    census = {
        'format': 'jdewitt-public-corpora/1',
        'generated': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'harvest': 'verification/public-corpus-census/harvest.py',
        'vendored_rows': len(all_rows),
        'unique_prompts': len(unique),
        'cross_source_duplicate_groups': len(duplicates),
        'sources': source_records,
    }
    (OUT / 'sources.json').write_text(json.dumps(census, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    dedupe_report = {
        'method': 'sha256 of NFKC-normalised, casefolded, whitespace-collapsed prompt',
        'rows_in': len(all_rows),
        'unique_prompts': len(unique),
        'duplicate_rows': len(all_rows) - len(unique),
        'cross_source_duplicate_groups': len(duplicates),
        'per_source': per_source,
        'duplicate_groups': sorted(duplicates, key=lambda d: (-len(d['members']), d['sha256'])),
    }
    (OUT / 'dedupe.json').write_text(json.dumps(dedupe_report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (OUT / 'coverage.json').write_text(json.dumps({
        'method': 'lexical keyword screen over the authored harsh-suite family vocabulary; a hit is a coverage lead, not a family claim',
        'families': coverage_matrix,
        'not_checked': ['that a keyword hit is the same topic family as ours',
                        'that a source covers a family it does not mention lexically',
                        'harmfulness of any prompt to a particular model'],
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'wrote {len(unique)} unique rows, {len(duplicates)} duplicate groups')


if __name__ == '__main__':
    main()
