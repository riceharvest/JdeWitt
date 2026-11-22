# SPDX-License-Identifier: AGPL-3.0-or-later
"""Import public prompt corpora into this tool's suite format.

The importer is deliberately narrow. It never moves prompts between splits,
never invents maths/code tasks, never guesses a language and never claims a
synthetic row grouping is a genuine topic family. Everything it does to a
corpus is recorded in the provenance file next to the suites.
"""
import csv
import datetime
import hashlib
import io
import json
import re
import stat
import string
from pathlib import Path

from .answers import atomic_json, atomic_text
from .data import canonical_prompt, digest, load_suite, validate_splits

KINDS = ('refusal', 'harmless', 'math', 'code', 'direction')
SPLITS = ('directions', 'selection', 'holdout')
FAMILY_STRATEGIES = ('column', 'row-group', 'constant')
TEXT_SUFFIXES = {'.jsonl', '.ndjson', '.json', '.csv', '.tsv', '.txt', '.parquet'}
# One prompt corpus is prompts, not a dataset dump: a file that big is a mistake (or a
# device/fifo pretending to be a file), and materialising it is how an importer OOMs.
LOCAL_FILE_MAX_BYTES = 4 << 30
LOCAL_TEXT_MAX_CHARS = 256 << 20


def token_set(text):
    return set(re.findall(r'[a-z0-9]+', canonical_prompt(text)))


def jaccard(left, right):
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def near_duplicate_dedupe(splits, threshold):
    """Drop prompts that restate an earlier one, across every split.

    Verbatim dedupe misses reworded duplicates: corpora share prompts with one
    word changed ("US president" vs "U.S. president"). First occurrence wins in
    split order, so a holdout prompt that restates a selection prompt is dropped
    rather than quietly leaking into certification.
    """
    if not 0 < threshold <= 1:
        raise ValueError('dedupe jaccard threshold must be in (0,1]')
    kept, dropped = {}, {}
    seen = []
    for split in SPLITS:
        if split not in splits:
            continue
        rows = []
        for row in splits[split]:
            tokens = token_set(row['prompt'])
            duplicate_of = next((other['id'] for other in seen if jaccard(tokens, other['tokens']) >= threshold), None)
            if duplicate_of is not None:
                dropped.setdefault(split, []).append({'id': row['id'], 'duplicate_of': duplicate_of})
                continue
            seen.append({'id': row['id'], 'tokens': tokens, 'split': split})
            rows.append(row)
        kept[split] = rows
    return kept, dropped


def slug(value):
    text = re.sub(r'[^0-9a-z]+', '-', str(value).strip().casefold()).strip('-')
    if not text:
        raise ValueError(f'Cannot build a safe identifier from {value!r}')
    return text[:64]


def local_file_audit(path):
    """Stat one importer input: regular target, bounded size, symlink named.

    Hugging Face's cache hands out symlinks into ``blobs/``, so a symlink is
    followed AND recorded rather than refused; a FIFO, device or directory behind
    the path is refused because reading it would block or never end.
    """
    path = Path(path)
    try:
        info = path.stat()
    except FileNotFoundError:
        raise ValueError(f'Missing local prompt file: {path}')
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'{path} is not a regular file (mode {oct(stat.S_IMODE(info.st_mode))}); '
                         f'a FIFO, device or directory cannot be read as a corpus')
    if info.st_size > LOCAL_FILE_MAX_BYTES:
        raise ValueError(f'{path} is {info.st_size} bytes; a prompt corpus above '
                         f'{LOCAL_FILE_MAX_BYTES} bytes is refused, not materialised')
    audit = {'bytes': info.st_size}
    if path.is_symlink():
        audit['symlink_to'] = str(path.resolve())
    return audit


def _file_sha256(path):
    """Chunked sha256: hashing a corpus must not decide whether it fits in memory."""
    digest_obj = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''):
            digest_obj.update(chunk)
    return digest_obj.hexdigest()


def _read_text_bounded(path):
    with open(path, 'r', encoding='utf-8', errors='strict') as handle:
        text = handle.read(LOCAL_TEXT_MAX_CHARS + 1)
    if len(text) > LOCAL_TEXT_MAX_CHARS:
        raise ValueError(f'{path} holds more than {LOCAL_TEXT_MAX_CHARS} characters; refused')
    return text


def read_local(path, column=None):
    """Read one local corpus file into records. No network, no guessing."""
    path = Path(path)
    local_file_audit(path)
    suffix = path.suffix.casefold()
    if suffix not in TEXT_SUFFIXES:
        raise ValueError(f'Unsupported prompt file type {suffix!r}; expected one of {sorted(TEXT_SUFFIXES)}')
    if suffix in {'.jsonl', '.ndjson'}:
        records, total = [], 0
        with open(path, 'r', encoding='utf-8', errors='strict') as handle:
            for line in handle:
                total += len(line)
                if total > LOCAL_TEXT_MAX_CHARS:
                    raise ValueError(f'{path} holds more than {LOCAL_TEXT_MAX_CHARS} characters; refused')
                if line.strip():
                    records.append(json.loads(line))
    elif suffix == '.json':
        loaded = json.loads(_read_text_bounded(path))
        records = loaded if isinstance(loaded, list) else loaded.get('data')
    elif suffix in {'.csv', '.tsv'}:
        delimiter = '\t' if suffix == '.tsv' else ','
        records = list(csv.DictReader(io.StringIO(_read_text_bounded(path)), delimiter=delimiter))
    elif suffix == '.parquet':
        records = read_parquet(path)
    else:
        records, total = [], 0
        with open(path, 'r', encoding='utf-8', errors='strict') as handle:
            for line in handle:
                total += len(line)
                if total > LOCAL_TEXT_MAX_CHARS:
                    raise ValueError(f'{path} holds more than {LOCAL_TEXT_MAX_CHARS} characters; refused')
                if line.strip():
                    records.append(line)
    return records


def read_parquet(path):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError('Reading parquet prompt corpora needs the optional datasets extra: '
                           'uv pip install "jdewitt[datasets]"') from error
    return pq.read_table(str(path)).to_pylist()


def resolve_hub_file(dataset, revision, split, config=None):
    """Pick one data file for a split. Deterministic; the chosen name is recorded."""
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(dataset, repo_type='dataset', revision=revision)
    prefix = f'{config}/' if config else ''
    candidates = [name for name in files if name.startswith(prefix) and Path(name).suffix.casefold() in TEXT_SUFFIXES]
    if not candidates:
        raise ValueError(f'{dataset}@{revision} exposes no importable data file')
    token = (split or '').split('[')[0].strip()
    if token:
        matched = [name for name in candidates if token.casefold() in Path(name).name.casefold()]
        if not matched:
            raise ValueError(f'{dataset}@{revision} has no data file matching split {token!r}')
        candidates = matched
    elif len(candidates) > 1:
        raise ValueError(f'{dataset}@{revision} exposes several data files; an explicit split is required')
    order = {'.parquet': 0, '.csv': 1, '.jsonl': 2, '.ndjson': 2, '.json': 3, '.txt': 4}
    return sorted(candidates, key=lambda name: (order.get(Path(name).suffix.casefold(), 9), name))[0]


def read_hub(source):
    from huggingface_hub import hf_hub_download
    dataset, revision = source['dataset'], source.get('revision')
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError('Pinned dataset revision is required; a moving branch is not reproducible')
    explicit = source.get('file')
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValueError('file must be a nonempty repository path')
        if Path(explicit).suffix.casefold() not in TEXT_SUFFIXES:
            raise ValueError(f'file must end in one of {sorted(TEXT_SUFFIXES)}')
        filename = explicit
    else:
        filename = resolve_hub_file(dataset, revision, source.get('split'), source.get('config'))
    path = Path(hf_hub_download(dataset, filename, revision=revision, repo_type='dataset'))
    records = read_local(path, source.get('column'))
    return records, {'dataset': dataset, 'revision': revision, 'file': filename,
                     'file_sha256': _file_sha256(path), **local_file_audit(path)}


def _normalize_revision_time(value):
    """HF returns float epoch or datetime depending on cache version; normalise both."""
    if isinstance(value, (int, float)):
        dt = datetime.datetime.fromtimestamp(float(value), datetime.timezone.utc)
        return dt.isoformat(), dt.timestamp()
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        return value.isoformat(), value.timestamp()
    return None, None


def corpus_cache_report():
    """Inventory the on-disk dataset import cache. Read-only; nothing is deleted."""
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
    revisions = []
    for repo in info.repos:
        if repo.repo_type != 'dataset':
            continue
        for rev in repo.revisions:
            iso, _ = _normalize_revision_time(rev.last_modified)
            revisions.append({'repo': repo.repo_id, 'revision': rev.commit_hash,
                              'refs': sorted(rev.refs), 'files': len(rev.files),
                              'bytes': rev.size_on_disk, 'last_modified': iso})
    revisions.sort(key=lambda r: (r['repo'], r['last_modified'] or ''))
    return {'cache_dir': str(info.cache_dir), 'datasets': revisions,
            'dataset_bytes': sum(r['bytes'] for r in revisions),
            'incomplete_files': len(info.incomplete_files),
            'warnings': [str(w) for w in info.warnings]}


def corpus_cache_prune(older_than_days, execute=False):
    """Preview or delete unreferenced dataset cache revisions older than N days.

    Revisions reachable by a branch or tag ref are never candidates; the importer
    only ever reads pinned revisions anyway. Without ``execute`` this only reports.
    """
    import time
    from huggingface_hub import scan_cache_dir
    if type(older_than_days) is not int or older_than_days < 1:
        raise ValueError('older_than_days must be an integer >= 1')
    info = scan_cache_dir()
    cutoff = time.time() - older_than_days * 86400
    doomed = []
    for repo in info.repos:
        if repo.repo_type != 'dataset':
            continue
        for rev in repo.revisions:
            if rev.refs:
                continue
            _, epoch = _normalize_revision_time(rev.last_modified)
            if epoch is None or epoch >= cutoff:
                continue
            doomed.append({'repo': repo.repo_id, 'revision': rev.commit_hash, 'bytes': rev.size_on_disk})
    result = {'cutoff_days': older_than_days, 'candidates': doomed,
              'candidate_bytes': sum(d['bytes'] for d in doomed), 'executed': False,
              'note': 'Pinned revisions (branch/tag refs) are never deleted.'}
    if doomed and execute:
        info.delete_revisions(*(d['revision'] for d in doomed)).execute()
        result['executed'] = True
    return result


def prune_small_families(splits, minimum):
    """Drop rows whose family no longer has enough variants after dedupe.

    Dedupe removes prompts, not families, so a two-variant family can lose one
    member and become unusable. Leaving it in would emit a suite that fails the
    run's consistency policy, so the orphaned rows are removed and reported.
    """
    if type(minimum) is not int or minimum < 2:
        raise ValueError('min_family_size must be an integer of at least 2')
    kept, dropped = {}, {}
    for split, rows in splits.items():
        sizes = {}
        for row in rows:
            if row.get('family_id'):
                sizes[row['family_id']] = sizes.get(row['family_id'], 0) + 1
        survivors, orphans = [], []
        for row in rows:
            family = row.get('family_id')
            if family and sizes[family] < minimum:
                orphans.append({'id': row['id'], 'family_id': family, 'family_size': sizes[family]})
                continue
            survivors.append(row)
        kept[split] = survivors
        if orphans:
            dropped[split] = orphans
    return kept, dropped


def resolve_value(record, spec, field):
    """A field is either a literal or {'column': name}. Nothing is inferred."""
    if not isinstance(spec, dict):
        return spec
    column = spec.get('column')
    if set(spec) != {'column'} or not isinstance(column, str) or not column.strip():
        raise ValueError(f'Invalid {field} mapping')
    if not isinstance(record, dict) or column not in record:
        raise ValueError(f'Source lacks the {field} column {column!r}')
    value = record[column]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Source has an empty {field} value in column {column!r}')
    return value.strip()


def apply_where(records, where):
    """Filter rows on one exact column value.

    Corpora routinely mix harmful and benign rows in a single file (WildJailbreak
    ships adversarial_harmful and adversarial_benign together), and splitting that
    into two sources must be the tool's job, not a hand-edited copy.
    """
    if not isinstance(where, dict) or set(where) != {'column', 'equals'}:
        raise ValueError('where accepts exactly a column and an equals value')
    column, wanted = where['column'], where['equals']
    if not isinstance(column, str) or not column.strip() or not isinstance(wanted, str):
        raise ValueError('where needs a column name and a string value')
    kept = [record for record in records
            if isinstance(record, dict) and str(record.get(column)) == wanted]
    if not kept:
        raise ValueError(f'where {column}={wanted!r} matched no rows')
    return kept


def format_prompt(record, template):
    """Compose a prompt from several columns, e.g. HarmBench context + behaviour."""
    if not isinstance(template, str) or not template.strip():
        raise ValueError('prompt template must be a nonempty string')
    fields = [name for _, name, _, _ in string.Formatter().parse(template) if name]
    if not fields:
        raise ValueError('prompt template needs at least one {column} placeholder')
    if not isinstance(record, dict):
        raise ValueError('A prompt template needs record-shaped sources')
    values = {}
    for name in fields:
        if name not in record:
            raise ValueError(f'prompt template references missing column {name!r}')
        value = record[name]
        if value is None:
            value = ''
        if not isinstance(value, str):
            raise ValueError(f'prompt template column {name!r} is not text')
        values[name] = value.strip()
    prompt = template.format(**values).strip()
    if not prompt:
        raise ValueError('prompt template produced an empty prompt')
    return prompt


def extract_prompts(records, column, list_column=None):
    if not records:
        raise ValueError('Prompt source is empty')
    if list_column is not None:
        # Some corpora store a multi-turn prompt as the repr of a list of strings
        # (Sorry-Bench). Joining the turns is the intended prompt; the raw repr
        # would feed the model Python syntax.
        import ast
        if not isinstance(list_column, str) or not list_column.strip():
            raise ValueError('list_column must be a column name')
        prompts = []
        for record in records:
            if not isinstance(record, dict) or list_column not in record:
                raise ValueError(f'Source lacks the list column {list_column!r}')
            raw = record[list_column]
            if isinstance(raw, str):
                try:
                    raw = ast.literal_eval(raw)
                except (ValueError, SyntaxError) as error:
                    raise ValueError(f'list column {list_column!r} is not a list literal') from error
            if not isinstance(raw, (list, tuple)) or not raw or any(not isinstance(item, str) for item in raw):
                raise ValueError(f'list column {list_column!r} must hold a nonempty list of strings')
            joined = '\n\n'.join(item.strip() for item in raw if item.strip())
            if not joined:
                raise ValueError(f'list column {list_column!r} produced an empty prompt')
            prompts.append(joined)
        return prompts
    if column is None:
        if any(not isinstance(record, str) for record in records):
            raise ValueError('Prompt column is required for record-shaped sources')
        return list(records)
    if not isinstance(column, str) or not column.strip():
        raise ValueError('Invalid prompt column')
    prompts = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get(column), str):
            raise ValueError(f'Source lacks the prompt column {column!r}')
        prompts.append(record[column])
    return prompts


def family_ids(records, source):
    strategy = source['family']
    if not isinstance(strategy, dict) or strategy.get('strategy') not in FAMILY_STRATEGIES:
        raise ValueError(f'family.strategy must be one of {list(FAMILY_STRATEGIES)}')
    # `group` lets a paired good/bad corpus share family ids; without it a
    # direction suite could never contain matched families.
    group = strategy.get('group', source['id'])
    keys = {key for key in strategy if key != 'group'}
    name, chosen = slug(group), strategy['strategy']
    if chosen == 'constant':
        if keys != {'strategy', 'value'}:
            raise ValueError('constant family strategy needs exactly a value')
        return [f'{name}-{slug(strategy["value"])}'] * len(records), 'authored-constant'
    if chosen == 'column':
        if keys != {'strategy', 'column'}:
            raise ValueError('column family strategy needs exactly a column')
        return ([f'{name}-{slug(resolve_value(record, {"column": strategy["column"]}, "family"))}'
                 for record in records], 'source-column')
    size = strategy.get('size')
    if keys != {'strategy', 'size'} or type(size) is not int or size < 2:
        raise ValueError('row-group family strategy needs an integer size of at least 2')
    offset = source['offset']
    groups = [f'{name}-o{offset}-g{index // size:06d}' for index in range(len(records))]
    return groups, 'synthetic-row-group'


def passthrough_rows(source, split):
    """Reuse an authored suite unchanged. Maths and code are authored, never imported."""
    if set(source) - {'id', 'passthrough', 'path', 'kinds'}:
        raise ValueError('A passthrough source accepts only id, passthrough, path and kinds')
    if 'path' not in source:
        raise ValueError('A passthrough source needs a local path')
    kinds = source.get('kinds')
    if kinds is not None and (not isinstance(kinds, list) or not kinds or set(kinds) - set(KINDS)):
        raise ValueError(f'kinds must be a nonempty subset of {list(KINDS)}')
    path = Path(source['path'])
    rows = load_suite(path)
    if kinds is not None:
        rows = [row for row in rows if row['kind'] in kinds]
    if not rows:
        raise ValueError(f'Passthrough source {source["id"]!r} selected no rows')
    provenance = {'id': source['id'], 'kind': 'passthrough', 'split': split,
                  'path': str(path), 'file_sha256': _file_sha256(path),
                  **local_file_audit(path),
                  'rows_used': len(rows), 'duplicates_dropped': 0, 'kinds': kinds,
                  'family_strategy': None, 'family_origin': 'authored-file',
                  'family_count': len({row.get('family_id') for row in rows if row.get('family_id')})}
    return [dict(row) for row in rows], provenance


def source_rows(source, split):
    """Turn one declared source into suite rows. Order is the source's own order."""
    if not isinstance(source, dict) or 'id' not in source:
        raise ValueError('Each source needs an id')
    if source.get('passthrough'):
        return passthrough_rows(source, split)
    if 'kind' not in source:
        raise ValueError('Each source needs an id and a kind')
    kind = source['kind']
    if kind not in KINDS:
        raise ValueError(f'Unknown source kind {kind!r}')
    offset, limit = source.get('offset', 0), source.get('limit')
    if type(offset) is not int or offset < 0:
        raise ValueError('offset must be a nonnegative integer')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer or omitted')
    stride = source.get('stride', 1)
    if type(stride) is not int or stride < 1:
        raise ValueError('stride must be a positive integer or omitted')
    if kind == 'direction' and source.get('label') not in {'good', 'bad'}:
        raise ValueError('Direction sources require an explicit good/bad label')
    if kind in {'math', 'code'} and 'dataset' in source:
        raise ValueError('Maths and code tasks are authored, never imported from a corpus')
    if kind not in {'refusal', 'direction'} and 'family' in source:
        raise ValueError(f'{kind} tasks carry no family; drop the family block')
    if 'dataset' in source:
        records, provenance = read_hub(source)
    elif 'path' in source:
        path = Path(source['path'])
        records = read_local(path, source.get('column'))
        provenance = {'path': str(path), 'file_sha256': _file_sha256(path),
                      **local_file_audit(path)}
    else:
        raise ValueError('Each source needs either a dataset or a path')
    if 'where' in source:
        records = apply_where(records, source['where'])
    window = records[offset:][::stride]
    selected = window if limit is None else window[:limit]
    if not selected:
        raise ValueError(f'Source {source["id"]!r} selected no rows')
    if 'template' in source:
        if 'column' in source or 'list_column' in source:
            raise ValueError('Use either column/list_column or template for the prompt, not both')
        prompts = [format_prompt(record, source['template']) for record in selected]
    else:
        prompts = extract_prompts(selected, source.get('column'), source.get('list_column'))
    kept, dropped, seen = [], 0, set()
    for record, prompt in zip(selected, prompts):
        key = canonical_prompt(prompt)
        if not key:
            raise ValueError(f'Source {source["id"]!r} contains an empty prompt')
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append((record, prompt))
    if not kept:
        raise ValueError(f'Source {source["id"]!r} has no unique prompts')
    if kind in {'refusal', 'direction'}:
        families, origin = family_ids([record for record, _ in kept], {**source, 'offset': offset})
    else:
        families, origin = [None] * len(kept), None
    rows = []
    for index, ((record, prompt), family) in enumerate(zip(kept, families)):
        row = {'id': f'{slug(source["id"])}-{offset + index:06d}', 'kind': kind, 'prompt': prompt,
               'language': resolve_value(record, source['language'], 'language')}
        if kind == 'direction':
            row['label'] = source['label']
        if 'category' in source:
            category = resolve_value(record, source['category'], 'category')
            if category:
                row['category'] = category
        if kind in {'refusal', 'direction'}:
            row['family_id'] = family
        if kind == 'math':
            row['expected'] = resolve_value(record, source.get('expected'), 'expected')
        if kind == 'code':
            row['tests'] = resolve_value(record, source.get('tests'), 'tests')
        rows.append(row)
    family_count = len({family for family in families if family})
    provenance = {**provenance, 'id': source['id'], 'kind': kind, 'split': split, 'offset': offset,
                  'limit': limit, **({'stride': stride} if stride != 1 else {}),
                  'rows_used': len(rows), 'duplicates_dropped': dropped,
                  'family_strategy': source['family']['strategy'] if origin else None,
                  'family_origin': origin, 'family_count': family_count}
    return rows, provenance


CENSUS_FORMAT = 'jdewitt-public-corpora/1'


def load_census(path):
    """Read the canonical public-corpora inventory.

    The census is the only place a source's licence and revision are declared.
    ``--provenance`` makes every imported row carry that declaration, so a wave
    can be re-run against exactly the prompts a competitor used and the licence
    travels with the artifact instead of living in a README.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Missing provenance census: {path}')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        raise ValueError(f'Provenance census is not valid JSON: {error}') from error
    if not isinstance(data, dict) or data.get('format') != CENSUS_FORMAT:
        raise ValueError(f'Provenance census must declare format {CENSUS_FORMAT!r}')
    sources = data.get('sources')
    if not isinstance(sources, list) or not sources:
        raise ValueError('Provenance census needs a nonempty sources list')
    entries = {}
    for entry in sources:
        if not isinstance(entry, dict) or not isinstance(entry.get('id'), str) or not entry['id'].strip():
            raise ValueError('Every census source needs a nonempty id')
        if entry['id'] in entries:
            raise ValueError(f'Duplicate census source id {entry["id"]!r}')
        if not isinstance(entry.get('license'), str) or not entry['license'].strip():
            raise ValueError(f'Census source {entry["id"]!r} states no licence; '
                             f'it cannot be imported until one is recorded')
        entries[entry['id']] = entry
    return {'path': str(path), 'sha256': _file_sha256(path),
            'generated': data.get('generated'), 'sources': entries}


def _census_entry(census, source):
    entry = census['sources'].get(source['id']) if census else None
    if census is not None and entry is None:
        raise ValueError(f'Source {source["id"]!r} is not in the provenance census; '
                         f'every imported row needs a recorded source and licence')
    if entry is None:
        return None
    if 'dataset' in source:
        if entry.get('dataset') != source['dataset']:
            raise ValueError(f'Census dataset for {source["id"]!r} is {entry.get("dataset")!r}, '
                             f'spec says {source["dataset"]!r}')
        if entry.get('revision') != source.get('revision'):
            raise ValueError(f'Census revision for {source["id"]!r} is {entry.get("revision")!r}, '
                             f'spec says {source.get("revision")!r}')
    elif not source.get('passthrough'):
        raise ValueError(f'Census source {source["id"]!r} cannot be imported from a local path '
                         f'unless it is an authored passthrough')
    return entry


def import_suites(spec, output, provenance_path=None):
    """Write the declared splits. The operator chooses every split, never the tool."""
    if not isinstance(spec, dict) or set(spec) - {'seed', 'splits', 'dedupe'} or not {'seed', 'splits'} <= set(spec):
        raise ValueError('Import spec requires exactly seed and splits, plus optional dedupe')
    if type(spec['seed']) is not int:
        raise ValueError('Import spec seed must be an integer')
    declared = spec['splits']
    if not isinstance(declared, dict) or not declared or set(declared) - set(SPLITS):
        raise ValueError(f'splits must be a nonempty mapping using {list(SPLITS)}')
    census = load_census(provenance_path) if provenance_path else None
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Import output must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    collected, provenance = {}, []
    unlicensed = []
    for split in SPLITS:
        if split not in declared:
            continue
        sources = declared[split]
        if not isinstance(sources, list) or not sources:
            raise ValueError(f'Split {split!r} needs a nonempty source list')
        rows = []
        for source in sources:
            entry = _census_entry(census, source)
            produced, record = source_rows(source, split)
            if entry is not None:
                licence = entry['license']
                for row in produced:
                    row['source'] = source['id']
                    row['license'] = licence
                record['license'] = licence
                if licence == 'unstated':
                    unlicensed.append(source['id'])
                record['census'] = {key: entry.get(key) for key in
                                    ('dataset', 'revision', 'license', 'license_url', 'url',
                                     'languages', 'structure', 'pairing', 'referenced_by', 'vendored')}
            elif source.get('passthrough') and census is not None:
                for row in produced:
                    row['source'] = source['id']
                    row['license'] = 'authored'
            rows.extend(produced)
            provenance.append(record)
        collected[split] = rows
    dedupe = spec.get('dedupe')
    dropped_near = {}
    dropped_orphans = {}
    if dedupe is not None:
        if not isinstance(dedupe, dict) or set(dedupe) - {'jaccard', 'min_family_size'} or 'jaccard' not in dedupe:
            raise ValueError('dedupe accepts a jaccard threshold and an optional min_family_size')
        collected, dropped_near = near_duplicate_dedupe(collected, dedupe['jaccard'])
        if 'min_family_size' in dedupe:
            collected, dropped_orphans = prune_small_families(collected, dedupe['min_family_size'])
    written = {}
    for split, rows in collected.items():
        if not rows:
            raise ValueError(f'Split {split!r} is empty after dedupe')
        path = output / f'{split}.jsonl'
        atomic_text(path, ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))
        written[split] = {'path': str(path), 'tasks': len(rows), 'digest': digest(rows)}
    suites = {name: load_suite(entry['path']) for name, entry in written.items()}
    if len(suites) > 1:
        validate_splits(suites)
    report = {'status': 'imported', 'seed': spec['seed'], 'splits': written, 'sources': provenance,
              'duplicates_dropped': sum(record['duplicates_dropped'] for record in provenance),
              'near_duplicates_dropped': sum(len(v) for v in dropped_near.values()),
              'near_duplicate_detail': dropped_near,
              'orphaned_family_rows_dropped': sum(len(v) for v in dropped_orphans.values()),
              'orphaned_family_detail': dropped_orphans,
              'categories': {name: sorted({row['category'] for row in rows if row.get('category')})
                             for name, rows in collected.items()},
              'synthetic_family_sources': sum(record['family_origin'] == 'synthetic-row-group'
                                               for record in provenance),
              'not_checked': ['that a synthetic row grouping is a genuine topic family',
                              'that the imported prompts are harmful to this model',
                              'token budgets', 'model fit', 'refusal wording in the imported corpus']}
    if census is not None:
        report['census'] = {'path': census['path'], 'sha256': census['sha256'],
                            'generated': census['generated'],
                            'sources': len(census['sources'])}
        report['unlicensed_sources'] = sorted(set(unlicensed))
    atomic_json(output / 'import-provenance.json', report)
    return report


def validate_import(spec, provenance_path=None):
    """Spec-shape and local-file checks only. No downloads, no writes."""
    if not isinstance(spec, dict) or set(spec) - {'seed', 'splits', 'dedupe'} or not {'seed', 'splits'} <= set(spec):
        raise ValueError('Import spec requires exactly seed and splits, plus optional dedupe')
    if type(spec['seed']) is not int:
        raise ValueError('Import spec seed must be an integer')
    census = load_census(provenance_path) if provenance_path else None
    dedupe = spec.get('dedupe')
    if dedupe is not None and (not isinstance(dedupe, dict) or set(dedupe) - {'jaccard', 'min_family_size'}
                               or 'jaccard' not in dedupe or not 0 < dedupe['jaccard'] <= 1):
        raise ValueError('dedupe accepts a jaccard threshold in (0,1] and an optional min_family_size')
    if dedupe is not None and 'min_family_size' in dedupe and (type(dedupe['min_family_size']) is not int
                                                               or dedupe['min_family_size'] < 2):
        raise ValueError('min_family_size must be an integer of at least 2')
    declared = spec['splits']
    if not isinstance(declared, dict) or not declared or set(declared) - set(SPLITS):
        raise ValueError(f'splits must be a nonempty mapping using {list(SPLITS)}')
    checked = []
    for split, sources in declared.items():
        if not isinstance(sources, list) or not sources:
            raise ValueError(f'Split {split!r} needs a nonempty source list')
        for source in sources:
            if 'path' not in source and 'dataset' not in source:
                raise ValueError('Each source needs either a dataset or a path')
            if 'path' in source and not Path(source['path']).is_file():
                raise ValueError(f'Missing local prompt file: {source["path"]}')
            if 'dataset' in source and not isinstance(source.get('revision'), str):
                raise ValueError('Pinned dataset revision is required; a moving branch is not reproducible')
            if source.get('passthrough') and 'dataset' in source:
                raise ValueError('A passthrough source reads a local file, not a dataset')
            if 'template' in source and ('column' in source or 'list_column' in source):
                raise ValueError('Use either column/list_column or template for the prompt, not both')
            if 'column' in source and 'list_column' in source:
                raise ValueError('Use either column or list_column for the prompt, not both')
            if (not source.get('passthrough') and 'column' not in source and 'template' not in source
                    and 'list_column' not in source):
                raise ValueError('Each corpus source needs a prompt column, list_column or template')
            if 'offset' in source and (type(source['offset']) is not int or source['offset'] < 0):
                raise ValueError('offset must be a nonnegative integer')
            if 'limit' in source and (type(source['limit']) is not int or source['limit'] < 1):
                raise ValueError('limit must be a positive integer')
            if 'stride' in source and (type(source['stride']) is not int or source['stride'] < 1):
                raise ValueError('stride must be a positive integer')
            entry = _census_entry(census, source)
            checked.append({'split': split, 'id': source.get('id'), 'kind': source.get('kind'),
                            'remote': 'dataset' in source, 'passthrough': bool(source.get('passthrough')),
                            'category': 'category' in source,
                            'license': entry['license'] if entry else
                                       ('authored' if source.get('passthrough') else None)})
    return {'status': 'metadata-valid', 'splits': sorted(declared), 'sources': len(checked),
            'remote_sources': sum(entry['remote'] for entry in checked),
            'categorised_sources': sum(entry['category'] for entry in checked),
            'provenance_census': None if census is None else {
                'path': census['path'], 'sha256': census['sha256'],
                'sources': len(census['sources'])},
            'every_source_licensed': all(entry['license'] for entry in checked),
            'dedupe': dedupe, 'sources_checked': checked,
            'downloads_performed': False, 'output_created': False,
            'not_checked': ['dataset availability and file layout', 'prompt harmfulness',
                            'family size against the run consistency policy',
                            'how many near-duplicates dedupe will actually remove']}
