#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage the Gradio Space tree an account owner pushes, from this clone.

    python scripts/space_stage.py --out DIR

The Space must be buildable without pip-installing this repo: `pyproject.toml`
pins an XPU torch that a CUDA ZeroGPU image cannot take, so the staged tree is
self-contained — a vendored `jdewitt/` copy, the bank directory verbatim, a thin
`app.py`, a `requirements.txt` that asks for the four runtime libraries and no
torch, and a README whose YAML tells Spaces what to launch.

The vendor set is not the whole package: it is the AST import closure of
`app.py`, following both module-level and function-local imports inside
`jdewitt/`. That keeps CLI-only modules (and the host paths a few of them carry:
`tune.py`, `ppl.py`, `quantize.py`, `certify.py`) out of a public Space, and it
means a module the app reaches but the tree cannot supply is a hard refusal
(`MISSING_IMPORT`) rather than a crash on first click. Every vendored file is
byte-identical to the clone's file; nothing is rewritten.

Refusals, all typed and all before a single byte is written:

    MISSING_IMPORT      a file `app.py` imports (directly or through a vendored
                        module) does not exist in `jdewitt/`
    UNDECLARED_IMPORT   the tree imports a package `requirements.txt` does not
                        ship (this is what keeps torch out)
    SANITISATION        a staged text file carries `/mnt/`, `/home/`, a board
                        card id (`t_` + 8 hex) or the word the repo bans
    TOO_LARGE           the tree outside `spaces/bank/` exceeds 5 MB
    BANK_INVALID        `spaces/bank/bank.json` is not a readable bank document
    OUT_NOT_EMPTY       DIR exists and already holds files

Run it twice into two directories and `diff -r` them: the output is a pure
function of this clone's bytes.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'jdewitt'
BANK = ROOT / 'spaces' / 'bank'
BANK_SCHEMA = 'jdewitt-space-bank-v1'
MAX_OUTSIDE_BANK = 5 * 1024 * 1024
SDK_VERSION = '5.50.0'
SHORT_DESCRIPTION = 'Covered GGUF edit, labelled from measurement'
# What the Space image plus requirements.txt can import. Gradio comes from the
# Gradio/ZeroGPU image; the other three are the Space's own requirements. Torch is
# not pinned in requirements.txt (the XPU pin in pyproject.toml must never reach a
# CUDA image) but the image does carry it, and the bank-apply path needs it to load
# the pinned plan tensor; `spaces` is the ZeroGPU image's own package (it declares
# the one @spaces.GPU function the hardware requires); nothing here needs
# transformers, llama-server or a tokenizer.
SPACE_REQUIREMENTS = ('gradio', 'numpy', 'gguf', 'huggingface_hub')
IMAGE_PROVIDED = ('torch', 'spaces')
BANNED = (
    ('absolute path', re.compile(r'/mnt/|/home/')),
    ('board card id', re.compile(r't_[0-9a-f]{8}')),
    ('banned word', re.compile(r'uncensored')),
)


class StageRefused(RuntimeError):
    """The tree must not be staged; `code` names the reason."""

    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f'{code}: {message}')


# --------------------------------------------------------------------------- imports


def _module_path(module):
    """The repo file that supplies a local module name, or None."""
    parts = str(module).split('.')
    if not parts or parts[0] != PACKAGE:
        return None
    base = ROOT.joinpath(PACKAGE)
    if len(parts) == 1:
        init = base / '__init__.py'
        return init if init.is_file() else None
    target = base.joinpath(*parts[1:])
    if target.with_suffix('.py').is_file():
        return target.with_suffix('.py')
    init = target / '__init__.py'
    return init if init.is_file() else None


def _imports(tree, module, is_package):
    """Local module names one parsed file imports, module-level or nested.

    Returns `(strict, candidates)`: `strict` names come from `import a.b.c` and
    must exist as modules; `candidates` come from `from a import b` and may be
    an attribute of `a` rather than a module.
    """
    parts = module.split('.')
    package = parts if is_package else parts[:-1]
    strict, candidates = set(), set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PACKAGE or alias.name.startswith(PACKAGE + '.'):
                    strict.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = package[:len(package) - (node.level - 1)]
                if node.module:
                    base = '.'.join(anchor + [node.module])
                    candidates.add(base)
                    candidates.update(f'{base}.{alias.name}' for alias in node.names)
                else:
                    candidates.update('.'.join(anchor + [alias.name]) for alias in node.names)
            elif node.module and (node.module == PACKAGE or node.module.startswith(PACKAGE + '.')):
                candidates.add(node.module)
                candidates.update(f'{node.module}.{alias.name}' for alias in node.names)
    return strict, candidates


def _external_imports(tree):
    """Top-level import names that are not part of this package."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names.add(node.module.split('.')[0])
    return {name for name in names if name != PACKAGE}


def _closure(app_source):
    """{module: path} for every local module `app.py` can reach, or MISSING_IMPORT."""
    seen = {}
    strict, candidates = _imports(ast.parse(app_source), 'app', False)
    pending = [(name, 'strict') for name in strict]
    pending += [(name, 'candidate') for name in candidates]
    while pending:
        name, kind = pending.pop()
        path = _module_path(name)
        if path is None:
            parent = _module_path(name.rsplit('.', 1)[0]) if '.' in name else None
            if kind == 'candidate' and parent is not None and parent.name != '__init__.py':
                continue  # an attribute of a resolved module, not a file
            raise StageRefused(
                'MISSING_IMPORT',
                f'app.py reaches {name!r} but no file in jdewitt/ supplies it '
                f'(looked for {PACKAGE}/{name.replace(".", "/")}.py and '
                f'{PACKAGE}/{name.replace(".", "/")}/__init__.py)')
        if path in seen:
            continue
        seen[path] = name
        local_strict, local_candidates = _imports(ast.parse(path.read_text()), name, path.name == '__init__.py')
        pending += [(item, 'strict') for item in local_strict]
        pending += [(item, 'candidate') for item in local_candidates]
    return {name: path for path, name in seen.items()}


def _check_external(names, origin):
    """Every import the Space will execute must be in the image or requirements."""
    allowed = set(sys.stdlib_module_names) | set(SPACE_REQUIREMENTS) | set(IMAGE_PROVIDED)
    missing = sorted(name for name in names if name not in allowed)
    if missing:
        raise StageRefused(
            'UNDECLARED_IMPORT',
            f'{origin} imports {", ".join(missing)}, which requirements.txt does not ship '
            f'and the ZeroGPU image does not guarantee; the Space requirements are '
            f'{", ".join(SPACE_REQUIREMENTS)} (the image also carries '
            f'{", ".join(IMAGE_PROVIDED)})')


# --------------------------------------------------------------------------- content


def _bank_document():
    """`spaces/bank/bank.json`, shape-checked just enough to build the README."""
    path = BANK / 'bank.json'
    if not path.is_file():
        raise StageRefused('BANK_INVALID', f'{path} is missing; the Space ships the bank verbatim')
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise StageRefused('BANK_INVALID', f'{path} cannot be read as JSON: {error}')
    if not isinstance(document, dict) or document.get('schema') != BANK_SCHEMA:
        raise StageRefused('BANK_INVALID', f'{path} is not a {BANK_SCHEMA} document')
    entries = document.get('entries')
    if not isinstance(entries, list):
        raise StageRefused('BANK_INVALID', f'{path}: entries must be a list')
    for entry in entries:
        base = entry.get('base') if isinstance(entry, dict) else None
        if not isinstance(base, dict) or not isinstance(base.get('repo'), str) or not base['repo']:
            raise StageRefused('BANK_INVALID', f'{path}: every entry needs base.repo')
    return document


OAUTH_FRONTMATTER = """hf_oauth: true
hf_oauth_scopes: [write-repos]
"""

OAUTH_BODY = """
## Publishing to your own account (v1.1, behind a flag)

The direct download is the delivery by default. With `JW_OAUTH_PUSH=1` set in the
Space's variables, the page also offers a **Publish** checkbox: sign in with the
Hub account that should own the file, and the app creates
`jdewitt-<base>-<quant>` under that account and uploads the edited GGUF, the
generated card, the sidecar and `SHA256SUMS` (plus a `.revert.npz` sidecar only
when one exists and is under a 1 GiB cap). The result line carries the repo URL
and the commit sha.

What is promised while it uploads: the bytes shown are the bytes this process
moved (read from disk and written to the Hub's signed upload URL), a rate over
the last 15 seconds, and a remaining time computed as `remaining / rate`. One
upload runs at a time, across worker processes. The repo is created **private**,
made public only when the upload and the commit both succeed, and deleted —
then re-read from the API to prove it is gone — if the upload fails or the
visitor presses **Cancel upload**. The Space only ever held a temporary copy;
the copy that survives is the visitor's.

The flag is off on any deployment whose request window cannot carry the largest
file, and the card in `verification/space/oauth-upload.json` carries the measured
verdict rather than a promise.
"""


def _readme(document, oauth_push=False):
    bases = sorted({str(entry['base']['repo']) for entry in document['entries']})
    models = 'models: []' if not bases else 'models:\n' + ''.join(f'  - {base}\n' for base in bases).rstrip('\n')
    oauth = OAUTH_FRONTMATTER if oauth_push else ''
    body = OAUTH_BODY if oauth_push else ''
    return f"""---
title: Banked GGUF edit
sdk: gradio
sdk_version: "{SDK_VERSION}"
app_file: app.py
short_description: {SHORT_DESCRIPTION}
{models}
{oauth}---

# Banked GGUF edit

This Space takes one GGUF. It reads the file's header, checks the model card's
licence, and — when the direction bank covers that architecture and quant —
applies the bank's measured direction to a copy and hands back the edited file
with its card, its sidecar and `SHA256SUMS`.

The result carries one of three labels, generated from the bank entry's
`measurement` and the file in hand:

- `measured_artifact_class` — this exact repo@revision and quant were measured;
- `banked_unmeasured` — the file is covered by its geometry but was not measured;
  the card says so in its first line and claims codec fidelity only;
- `refused` — one typed code naming the reason and what would cover it.

No inference runs here and no token is generated: the requirements carry no
transformers, no llama-server and no tokenizer. Nothing is uploaded to us — your
file lives only in the Space's ephemeral disk, the job workspace is deleted after
the response, and a gated repository is refused before any download.

`jdewitt/` is a vendored copy of this repository's package, limited to the modules
`app.py` imports; `spaces/bank/` is the bank directory verbatim.
""" + body


def _app_source():
    """`spaces/app.py` is the source of truth; the stage ships it verbatim."""
    return APP_PATH.read_text(encoding='utf-8')


APP_PATH = ROOT / 'spaces' / 'app.py'

REQUIREMENTS = """numpy>=2
gguf>=0.19
huggingface_hub>=0.34
gradio>=5,<6
"""
OAUTH_REQUIREMENTS = """numpy>=2
gguf>=0.19
huggingface_hub>=0.34
gradio[oauth]>=5,<6
"""

GITIGNORE = """__pycache__/
*.py[cod]
.venv/
"""


def _tree_files(oauth_push=False):
    """{relative path: bytes} for everything the Space tree carries."""
    document = _bank_document()
    app = _app_source().encode()
    files = {
        'README.md': _readme(document, oauth_push).encode(),
        'app.py': app,
        'requirements.txt': (OAUTH_REQUIREMENTS if oauth_push else REQUIREMENTS).encode(),
        '.gitignore': GITIGNORE.encode(),
    }
    app_source = app.decode()
    external = _external_imports(ast.parse(app_source))
    for module, path in sorted(_closure(app_source).items()):
        relative = Path(PACKAGE) / path.relative_to(ROOT / PACKAGE)
        files[relative.as_posix()] = path.read_bytes()
        external |= _external_imports(ast.parse(path.read_text()))
    # The package marker is not reachable through an import statement, but a
    # directory without it is only a namespace portion: the clone's regular
    # `jdewitt` package would shadow the staged copy.
    files[f'{PACKAGE}/__init__.py'] = (ROOT / PACKAGE / '__init__.py').read_bytes()
    _check_external(external, 'the staged app')
    # The vendored package is AGPL; a public Space is a distribution, so the
    # licence text travels with it.
    for name in ('LICENSE', 'NOTICE'):
        if (ROOT / name).is_file():
            files[name] = (ROOT / name).read_bytes()
    for path in sorted(item for item in BANK.rglob('*') if item.is_file()):
        files[('spaces/bank/' + path.relative_to(BANK).as_posix())] = path.read_bytes()
    return files


def _is_text(data):
    if b'\x00' in data:
        return False
    try:
        data.decode('utf-8')
    except UnicodeDecodeError:
        return False
    return True


def _check_sanitised(files):
    """No host path, board id or banned word may travel in a staged text file."""
    hits = []
    for name in sorted(files):
        data = files[name]
        if not _is_text(data):
            continue
        text = data.decode('utf-8')
        for label, pattern in BANNED:
            for match in pattern.finditer(text):
                line = text.count('\n', 0, match.start()) + 1
                hits.append(f'{name}:{line}: {label} ({match.group(0)!r})')
    if hits:
        shown = '; '.join(hits[:6]) + (f' and {len(hits) - 6} more' if len(hits) > 6 else '')
        raise StageRefused(
            'SANITISATION',
            f'a staged text file carries what must not leave this clone: {shown}')
    return True


def _outside_bank(files):
    return sum(len(data) for name, data in files.items() if not name.startswith('spaces/bank/'))


def build(oauth_push=False):
    """Assemble, check and return (files, checks) without touching the filesystem."""
    files = _tree_files(oauth_push)
    outside = _outside_bank(files)
    if outside > MAX_OUTSIDE_BANK:
        raise StageRefused(
            'TOO_LARGE',
            f'the tree outside spaces/bank/ is {outside} bytes, over the {MAX_OUTSIDE_BANK}-byte cap')
    checks = {
        'sanitised': _check_sanitised(files),
        'app_parses': bool(ast.parse(files['app.py'].decode())),
        'imports_resolved': True,  # `_closure` refused above if any local import was missing
        'size_outside_bank_bytes': outside,
        'oauth_push': bool(oauth_push),
    }
    return files, checks


def stage(out, oauth_push=False):
    """Write the tree under `out` (which must be absent or empty) and report it."""
    out = Path(out)
    if out.exists():
        if not out.is_dir():
            raise StageRefused('OUT_NOT_DIR', f'{out} exists and is not a directory')
        if any(out.iterdir()):
            raise StageRefused('OUT_NOT_EMPTY', f'{out} is not empty; stage into a fresh directory')
    files, checks = build(oauth_push)
    for name in sorted(files):
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[name])
    report = {
        'schema': 'jdewitt-space-stage-v1',
        'out': str(out),
        'files': [{'path': name, 'bytes': len(files[name])} for name in sorted(files)],
        'bytes': sum(len(data) for data in files.values()),
        'checks': checks,
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Stage the Gradio Space tree (README, app, requirements, vendored jdewitt, bank).')
    parser.add_argument('--out', type=Path, required=True, help='directory to write the Space tree into')
    parser.add_argument('--oauth-push', action='store_true',
                        help='add hf_oauth: true / hf_oauth_scopes: [write-repos] and gradio[oauth] '
                             '(v1.1 delivery; off unless the measured 5.5 GB window fits)')
    args = parser.parse_args(argv)
    try:
        report = stage(args.out, oauth_push=args.oauth_push)
    except StageRefused as refusal:
        print(json.dumps({'schema': 'jdewitt-space-stage-v1', 'error': {
            'code': refusal.code, 'message': refusal.message}}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
