# SPDX-License-Identifier: AGPL-3.0-or-later
"""Evidence-gated preset profiles.

A preset is a named bundle of tuned run fields for a model family. Presets that
carry no evidence reference are refused at load time: an unmeasured profile is
guessing dressed as a default, and this tool's contract is that tuned values
point at measurement.
"""
import json
import re
from pathlib import Path


def preset_directory():
    return Path(__file__).parent / 'presets'


def available_presets():
    out = []
    for path in sorted(preset_directory().glob('*.json')):
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        out.append({'name': path.stem, 'evidence': document.get('evidence')})
    return out


def load_preset(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('Preset names must be safe single path segments')
    path = preset_directory() / f'{name}.json'
    if not path.is_file():
        raise FileNotFoundError(f'No preset {name!r}; available: {[p["name"] for p in available_presets()]}')
    document = json.loads(path.read_text())
    evidence = document.get('evidence')
    evidence_ok = (isinstance(evidence, str) and evidence.strip()) or \
        (isinstance(evidence, list) and evidence and all(isinstance(e, str) and e.strip() for e in evidence))
    if not evidence_ok:
        raise ValueError(f'Preset {name!r} has no evidence reference; presets without measured evidence are refused')
    settings = document.get('settings')
    if not isinstance(settings, dict) or not settings:
        raise ValueError(f'Preset {name!r} needs a nonempty settings object')
    return {'name': name, 'evidence': evidence, 'settings': settings}


def apply_preset(obj, name):
    """Merge preset settings under an explicit configuration; the config always wins."""
    settings = load_preset(name)['settings']
    return {**settings, **obj}
