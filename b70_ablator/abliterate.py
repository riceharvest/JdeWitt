# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-command flow: build a validated auto-search configuration and run it.

abliterate MODEL --suites DIR --policy {standard,strict} --output DIR
    [--revision R] [--license L] [--total-trials N] [--rank K]
    [--preset NAME] [--defaults FILE] [--set FIELD=VALUE ...] [--dry-run]

Suites directory must contain directions.jsonl, selection.jsonl and
holdout.jsonl. Tool paths (converter, quantizer, llama-server, python,
sandbox image) and any run field not derivable here come from --defaults
JSON, --set overrides or an existing preset; nothing is guessed. The
consistency policy is the gate discipline, not a model-specific tuning,
so policies are shipped; model-family tunables live in evidence-gated
presets only.
"""
import json
from dataclasses import asdict
from pathlib import Path

from .auto import RunConfig
from .methods import DIRECTION_METHODS
from .presets import apply_preset


POLICIES = {
    'standard': {'min_direction_examples': 2, 'min_direction_families': 2,
                 'min_direction_overlap': 0.5, 'min_refusal_families': 2,
                 'min_variants_per_family': 2, 'confidence': 0.9, 'max_task_regressions': 0},
    'strict': {'min_direction_examples': 3, 'min_direction_families': 4,
               'min_direction_overlap': 0.6, 'min_refusal_families': 3,
               'min_variants_per_family': 3, 'confidence': 0.95, 'max_task_regressions': 0},
}

# Starting method_options per method; method-specific tuning belongs to
# evidence-gated presets and --set, these are only what validation requires.
DEFAULT_OPTIONS = {'whitened-svd': {'ridge': 1e-4}, 'leace': {'ridge': 1e-4},
                   'som': {'som_steps': 100, 'som_learning_rate': 0.05},
                   'nullspace': {'nullspace_deflations': 3}}
DEFAULT_RANK = {'leace': 1}


def build_config(model, suites, output, policy, total_trials, rank,
                 revision, license_id, defaults=None, preset=None, overrides=()):
    if policy not in POLICIES:
        raise ValueError(f'Unknown consistency policy {policy!r}; available: {sorted(POLICIES)}')
    if total_trials is None:
        total_trials = 4 * len(DIRECTION_METHODS)
    if type(total_trials) is not int or total_trials < len(DIRECTION_METHODS):
        raise ValueError(f'total_trials must be an integer >= {len(DIRECTION_METHODS)} (one per candidate method)')
    if type(rank) is not int or rank < 1:
        raise ValueError('rank must be a positive integer')
    model_dir, suites_dir, output_dir = Path(model), Path(suites), Path(output)
    if not model_dir.is_dir():
        raise FileNotFoundError(f'Model directory not found: {model_dir}')
    suites = {}
    for split in ('directions', 'selection', 'holdout'):
        path = suites_dir / f'{split}.jsonl'
        if not path.is_file():
            raise FileNotFoundError(f'Suites directory needs {split}.jsonl: {path}')
        suites[split] = str(path.resolve())
    run = {'model': str(model_dir.resolve()), 'revision': revision, 'model_license': license_id,
           **suites, 'output': str(output_dir / 'abliterated'),
           'consistency': dict(POLICIES[policy])}
    if defaults is not None:
        loaded = json.loads(Path(defaults).read_text())
        if not isinstance(loaded, dict):
            raise ValueError('--defaults must be a JSON object of run fields')
        run = {**loaded, **run}
    if preset is not None:
        run = apply_preset(run, preset)
    # Fill defaults from a RunConfig template so every field exists before
    # candidate expansion, then let validation re-check the merged result.
    template = RunConfig(model='', revision='', model_license='', directions='', selection='', holdout='',
                         max_tokens=1024, ctx_size=4096, trials=4, startup_trials=1, finalists=2, rank=rank,
                         seed=42, max_full_reply_kl=0.1, max_capability_drop=0.0, vram_headroom_gib=2.0,
                         converter='', quantizer='', converter_python='', llama_server='',
                         sandbox_image='python:3.12-slim', output='')
    merged = {**asdict(template), **run}
    methods = sorted(DIRECTION_METHODS)
    share, remainder = divmod(total_trials, len(methods))
    candidates = [{'id': method, 'method': method, 'rank': DEFAULT_RANK.get(method, rank),
                    'method_options': dict(DEFAULT_OPTIONS.get(method, {})),
                    'trials': share + (1 if i < remainder else 0)}
                  for i, method in enumerate(methods)]
    if overrides:
        from .overrides import apply_overrides
        document = apply_overrides({'run': merged, 'candidates': candidates, 'total_trials': total_trials},
                                   overrides)
        merged, candidates, total_trials = document['run'], document['candidates'], document['total_trials']
    return {'run': merged, 'candidates': candidates, 'total_trials': total_trials}
