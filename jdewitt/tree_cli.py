# SPDX-License-Identifier: AGPL-3.0-or-later
"""Resumable experimental tree screening, kept separate from certified exports."""
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import re
import time

from .answers import AnswerStore, atomic_json, exclusive
from .data import digest
from .storage import file_digest
from .tree_config import read_tree_config, validate_model_identity


def _upstream_lens(path, identity, expected=None):
    import torch
    try:
        # Memory-map the fitted lens, not model weights. Do not eagerly copy a
        # whole multi-GB lens into RAM just to select several checkpoints.
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except (pickle.UnpicklingError, EOFError, TypeError) as error:
        raise ValueError('Invalid safe upstream Jacobian lens tensor file') from error
    if (not isinstance(payload, dict) or set(payload) != {'J', 'n_prompts', 'source_layers', 'd_model'}
            or type(payload['d_model']) is not int or payload['d_model'] != identity['hidden_size']):
        raise ValueError('Upstream Jacobian lens width/format mismatch')
    matrices, layers = payload['J'], payload['source_layers']
    d = identity['hidden_size']
    if (not isinstance(matrices, dict) or not matrices or not isinstance(layers, list)
            or layers != sorted(matrices) or len(set(layers)) != len(layers)
            or any(type(i) is not int or not 0 <= i < identity['layers'] for i in layers)
            or type(payload['n_prompts']) is not int or payload['n_prompts'] < 1
            or any(not isinstance(j, torch.Tensor) or j.shape != (d, d) or not j.is_floating_point()
                   for j in matrices.values())):
        raise ValueError('Upstream Jacobian lens tensor/checkpoint metadata mismatch')
    if expected is not None and (layers != expected['source_layers'] or payload['n_prompts'] != expected['n_prompts']):
        raise ValueError('Upstream lens content does not match its bound manifest')
    return payload


def bind_lens(lens_file, model_identity, output):
    """Bind an upstream lens to an OPERATOR-ATTESTED model, not a proven origin."""
    validate_model_identity(model_identity)
    lens_file, output = Path(lens_file).resolve(), Path(output).resolve()
    if output.exists() or output == lens_file:
        raise FileExistsError('Use a new manifest path; existing provenance is never overwritten')
    before = file_digest(lens_file)
    payload = _upstream_lens(lens_file, model_identity)
    import torch
    if any(not bool(torch.isfinite(j).all()) for j in payload['J'].values()):
        raise ValueError('Nonfinite upstream Jacobian lens tensor')
    if file_digest(lens_file) != before:
        raise ValueError('Lens changed while binding provenance')
    manifest = {'format': 'upstream-jlens-v1', 'lens_file': str(lens_file), 'sha256': before,
                'model_identity': model_identity, 'source_layers': payload['source_layers'],
                'n_prompts': payload['n_prompts']}
    atomic_json(output, manifest)
    return {'manifest': str(output), 'sha256': before, 'model_origin': 'operator-attested',
            'note': 'Upstream lens files contain no model ID. Verify their actual fitting provenance yourself.'}


def load_action_bank(path, model_identity, checkpoints):
    """Portable direction evidence only. This never loads/offloads model weights."""
    import torch
    from .tree_search import Action, _basis_ok
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (not isinstance(payload, dict) or set(payload) != {'model_identity', 'actions'}
            or payload['model_identity'] != model_identity or not isinstance(payload['actions'], dict)
            or not payload['actions']):
        raise ValueError('Learned tree direction bank model identity/format mismatch')
    result = {}
    for name, bases in payload['actions'].items():
        if (not isinstance(name, str) or not name.strip() or name == 'none'
                or not isinstance(bases, dict) or set(bases) != set(checkpoints)):
            raise ValueError('Learned tree direction bank checkpoint/name mismatch')
        for basis in bases.values():
            _basis_ok(basis, model_identity['hidden_size'], torch.device('cpu'))
        result[name] = Action(name, bases)
    return result


def _seal(path, identity, payload):
    record = {'identity': identity, 'payload': payload}
    record['record_digest'] = digest(record)
    atomic_json(path, record)


def _read_sealed(path, identity):
    record = json.loads(Path(path).read_text())
    if (not isinstance(record, dict) or set(record) != {'identity', 'payload', 'record_digest'}
            or record['identity'] != identity
            or digest({k: v for k, v in record.items() if k != 'record_digest'}) != record['record_digest']):
        raise ValueError('Tree record integrity/identity mismatch')
    return record['payload']


def _identity(raw, paths, manifest, suite):
    source = Path(raw['model_identity']['model'])
    model_files = None
    if source.is_dir():
        model_files = {p.name: file_digest(p) for p in sorted(source.iterdir()) if p.is_file()
                       and not p.name.startswith('.') and p.suffix in {'.safetensors', '.json', '.jinja', '.py', '.model', '.txt'}}
        if not model_files or not any(k.endswith('.safetensors') for k in model_files):
            raise ValueError('Tree local model source requires original safetensors and metadata')
    return {'version': 1, 'config': raw, 'suite_digest': digest(suite), 'lens_sha256': manifest['sha256'],
            'direction_bank_sha256': file_digest(paths['direction_bank']) if 'direction_bank' in paths else None,
            'behavior_ranker_sha256': file_digest(paths['behavior_artifact']) if 'behavior_artifact' in paths else None,
            'lens_manifest_sha256': file_digest(paths['lens_manifest']), 'local_source': model_files,
            'code': {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'dependencies': {name: version(name) for name in ('torch', 'transformers', 'numpy', 'safetensors')},
            'scope': 'prefill-last-position-only; runtime trajectory, not a static weight edit'}


def _model_factory(raw):
    from .hf import HFModel
    from .pipeline import configure_compiler_paths
    model = raw['model_identity']
    if not Path(model['model']).is_dir() and re.fullmatch('[0-9a-f]{40}', model['revision']) is None:
        raise ValueError('Remote tree models require an immutable full revision SHA')
    configure_compiler_paths()
    return HFModel(model['model'], model['revision'], rank=1, max_tokens=raw['max_tokens'],
                   dtype=raw['dtype'], ctx_size=raw['ctx_size'], trust_remote_code=False,
                   quantization='none', trial_operator='merged-resident')


def _check_model(hf, identity):
    config = getattr(hf.config, 'text_config', hf.config)
    if (hf.config.model_type != identity['architecture'] or config.hidden_size != identity['hidden_size']
            or config.num_hidden_layers != identity['layers']):
        raise ValueError('Loaded model differs from the declared lens model identity')


def _make_readout(hf, raw, settings, payload, artifact_path=None, lens_sha256=None):
    import torch
    from .tree_search import JSpaceReadout
    # Locate the same explicit backbone without preparing/forwarding a prompt.
    from .tree_config import SEGMENT_PATHS
    root = hf.model
    for part in SEGMENT_PATHS[raw['model_identity']['architecture']].split('.'):
        root = getattr(root, part)
    d = raw['model_identity']['hidden_size']
    jacobians = {cp: payload['J'][cp].to(device=hf.device, dtype=torch.float32)
                 for cp in settings.checkpoints[:-1]}
    jacobians[settings.checkpoints[-1]] = torch.eye(d, device=hf.device, dtype=torch.float32)
    head = hf.model.get_output_embeddings()
    if head is None or head.bias is not None:
        raise ValueError('Tree readout requires a bias-free explicit unembedding adapter')
    if head.weight.shape[1] != d or head.weight.device != hf.device:
        raise ValueError('Tree readout head differs from the lens residual space/device')
    readout = JSpaceReadout(jacobians, root.norm, head.weight, raw['refusal_tokens'], raw['reference_tokens'],
                           raw['protected_tokens'], max_protected_kl=raw['max_protected_kl'],
                           max_norm_change=raw['max_norm_change'], drift_weight=raw['drift_weight'])
    if 'behavior' in raw:
        from .tree_behavior import TrajectoryReadout
        if lens_sha256 is None:
            raise ValueError('Behavior features require the actual fitted-lens content identity')
        readout = TrajectoryReadout(readout, raw, lens_sha256, artifact_path, runtime_fingerprint=hf.fingerprint)
    return readout


def _tree_record(path, identity, task, finalists):
    payload = _read_sealed(path, identity)
    if (not isinstance(payload, dict) or payload.get('id') != task['id'] or payload.get('prompt') != task['prompt']
            or not isinstance(payload.get('result'), dict)
            or len(payload['result'].get('finalists', [])) != finalists):
        raise ValueError('Saved tree task/finalist integrity mismatch')
    return payload


def run_tree(path, model_factory=None):
    """Only this command owns B70 inference; validate-tree is strictly metadata-only.

    A model_factory is injectable solely for labeled scheduling fixtures. The
    production factory always requires B70 and never substitutes CPU inference.
    """
    raw, paths, settings, manifest, lens_file, suite, _, _ = read_tree_config(path)
    collect = raw.get('behavior', {}).get('mode') == 'collect'
    identity = _identity(raw, paths, manifest, suite)
    output = paths['output']
    with exclusive(output, '.tree.lock'):
        stores = {'baseline': AnswerStore(output / 'answers' / 'baseline', suite, {**identity, 'variant': 'baseline'})}
        stores.update({f'rank-{i}': AnswerStore(output / 'answers' / f'rank-{i}', suite,
                       {**identity, 'variant': f'proxy-rank-{i}'}) for i in range(settings.finalists)})
        from .tree_behavior import collection_complete, collection_evidence
        saved, missing, reused = {}, [], 0
        for task in suite:
            trace_path = output / 'trees' / (digest(task['id']) + '.json')
            responses = {name: store.get(task) for name, store in stores.items()}
            reused += sum(r is not None for r in responses.values())
            if trace_path.exists():
                saved[task['id']] = _tree_record(trace_path, identity, task, settings.finalists)
            elif any(r is not None for r in responses.values()):
                raise ValueError('Saved tree answers are missing their bound search trace')
            if (not trace_path.exists() or any(r is None for r in responses.values())
                    or (collect and not collection_complete(output, task, identity))):
                missing.append(task)
        if (output / 'report.json').exists():
            previous = _read_sealed(output / 'report.json', identity)
            runtime = output / 'runtime.json'
            if not runtime.is_file():
                raise ValueError('Completed tree runtime provenance integrity is missing')
            _read_sealed(runtime, identity)
            if missing:
                raise ValueError('Completed tree report has missing answer/trace evidence')
            evidence = {str(p.relative_to(output)): file_digest(p) for p in sorted((output / 'trees').glob('*.json'))}
            if previous.get('tree_evidence') != evidence:
                raise ValueError('Completed tree trace evidence integrity mismatch')
            if collect and previous.get('behavior_evidence') != collection_evidence(output):
                raise ValueError('Completed behavior evidence integrity mismatch')
        generated, behavior_generated = 0, 0
        if missing:
            # Validate actual fitted tensors before loading any model weights.
            payload = _upstream_lens(lens_file, raw['model_identity'], manifest)
            if file_digest(lens_file) != manifest['sha256']:
                raise ValueError('Tree lens changed while loading')
            import torch
            from . import tree_search
            from .tree_hf import HFSegments, PrefillReplay
            bank = load_action_bank(paths['direction_bank'], raw['model_identity'], settings.checkpoints[:-1]) \
                if 'direction_bank' in paths else {}
            if 'direction_bank' in paths and file_digest(paths['direction_bank']) != identity['direction_bank_sha256']:
                raise ValueError('Learned direction bank changed while loading')
            hf = (model_factory or _model_factory)(raw)
            try:
                _check_model(hf, raw['model_identity'])
                hf.preflight(suite)
                fingerprint_path = output / 'runtime.json'
                if fingerprint_path.exists():
                    if _read_sealed(fingerprint_path, identity) != hf.fingerprint:
                        raise ValueError('Tree tokenizer/generation/runtime fingerprint changed')
                else:
                    _seal(fingerprint_path, identity, hf.fingerprint)
                with torch.inference_mode():
                    readout = _make_readout(hf, raw, settings, payload, paths.get('behavior_artifact'), manifest['sha256'])
                    if ('behavior_artifact' in paths
                            and file_digest(paths['behavior_artifact']) != identity['behavior_ranker_sha256']):
                        raise ValueError('Behavior ranker changed while loading')
                    del payload
                    actions = []
                    for name, spec in raw['actions'].items():
                        if isinstance(spec, list):
                            actions.append(readout.pullback_action(name, spec, settings.checkpoints[:-1]))
                        else:
                            if spec['bank'] not in bank:
                                raise ValueError('Requested learned tree direction is absent from the bank')
                            actions.append(tree_search.Action(name, {cp: basis.to(device=hf.device, dtype=torch.float32)
                                           for cp, basis in bank[spec['bank']].bases.items()}))
                    del bank
                    for task in missing:
                        trace_path = output / 'trees' / (digest(task['id']) + '.json')
                        inputs, _ = hf.encode(task['prompt'])
                        prompt_length = inputs['input_ids'].shape[1]
                        record = saved.get(task['id'])
                        if record is None:
                            if hf.device.type == 'xpu':
                                torch.xpu.synchronize()
                            start = time.perf_counter()
                            backend = HFSegments(hf, inputs)
                            result = tree_search.search_tree(backend, readout, actions, settings)
                            if len(result.finalists) != settings.finalists:
                                raise ValueError('Not enough admissible finalists; no gate was relaxed')
                            if hf.device.type == 'xpu':
                                torch.xpu.synchronize()
                            record = {'id': task['id'], 'prompt': task['prompt'], 'prompt_length': prompt_length,
                                      'search_seconds': time.perf_counter() - start, 'result': asdict(result)}
                            _seal(trace_path, identity, record)
                            del backend, result
                        if record['prompt_length'] != prompt_length:
                            raise ValueError('Resumed tree prompt tokenization changed')
                        # Baseline and finalists have identical output budgets/floors.
                        floor = raw['refusal_floor'] if task['kind'] == 'refusal' else None
                        variants = [('baseline', [])] + [(f'rank-{i}', item['path'])
                                    for i, item in enumerate(record['result']['finalists'])]
                        for name, trajectory in variants:
                            if stores[name].get(task) is not None:
                                continue
                            with PrefillReplay(hf.editor.layers, actions, trajectory, prompt_length):
                                response = hf.generate(task['prompt'], min_tokens=floor)
                            if floor is not None and response.get('floor_enforced') is not True:
                                raise ValueError('Tree decoder did not enforce the requested refusal floor')
                            stores[name].put(task, response)
                            generated += 1
                            atomic_json(output / 'progress.json', {'stage': 'tree-screening',
                                'generated_this_attempt': generated, 'reused': reused, 'task_id': task['id'], 'variant': name})
                        if collect:
                            from .tree_behavior import collect_branches
                            known = [(trajectory, stores[name].get(task)) for name, trajectory in variants]
                            collection = collect_branches(hf, task, record, actions, raw, output, identity, known)
                            behavior_generated += collection['generated']
                        del inputs, record
                    # Drop references before HFModel.release so no model-owned
                    # norm/head parameters survive target teardown through readout.
                    del readout, actions
            finally:
                hf.release()
        result = {'status': 'experimental-unvalidated', 'scope': 'prefill-only runtime interventions; no static GGUF export',
                  'generated': generated, 'reused': reused, 'prompts': len(suite),
                  'behavior_generated': behavior_generated,
                  'behavior_evidence': collection_evidence(output) if collect else None,
                  'answers': {name: store.status() for name, store in stores.items()},
                  'tree_evidence': {str(p.relative_to(output)): file_digest(p)
                                    for p in sorted((output / 'trees').glob('*.json'))},
                  'not_checked': ['proxy calibration on edited states', 'complete-answer fulfillment',
                                  'code/math preservation', 'matched compute advantage']}
        if any(state['missing'] for state in result['answers'].values()):
            raise ValueError('Tree job cannot complete with missing answers')
        if collect and any(not collection_complete(output, task, identity) for task in suite):
            raise ValueError('Tree job cannot complete with missing behavior collection')
        _seal(output / 'report.json', identity, result)
        atomic_json(output / 'progress.json', {'stage': 'tree-screening-complete', **result})
        return result
