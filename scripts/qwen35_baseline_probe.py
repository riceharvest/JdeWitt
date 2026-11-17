"""Real Qwen3.5-0.8B baseline screen. B70 only; no ablation or learned scorer.

Reuses existing generation budgets, reads development data only, and persists raw
answers before wording screening. No missing refusal wording is called compliance.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch

from jdewitt.answers import AnswerStore, atomic_json, portable_response
from jdewitt.data import digest, load_suite
from jdewitt.hf import HFModel
from jdewitt.pipeline import configure_compiler_paths
from jdewitt.refusal_check import RefusalChecker
from jdewitt.scoring import extract_answer, degenerate_output
from jdewitt.storage import file_digest
from jdewitt.tree_hf import HFSegments
from jdewitt.tree_cli import bind_lens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--lens', type=Path, required=True)
    parser.add_argument('--budget-config', type=Path, required=True)
    parser.add_argument('--authored-selection', type=Path, required=True)
    parser.add_argument('--public-selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    budget = json.loads(args.budget_config.read_text())
    authored = load_suite(args.authored_selection)
    public = load_suite(args.public_selection)
    # One real existing development prompt per explicitly labelled category.
    # These are a bounded baseline screen, not a new authored benchmark/holdout.
    categories = {}
    for task in public:
        if task['kind'] == 'refusal' and task.get('category'):
            categories.setdefault(task['category'], task)
    suite = authored + [categories[key] for key in sorted(categories)]
    if len({row['id'] for row in suite}) != len(suite):
        raise ValueError('Development probe task IDs collide')
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'suite.json', suite)
    identity = {'model': args.model, 'revision': args.revision, 'lens_sha256': file_digest(args.lens),
                'max_tokens': budget['max_tokens'], 'ctx_size': budget['ctx_size'],
                'refusal_floor': budget['refusal_floor'], 'dtype': budget['dtype'], 'suite_digest': digest(suite),
                'probe_source_sha256': file_digest(Path(__file__)),
                'hf_source_sha256': file_digest(Path(__file__).parents[1] / 'jdewitt/hf.py')}
    store = AnswerStore(args.output / 'answers', suite, identity)
    configure_compiler_paths()
    started = time.perf_counter()
    target = HFModel(args.model, args.revision, rank=1, max_tokens=budget['max_tokens'],
                     ctx_size=budget['ctx_size'], dtype=budget['dtype'], quantization='none',
                     trust_remote_code=False, trial_operator='merged-resident')
    try:
        preflight = target.preflight(suite)
        config = getattr(target.config, 'text_config', target.config)
        model_identity = {'model': args.model, 'revision': args.revision, 'architecture': target.config.model_type,
                          'layers': config.num_hidden_layers, 'hidden_size': config.hidden_size}
        manifest_path = args.output / 'lens-manifest.json'
        if not manifest_path.exists():
            bind_lens(args.lens, model_identity, manifest_path)
        lens_manifest = json.loads(manifest_path.read_text())
        if lens_manifest['model_identity'] != model_identity or lens_manifest['sha256'] != identity['lens_sha256']:
            raise ValueError('Saved lens manifest differs from the pinned model/lens')
        payload = torch.load(args.lens, map_location='cpu', weights_only=True, mmap=True)
        if payload['d_model'] != config.hidden_size or set(payload['source_layers']) != set(range(config.num_hidden_layers - 1)):
            raise ValueError('Published lens does not cover the target nonterminal checkpoints')
        print(json.dumps({'stage': 'model-loaded', 'device': str(target.device), 'identity': model_identity,
                          'lens_prompts': payload['n_prompts'], 'tasks': len(suite),
                          'kinds': dict(Counter(t['kind'] for t in suite))}), flush=True)
        del payload
        warmup = target.generate('Reply with exactly the word READY.')
        atomic_json(args.output / 'warmup.json', portable_response(warmup))
        if 'READY' not in warmup['content'] or warmup['finish_reason'] != 'stop':
            raise ValueError('Target failed the basic generation preflight; inspect saved warmup')
        del warmup
        with torch.inference_mode():
            inputs, _ = target.encode('Reply with exactly the word READY.')
            backend = HFSegments(target, inputs)
            initial = backend.prepare()
            hidden = backend.advance(initial, 0, len(backend.layers) // 2)
            hidden = backend.advance(hidden, len(backend.layers) // 2, len(backend.layers))
            direct = target.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1]
            segmented = target.model.get_output_embeddings()(backend.root.norm(hidden[:, -1]))
            difference = float((direct.float() - segmented.float()).abs().max())
            if not bool(torch.isfinite(direct).all()) or difference != 0.:
                raise ValueError(f'Actual-checkpoint segmented prefill differs from native logits: {difference}')
            parity = {'native_segment_max_abs_difference': difference, 'all_weights_on_b70':
                      all(p.device.type == 'xpu' for p in target.model.parameters()),
                      'lens_layers': lens_manifest['source_layers'], 'lens_width': config.hidden_size}
            atomic_json(args.output / 'preflight.json', {'identity': identity, 'runtime': target.fingerprint,
                        'token_budget': preflight, 'parity': parity})
            print(json.dumps({'stage': 'native-segment-parity', **parity}), flush=True)
            del inputs, backend, initial, hidden, direct, segmented
        checker = RefusalChecker()
        scored = []
        for index, task in enumerate(suite):
            response = store.get(task)
            if response is None:
                response = target.generate(task['prompt'], min_tokens=budget['refusal_floor'] if task['kind'] == 'refusal' else None)
                store.put(task, response)
                response = store.get(task)
            answer = extract_answer(response['content'], response['finish_reason'], response['thinking_open'],
                                    response.get('final_marker'), response['floor_enforced'])
            wording = checker.check(answer.text, task['language'])
            degenerate, reason = degenerate_output(answer.text)
            row = {'id': task['id'], 'kind': task['kind'], 'language': task['language'],
                   'category': task.get('category'), 'wording': wording,
                   'boundary_complete': answer.complete, 'boundary_reason': answer.reason,
                   'natural_stop': response['finish_reason'] == 'stop',
                   'completion_tokens': response['completion_tokens'], 'degenerate': degenerate,
                   'degenerate_reason': reason, 'response_digest': digest(response)}
            scored.append(row)
            atomic_json(args.output / 'screen-progress.json', {'completed': index + 1, 'total': len(suite), 'rows': scored})
            print(json.dumps({'completed': index + 1, 'total': len(suite), **row}), flush=True)
        harmful = [r for r in scored if r['kind'] == 'refusal']
        result = {'status': 'baseline-wording-screen-only', 'identity': identity, 'total': len(suite),
                  'harmful_total': len(harmful), 'harmful_wording_counts': dict(Counter(r['wording']['result'] for r in harmful)),
                  'harmful_nonterminal': sum(not r['natural_stop'] for r in harmful),
                  'harmful_degenerate': sum(r['degenerate'] for r in harmful),
                  'all_nonterminal': sum(not r['natural_stop'] for r in scored),
                  'seconds_including_load': time.perf_counter() - started, 'parity': parity, 'rows': scored,
                  'not_checked': ['independent full-answer fulfillment', 'code/math correctness', 'learned ranker',
                                  'tree search quality', 'lens exact fitting revision provenance']}
        atomic_json(args.output / 'report.json', result)
        print(json.dumps({'stage': 'complete', **{k: v for k, v in result.items() if k not in {'rows', 'identity'}}}), flush=True)
    finally:
        target.release()


if __name__ == '__main__':
    main()
