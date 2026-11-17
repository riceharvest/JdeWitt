"""Read the actual fitted lens on saved baseline prompts; no edits or new decoding."""
import argparse
import json
from pathlib import Path

import torch

from jdewitt.answers import atomic_json
from jdewitt.hf import HFModel
from jdewitt.pipeline import configure_compiler_paths
from jdewitt.storage import file_digest
from jdewitt.tree_hf import HFSegments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--lens', type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((args.baseline / 'report.json').read_text())
    suite = json.loads((args.baseline / 'suite.json').read_text())
    identity = baseline['identity']
    if file_digest(args.lens) != identity['lens_sha256']:
        raise ValueError('Lens differs from the baseline pinned artifact')
    # Use the first measured explicit refusal and first benign English control,
    # not fabricated labels or a newly authored harmful request.
    refused = {r['id'] for r in baseline['rows'] if r['kind'] == 'refusal'
               and r['wording']['result'] == 'refusal_detected' and r['natural_stop']}
    refusal = next(row for row in suite if row['id'] in refused and row['language'] == 'en')
    control = next(row for row in suite if row['kind'] == 'harmless' and row['language'] == 'en')
    configure_compiler_paths()
    target = HFModel(identity['model'], identity['revision'], rank=1, max_tokens=identity['max_tokens'],
                     ctx_size=identity['ctx_size'], dtype=identity['dtype'], quantization='none')
    records = []
    try:
        lens = torch.load(args.lens, map_location='cpu', weights_only=True, mmap=True)
        head = target.model.get_output_embeddings()
        with torch.inference_mode():
            for task in (refusal, control):
                inputs, _ = target.encode(task['prompt'])
                backend = HFSegments(target, inputs)
                hidden = backend.prepare()
                layers = []
                for layer in range(backend.n_layers):
                    hidden = backend.advance(hidden, layer, layer + 1)
                    vector = hidden[0, -1].float()
                    if layer in lens['J']:
                        matrix = lens['J'][layer].to(device=target.device, dtype=torch.float32)
                        vector = vector @ matrix.T
                        del matrix
                    elif layer != backend.n_layers - 1:
                        raise ValueError('Missing nonterminal lens checkpoint')
                    normalized = backend.root.norm(vector.to(head.weight.dtype))
                    logits = head(normalized).float()
                    if not bool(torch.isfinite(logits).all()):
                        raise ValueError('Nonfinite actual-checkpoint Jacobian readout')
                    scores, tokens = logits.topk(8)
                    layers.append({'checkpoint': layer, 'readout': 'actual-final' if layer == backend.n_layers - 1 else 'jacobian',
                                   'top_tokens': [{'id': int(token), 'text': target.tokenizer.decode([int(token)]),
                                                   'logit': float(score)} for token, score in zip(tokens, scores)]})
                    del normalized, logits, vector, scores, tokens
                records.append({'id': task['id'], 'kind': task['kind'], 'layers': layers})
                del inputs, backend, hidden
        result = {'status': 'actual-B70-finite-Jacobian-readouts', 'lens_sha256': identity['lens_sha256'],
                  'model_revision': identity['revision'], 'prompts': len(records),
                  'readouts': sum(len(row['layers']) for row in records), 'records': records,
                  'not_checked': ['refusal-prediction accuracy', 'edited-state calibration', 'ranker training',
                                  'independent task fulfillment', 'exact upstream lens-fitting model revision']}
        atomic_json(args.baseline / 'lens-readouts.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)
        # Keep display bounded; detailed readouts are saved for all checkpoints.
        for row in records:
            print(json.dumps({'id': row['id'], 'late_readouts': [
                {'checkpoint': item['checkpoint'], 'tokens': [t['text'] for t in item['top_tokens']]}
                for item in row['layers'][-5:]]}), flush=True)
        del head, lens
    finally:
        target.release()


if __name__ == '__main__':
    main()
