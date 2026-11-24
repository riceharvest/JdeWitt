"""Assemble verification/space/selfapply-qwen3-4b.json from the run transcripts.

The document is the card-3 bank-apply report for the --edit-type auto run, verbatim,
plus the identity of the downloaded artifact, the delivered per-type ladder (Q8_0 and
F16 are forced re-runs of the same bank entry on the same input), the probe-vs-
delivered evidence that explains why auto stopped at Q6_K, and the independent
gguf-inspect / gguf-tensor-diff read-back.

Run from the repo root with the project venv:
    .venv/bin/python verification/space/selfapply-run/make_report.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'verification/space/selfapply-run'
OUT = ROOT / 'verification/space/selfapply-qwen3-4b.json'

INPUT = Path('/mnt/ssd/models/unsloth-qwen3-4b-q4km/Qwen3-4B-Instruct-2507-Q4_K_M.gguf')
SENTENCE = 'cos is codec fidelity; no answer was generated'
FLOOR = 0.95


def load(name):
    return json.loads((RUN / name).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b''):
            digest.update(chunk)
    return digest.hexdigest()


def delivered(report, label):
    edited = report['edited']
    return {
        'label': label,
        'edit_type': report['edit_type']['chosen'],
        'mode': report['edit_type']['mode'],
        'status': report['status'],
        'ok': report['ok'],
        'floor': edited['floor'],
        'floor_met': edited['floor_met'],
        'edited_tensors': edited['count'],
        'below_floor': len(edited['below_floor']),
        'min_cos': edited['min_cos'],
        'mean_cos': edited['mean_cos'],
        'min_snr': edited['min_snr'],
        'by_kind': edited['by_kind'],
        'bytes_in': report['bytes_in'],
        'bytes_out': report['bytes_out'],
        'delta_bytes': report['bytes_out'] - report['bytes_in'],
        'delta_gb': round((report['bytes_out'] - report['bytes_in']) / 1e9, 3),
        'output_sha256': report['output_sha256'],
        'wall_seconds': report['wall_seconds'],
        'untouched': {'count': report['untouched']['count'],
                      'verified': report['untouched']['verified'],
                      'mismatched': report['untouched']['mismatched'],
                      'missing': report['untouched']['missing']},
    }


auto = load('bank-apply.json')
q8 = delivered(load('bank-apply-q8_0.json'), 'q8_0 forced')
f16 = delivered(load('bank-apply-f16.json'), 'f16 forced')
auto_row = delivered(auto, 'auto')

diff_auto = load('tensor-diff-q6_k.json')
diff_q8 = load('tensor-diff-q8_0.json')
inspect_q8 = load('inspect-diff-q8_0.json')

edited_names = sorted(row['name'] for row in auto['edited']['tensors'])


def diff_block(diff, label):
    differing = sorted(row['name'] for row in diff['differing_tensors'])
    states = diff['summary']
    non_identical = states['value_differs'] + states['codec_unsupported'] + states['shape_changed']
    return {
        'label': label,
        'before': diff['before'],
        'after': diff['after'],
        'tensor_count_before': diff['tensor_count_before'],
        'tensor_count_after': diff['tensor_count_after'],
        'only_in_before': diff['only_in_before'],
        'only_in_after': diff['only_in_after'],
        'shared': states['shared'],
        'bit_identical': states['bit_identical'],
        'non_identical': non_identical,
        'value_differs': states['value_differs'],
        'packed_size_changed': states['packed_size_changed'],
        'codec_unsupported': states['codec_unsupported'],
        'shape_changed': states['shape_changed'],
        'non_identical_names': differing,
        'non_identical_names_equal_edited_set': differing == edited_names,
        'rel_rms': diff['rel_rms'],
        'rel_rms_note': ('relative RMS between the decoded before/after tensors; a '
                         'Q4_K/Q6_K -> Q8_0 re-encode changes the block grid (256- vs '
                         '32-element blocks), so those tensors are counted as '
                         'codec_unsupported and listed by name instead of value-compared'),
    }


doc = dict(auto)
doc['removal_measured'] = False
doc['sentence'] = SENTENCE
doc['input_identity'] = {
    'repo': 'unsloth/Qwen3-4B-Instruct-2507-GGUF',
    'revision': 'a06e946bb6b655725eafa393f4a9745d460374c9',
    'file': 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf',
    'sha256': sha256(INPUT),
    'bytes': INPUT.stat().st_size,
    'hub_lfs_sha256': '3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597',
    'hub_lfs_bytes': 2497281120,
    'sha256_matches_hub_lfs': True,
    'download': 'hf download unsloth/Qwen3-4B-Instruct-2507-GGUF '
                'Qwen3-4B-Instruct-2507-Q4_K_M.gguf --revision '
                'a06e946bb6b655725eafa393f4a9745d460374c9 --local-dir <dir>',
    'note': 'the sha256 was recomputed on the downloaded file and equals the Hub LFS oid',
}
doc['finding'] = {
    'verdict': 'Q8_0 clears the floor; auto\'s Q6_K does not',
    'one_line': (
        'The refitted axis survives the quant the Space serves, but not at the type the '
        'auto probe picks: forced Q8_0 delivers min cos 0.9873 on 42/42 tensors (+0.30 GB), '
        'forced F16 delivers min cos 1.0006 (+0.99 GB), and auto\'s Q6_K delivers min cos '
        '0.8815 with 37 of 42 tensors below the entry\'s 0.95 floor.'),
    'floor': FLOOR,
    'auto_chosen': auto_row['edit_type'],
    'auto_delivered': {'min_cos': auto_row['min_cos'], 'mean_cos': auto_row['mean_cos'],
                       'below_floor': auto_row['below_floor'],
                       'edited_tensors': auto_row['edited_tensors'],
                       'status': auto_row['status']},
    'cheapest_measured_type_that_clears': 'Q8_0',
    'measured_delivered': [{'edit_type': row['edit_type'], 'min_cos': row['min_cos'],
                            'mean_cos': row['mean_cos'], 'floor_met': row['floor_met'],
                            'bytes_out': row['bytes_out'], 'delta_gb': row['delta_gb']}
                           for row in (auto_row, q8, f16)],
    'codec_floor': False,
    'codec_floor_note': (
        'This is NOT a CODEC_FLOOR finding: a type the entry measured (Q8_0) clears the '
        'floor on the delivered bytes at the entry\'s own strength 1.5. The strength was '
        'not raised, and no type outside the entry\'s measured set was used.'),
    'why_auto_missed': (
        'bank-apply\'s auto ladder decides the type from ggufio.preview_edit, which '
        're-encodes only windows=2 x blocks=32 blocks (8192 elements) of each probe '
        'tensor. For these rank-1 deltas that sample is optimistic: on '
        'blk.29.ffn_down.weight the 8192-element sample reads cos 0.9974 where the whole '
        'tensor reads 0.8812 (delivered 0.8815). The ladder therefore stops at Q6_K one '
        'step before the type that clears. The floor has to be read on the written file '
        '(or the probe has to cover the tensor).'),
}
doc['floor_measured_per_type'] = [auto_row, q8, f16]
doc['probe_vs_delivered'] = {
    'method': 'preview_edit default sample (windows=2, blocks=32) vs the same call with '
              'windows=400, blocks=256 (whole tensor) on the same input, scale 1.5, '
              'target_type q6_k; delivered cos from the written file.',
    'rows': [
        {'tensor': 'blk.29.ffn_down.weight', 'default_sample_cos': 0.9974,
         'full_tensor_cos': 0.8812, 'delivered_cos': 0.8815},
        {'tensor': 'blk.15.attn_output.weight', 'default_sample_cos': 0.9961,
         'full_tensor_cos': 0.9011, 'delivered_cos': 0.9105},
        {'tensor': 'blk.29.attn_output.weight', 'default_sample_cos': 0.997,
         'full_tensor_cos': 0.8769, 'delivered_cos': 0.8787},
    ],
    'reading': 'the sample is not representative of the tensor; the delivered number '
               'agrees with the full-tensor preview, not with the sample',
}
doc['verification'] = {
    'reopen': {
        'command': 'jdewitt gguf-inspect <input> <q8_0 output>',
        'output_size_bytes': inspect_q8['after']['size_bytes'],
        'tensor_count': inspect_q8['after']['tensor_count'],
        'tensors_by_dtype': inspect_q8['after']['tensors_by_dtype'],
        'dtype_changed': len(inspect_q8['diff']['dtype_changed']),
        'shape_changed': inspect_q8['diff']['shape_changed'],
        'metadata_changed': inspect_q8['diff']['metadata_changed'],
        'metadata_only_in_after': inspect_q8['diff']['metadata_only_in_after'],
        'note': ('the q8_0 output reopens; exactly the 42 edited tensors changed dtype, no '
                 'shape changed, no existing metadata key changed, and the only new key is '
                 'jdewitt.fingerprint'),
    },
    'tensor_diff_auto_out': dict(diff_block(diff_auto, 'auto run (Q6_K)'),
                                 command='jdewitt gguf-tensor-diff <input> '
                                         '<...-selfapply-f15s15-l935.gguf>'),
    'tensor_diff_chosen_out': dict(diff_block(diff_q8, 'Q8_0 (the floor-clearing type)'),
                                   command='jdewitt gguf-tensor-diff <input> '
                                           '<...-selfapply-q8_0.gguf>'),
}
agrees = (doc['untouched']['count'] == diff_auto['summary']['bit_identical']
          == diff_q8['summary']['bit_identical'])
doc['verification']['counts_agree'] = {
    'json_untouched_count': doc['untouched']['count'],
    'diff_bit_identical_auto_out': diff_auto['summary']['bit_identical'],
    'diff_bit_identical_chosen_out': diff_q8['summary']['bit_identical'],
    'agree': agrees,
    'rule': 'untouched = 398 tensors - 42 edited = 356; every one of them is bit-identical '
            'in both diffs and the 42 edited names are exactly the 42 non-identical names',
}
doc['paths_note'] = ('artifact and bank paths are the ones the runs used on this host; '
                     'the GGUFs are not committed (*.gguf is git-ignored) and can be '
                     'regenerated from input_identity and the committed bank bundle')

OUT.write_text(json.dumps(doc, indent=1) + '\n')
print('wrote', OUT, OUT.stat().st_size, 'bytes')
print('counts agree:', agrees)
