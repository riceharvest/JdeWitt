#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU falsifier for the SAE feature-ablation contract (P5 workstream b).

No model, no checkpoint, no generation, no claim about refusal behaviour. It checks
the arithmetic and the edit surface:

  1. ``feature_ablation`` on synthetic rectangular tensors -- 512x64 and 256x512
     output-side, plus the GQA shape (256 in x 1280 out in ggml order, i.e. a torch
     ``[1280, 256]`` read-side matrix) -- prints per-tensor
     cos(realized, intended);
  2. the resident path on a tiny 2-layer model through a real ``Editor``
     (``edit_targets='full'``, so attention/FFN read and write sides are all in the
     target list) prints per-tensor cos, then reverts and reports
     ``revert byte-identical true`` (``torch.equal`` against the pre-edit copy);
  3. after ablating direction ``d`` from ``W`` the residual projection along ``d`` is
     0 to float tolerance, printed per tensor;
  4. a zero-norm direction and a NaN tensor are refused, each by its own exit code
     (``--expect-refusal CASE`` is the mode the parent re-executes).

Exit 0 means every printed check held. This is a falsifier, not a test suite and not
evidence that an SAE feature carries refusal: that needs the model and the arm card,
including ``cos_to_axis(feature, oracle_axis)``.
"""
import argparse
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch
from torch import nn

from jdewitt import sae

RESIDUAL_TOL = 1e-5
COS_FLOOR = 0.999


def _require(condition, message):
    if not condition:
        raise SystemExit(f'FAIL: {message}')


def _synthetic_cases():
    """(label, weight, direction, side) -- shapes as the gate names them."""
    g = torch.Generator().manual_seed(20261122)
    cases = []
    for label, shape in (('512x64', (512, 64)), ('256x512', (256, 512))):
        cases.append((label, torch.randn(shape, generator=g), torch.randn(shape[0], generator=g), 'output'))
    # GQA: hidden 256 in, 10 heads x 128 = 1280 out. The tensor is [out, in] here and
    # the residual-stream direction addresses its READ side, because out != hidden.
    cases.append(('1280x256 (GQA q_proj, ggml 256x1280)',
                  torch.randn(1280, 256, generator=g), torch.randn(256, generator=g), 'input'))
    return cases


def _residual(basis, weight, side):
    rows = torch.as_tensor(basis).reshape(-1, 1)
    if side == 'output':
        return float((rows.T @ weight).norm())
    return float((weight @ rows).norm())


def check_tensors():
    print('tensor cases -- feature_ablation on synthetic rectangular weights (fp32 control)')
    header = f'  {"tensor":34s} {"side":7s} {"cos(realized,intended)":>22s} {"max|delta err|":>14s} {"residual":>10s}'
    print(header)
    worst_residual = 0.0
    for label, weight, direction, side in _synthetic_cases():
        delta = sae.feature_ablation(direction, weight, 1.0, side)
        after = weight + delta
        cos = sae.realized_cos(weight, delta, after)
        d = direction.reshape(-1)
        before_proj = _residual(d, weight, side)
        after_proj = _residual(d, after, side)
        relative = after_proj / before_proj
        worst_residual = max(worst_residual, relative)
        print(f'  {label:34s} {side:7s} {cos:22.6f} {float((after - weight - delta).abs().max()):14.3e} '
              f'{relative:10.3e}')
        _require(relative <= RESIDUAL_TOL,
                 f'{label}: residual projection along d is {relative:.3e} of its pre-edit '
                 f'norm; expected <= {RESIDUAL_TOL:.0e}')
        _require(cos >= COS_FLOOR, f'{label}: realized cos {cos:.6f} < {COS_FLOOR}')
    print(f'  max residual across tensor cases: {worst_residual:.3e} (tolerance {RESIDUAL_TOL:.0e})')


def _tiny_model(hidden=64, heads=5, kv_heads=1, inter=128, layers=2):
    model = nn.Module()
    blocks = []
    for _ in range(layers):
        block = nn.Module()
        block.self_attn = nn.Module()
        block.self_attn.o_proj = nn.Linear(hidden, hidden, bias=False)
        block.self_attn.q_proj = nn.Linear(hidden, heads * hidden, bias=False)   # GQA: out != hidden
        block.self_attn.k_proj = nn.Linear(hidden, kv_heads * hidden, bias=False)
        block.self_attn.v_proj = nn.Linear(hidden, kv_heads * hidden, bias=False)
        block.mlp = nn.Module()
        block.mlp.gate_proj = nn.Linear(hidden, inter, bias=False)
        block.mlp.up_proj = nn.Linear(hidden, inter, bias=False)
        block.mlp.down_proj = nn.Linear(inter, hidden, bias=False)
        blocks.append(block)
    model.layers = nn.ModuleList(blocks)
    model.config = SimpleNamespace(model_type='llama', hidden_size=hidden)
    return model


def check_resident(seed=7):
    from jdewitt.editor import Editor

    hidden, n_features = 64, 2
    g = torch.Generator().manual_seed(seed)
    model = _tiny_model(hidden=hidden)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=g))

    # A minimal SAE on disk, loaded through load(): shapes and finiteness are checked
    # there, and the feature directions come from the decoder columns, not a constant.
    with tempfile.TemporaryDirectory() as directory:
        path = f'{directory}/sae.pt'
        torch.save({'W_enc': torch.randn(8, hidden, generator=g),
                    'W_dec': torch.randn(hidden, 8, generator=g),
                    'b_enc': torch.zeros(8), 'b_dec': torch.zeros(hidden)}, path)
        loaded = sae.load(path)
    _require(loaded.width == hidden and loaded.features == 8, 'loaded SAE has the wrong shape')
    features = [0, 3]
    directions = loaded.directions(features)          # [2, 64], columns are not orthogonal
    basis, orthogonalized = sae.orthonormalize(directions)
    _require(orthogonalized, 'expected non-orthogonal SAE columns to be orthonormalized')
    _require(torch.allclose(basis @ basis.T, torch.eye(n_features), atol=1e-5),
             'orthonormalize did not return an orthonormal basis')

    editor = Editor(model, rank=n_features, edit_targets='full')
    print(f'\nresident cases -- Editor(edit_targets=full, rank={n_features}) on a '
          f'{len(model.layers)}-layer model, {n_features} features {features}')
    print(f'  {"layer":5s} {"group":8s} {"side":7s} {"key":36s} {"shape":12s} {"cos":>10s} {"residual":>10s}')
    edit = sae.apply_to_resident(editor, directions, scale=1.0)
    worst_cos, worst_residual = 1.0, 0.0
    for record in edit.records:
        relative = record['residual_projection'] / max(record['residual_before'], 1e-12)
        worst_cos = min(worst_cos, record['cos'])
        worst_residual = max(worst_residual, relative)
        print(f'  {record["layer"]:<5d} {record["group"]:8s} {record["side"]:7s} '
              f'{record["source_key"]:34s} {str(record["shape"]):12s} '
              f'{record["cos"]:10.6f} {relative:10.3e}')
    sides = sorted({record['side'] for record in edit.records})
    gqa = [record for record in edit.records if record['group'] == 'attn_in'
           and record['source_key'].endswith('q_proj.weight')]
    _require(len(edit.records) == len(editor.targets),
             f'edited {len(edit.records)} of {len(editor.targets)} resident targets')
    _require(sides == ['input', 'output'], f'resident edit covered sides {sides}, expected both')
    _require(gqa and gqa[0]['shape'][0] == 5 * hidden,
             'the GQA q_proj target is missing or was edited on the wrong side')
    _require(worst_residual <= RESIDUAL_TOL,
             f'resident residual projection along d is {worst_residual:.3e}; expected <= {RESIDUAL_TOL:.0e}')
    _require(worst_cos >= COS_FLOOR,
             f'resident realized cos {worst_cos:.6f} < {COS_FLOOR}: the factor path does not '
             f'serve the projection it claims')
    print(f'  targets {len(edit.records)}, min cos {worst_cos:.6f}, max residual {worst_residual:.3e}')

    # The arm-card number this lane must report instead of a novelty claim.
    axis = torch.randn(hidden, generator=g)
    print(f'  cos(feature {features[0]}, axis) = {sae.cos_to_axis(directions[0], axis):+.6f} '
          f'-- report this against the oracle axis in the arm card, not "new method"')

    report = edit.revert()
    identical = str(bool(report['byte_identical'])).lower()
    print(f'revert byte-identical {identical} ({report["targets"]} targets, '
          f'factors_cleared {str(bool(report["factors_cleared"])).lower()})')
    _require(report['byte_identical'], 'revert did not restore the pre-edit weight bytes')
    _require(report['factors_cleared'], 'revert left factors or row_scale behind')
    _require(not editor.targets[0][-1].enabled, 'revert left a target enabled')
    # And the served weight is the pre-edit weight again, not just the stored one.
    for _index, _group, _side, _parent, _name, adapter in editor.targets:
        _require(not adapter.enabled and float(adapter.row_scale.max()) == 1.0,
                 f'{adapter.source_key} still serves an edit after revert')


def check_parity():
    """The contract's central claim, measured: this IS spanfit.removal_delta.

    The repo's removal_delta assumes a unit axis and does not divide by ``d^T d``.
    A raw (unnormalized) SAE decoder column must therefore land the same delta as
    removal_delta fed the unit version -- if it does not, the SAE lane is a different
    edit wearing the same name.
    """
    from jdewitt.spanfit import removal_delta

    g = torch.Generator().manual_seed(5)
    weight = torch.randn(256, 512, generator=g)
    raw = torch.randn(256, generator=g) * 7.0          # deliberately not unit norm
    ours = sae.feature_ablation(raw, weight)
    theirs = removal_delta(weight, raw / raw.norm())
    scale = float(theirs.abs().max())
    diff = float((ours - theirs).abs().max())
    print('\nparity -- spanfit.removal_delta on the unit axis vs feature_ablation on the raw column')
    print(f'  max|diff| {diff:.3e} against a max |delta| of {scale:.3f} '
          f'({diff / scale:.3e} relative)')
    _require(diff <= 1e-5 * scale,
             f'feature_ablation and spanfit.removal_delta disagree by {diff:.3e} '
             f'({diff / scale:.3e} relative): the SAE lane is not the same edit')


def check_guards(hidden=16):
    """The resident path's own refusals: rank mismatch and a second apply."""
    from jdewitt.editor import Editor

    g = torch.Generator().manual_seed(19)
    directions = torch.randn(2, hidden, generator=g)

    def refused(call, needle, what):
        try:
            call()
        except ValueError as error:
            _require(needle in str(error), f'{what} was refused, but the message is not '
                                           f'actionable: {error}')
            print(f'  ok   {what:24s} refused: {str(error)[:96]}')
            return
        raise SystemExit(f'FAIL: {what} was not refused')

    print('\nguards -- the resident path must refuse what it cannot honour:')
    # An Editor built at rank 3 cannot serve a 2-feature edit without padding the
    # basis, which would remove directions nobody selected.
    refused(lambda: sae.apply_to_resident(Editor(_tiny_model(hidden=hidden, inter=32),
                                                rank=3, edit_targets='writeback'), directions),
            'built at rank', 'rank mismatch')
    editor = Editor(_tiny_model(hidden=hidden, inter=32), rank=2, edit_targets='writeback')
    edit = sae.apply_to_resident(editor, directions)
    refused(lambda: edit.apply(directions), 'already applied', 'double apply')
    edit.revert()


def _refusal_case(case):
    """Build the bad input for one case and call the API; does not catch, by design."""
    g = torch.Generator().manual_seed(3)
    weight = torch.randn(64, 32, generator=g)
    if case == 'zero-norm':
        return sae.feature_ablation(torch.zeros(64), weight)
    if case == 'nan-direction':
        direction = torch.randn(64, generator=g)
        direction[5] = float('nan')
        return sae.feature_ablation(direction, weight)
    if case == 'nan-weight':
        weight = weight.clone()
        weight[7, 3] = float('nan')
        return sae.feature_ablation(torch.randn(64, generator=g), weight)
    raise SystemExit(f'unknown refusal case {case!r}')


def expect_refusal(case):
    """Child mode: exit nonzero if (and only if) the bad input was refused."""
    try:
        _refusal_case(case)
    except ValueError as error:
        raise SystemExit(f'refused {case}: {error}')
    except TypeError as error:
        raise SystemExit(f'refused {case}: {error}')
    print(f'NOT REFUSED: {case} returned a delta; the guard is gone')
    raise SystemExit(0)


def check_refusals():
    print('\nrefusals -- each case is re-executed in a child and must exit nonzero:')
    worst = 0
    for case in ('zero-norm', 'nan-direction', 'nan-weight'):
        result = subprocess.run([sys.executable, __file__, '--expect-refusal', case],
                                capture_output=True, text=True, check=False)
        line = (result.stderr.strip().splitlines() or ['<no message>'])[0]
        worst = max(worst, 0 if result.returncode else 1)
        shown = 'ok  ' if result.returncode else 'FAIL'
        print(f'  {shown} {case:14s} rc={result.returncode}  {line}')
        _require(result.returncode != 0, f'{case} was not refused')
    _require(worst == 0, 'a refusal case exited zero')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--expect-refusal', choices=('zero-norm', 'nan-direction', 'nan-weight'),
                        help='child mode: attempt one bad input and exit nonzero when refused')
    args = parser.parse_args(argv)
    if args.expect_refusal:
        return expect_refusal(args.expect_refusal)

    print('SAE feature-ablation contract check (CPU, no model, no weights)\n')
    check_tensors()
    check_parity()
    check_resident()
    check_guards()
    check_refusals()
    print('\nok: contract holds on CPU. Nothing here measures refusal or retention.')


if __name__ == '__main__':
    main()
