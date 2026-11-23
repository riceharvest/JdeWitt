"""Pair-diff axis vs the banked oracle axis: the decision instrument, on fixtures.

Both fixture outcomes are asserted from the command's own stdout, not inferred
from the packet: an axis identical to the oracle must PRINT cos 1.0 and ``dead``,
and an orthogonal axis must PRINT ``orthogonal``.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
L, W = 6, 8
FOOTPRINT = [2, 3, 4]


def _unit(seed: int) -> torch.Tensor:
    vector = torch.randn(W, generator=torch.Generator().manual_seed(seed))
    return vector / vector.norm()


def _oracle() -> torch.Tensor:
    axes = torch.zeros(L, W)
    for layer in FOOTPRINT:
        axes[layer] = _unit(100 + layer)
    return axes


def _orthogonal_to(row: torch.Tensor, seed: int = 400) -> torch.Tensor:
    candidate = _unit(seed)
    candidate = candidate - (candidate @ row) * row
    return candidate / candidate.norm()


def _write_corpus(tmp_path, n_pairs=10, delta_kind="same", extra_unread=False):
    """Capture records + vetted pair JSONL; ``delta_kind`` picks the pair difference."""
    from jdewitt.delivery_pairs import make_pair, vet, write_pairs

    oracle = _oracle()
    generator = torch.Generator().manual_seed(7)
    records, pairs = [], []
    for index in range(n_pairs):
        prompt_id = f"row-{index:03d}"
        refusal = torch.randn(L, W, generator=generator)
        delta = torch.zeros(L, W)
        for layer in FOOTPRINT:
            if delta_kind == "same":
                delta[layer] = oracle[layer]
            elif delta_kind == "half":
                # cos 0.5: half the oracle direction plus an orthogonal half
                delta[layer] = 0.5 * oracle[layer] + (3 ** 0.5 / 2) * _orthogonal_to(oracle[layer])
            else:
                delta[layer] = _orthogonal_to(oracle[layer])
        delivery = refusal + delta
        records.append(({"prompt_id": prompt_id, "label": "bad", "frame": "none", "kind": "refusal"},
                        refusal))
        records.append(({"prompt_id": prompt_id, "label": "deliver", "frame": "seed", "kind": "refusal"},
                        delivery))
        pair = make_pair(prompt_id=prompt_id, prompt=f"request {index}", kind="refusal",
                         family_id="sb-hold-1", category="1",
                         source={"kind": "capture", "records": "fixture.pt"},
                         refusal={"frame": "none", "answer": "I cannot help."},
                         delivery={"frame": "seed", "answer": "Here is exactly how."})
        pairs.append(vet(pair, "deliver", "fixture: delivery answers the request", "test-reader"))
    if extra_unread:
        records.append(({"prompt_id": "row-unread", "label": "deliver", "frame": "seed",
                         "kind": "refusal"}, torch.randn(L, W, generator=generator)))

    records_path = tmp_path / "records.pt"
    torch.save({"records": records}, records_path)
    verdicts_path = tmp_path / "pairs.jsonl"
    write_pairs(verdicts_path, pairs)
    oracle_path = tmp_path / "oracle.pt"
    torch.save({"axes_attn": oracle}, oracle_path)
    return records_path, verdicts_path, oracle_path


def _run(tmp_path, records, verdicts, oracle, *extra):
    out = tmp_path / "packet.json"
    env = dict(os.environ, PYTHONPATH=str(REPO))
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "pair_axis_check.py"),
         "--records", str(records), "--verdicts", str(verdicts), "--oracle", str(oracle),
         "--out", str(out), *extra],
        cwd=REPO, env=env, capture_output=True, text=True), out


def _packet(out: Path) -> dict:
    return json.loads(out.read_text())


def test_identical_axis_prints_cos_one_and_dead(tmp_path):
    records, verdicts, oracle = _write_corpus(tmp_path)
    done, out = _run(tmp_path, records, verdicts, oracle)
    assert done.returncode == 0, done.stderr
    assert "+1.000000" in done.stdout, done.stdout
    assert "DEAD" in done.stdout and "rank-1 delivery hypothesis dead" in done.stdout
    packet = _packet(out)
    assert packet["decision"] == "dead"
    assert packet["mean_abs_cos"] == pytest.approx(1.0)
    assert packet["min_abs_cos"] == pytest.approx(1.0)
    assert [row["layer"] for row in packet["cos"]] == FOOTPRINT
    assert packet["fit"]["pairs_used"] == 10
    assert packet["verdicts"]["n_vetted_pairs"] == 10
    assert len(packet["records"]["sha256"]) == 64


def test_orthogonal_axis_prints_orthogonal_and_justifies_one_merge(tmp_path):
    records, verdicts, oracle = _write_corpus(tmp_path, delta_kind="orthogonal")
    done, out = _run(tmp_path, records, verdicts, oracle)
    assert done.returncode == 0, done.stderr
    assert "+0.000000" in done.stdout, done.stdout
    assert "ORTHOGONAL" in done.stdout and "one merge test justified" in done.stdout
    assert _packet(out)["decision"] == "orthogonal"


def test_between_the_thresholds_reports_inconclusive_and_spends_no_merge(tmp_path):
    records, verdicts, oracle = _write_corpus(tmp_path, delta_kind="half")
    done, out = _run(tmp_path, records, verdicts, oracle)
    assert done.returncode == 0, done.stderr
    assert "INCONCLUSIVE" in done.stdout and "do NOT spend a merge" in done.stdout
    packet = _packet(out)
    assert packet["decision"] == "inconclusive"
    assert 0.3 < packet["mean_abs_cos"] < 0.9


def test_fewer_than_ten_vetted_pairs_exits_3_insufficient(tmp_path):
    records, verdicts, oracle = _write_corpus(tmp_path, n_pairs=9)
    done, out = _run(tmp_path, records, verdicts, oracle)
    assert done.returncode == 3, (done.returncode, done.stdout, done.stderr)
    assert "insufficient" in done.stdout
    packet = _packet(out)
    assert packet["decision"] == "insufficient" and packet["cos"] == []
    assert packet["verdicts"]["n_vetted_pairs"] == 9


def test_deliver_record_without_a_verdict_exits_nonzero_naming_it(tmp_path):
    records, verdicts, oracle = _write_corpus(tmp_path, extra_unread=True)
    done, _ = _run(tmp_path, records, verdicts, oracle)
    assert done.returncode != 0
    assert "deliver record without a verdict" in done.stderr
    assert "row-unread[frame=seed]" in done.stderr


def test_oracle_directory_resolves_its_single_axes_file(tmp_path):
    records, verdicts, oracle = _write_corpus(tmp_path)
    oracle_dir = tmp_path / "oracle-dir"
    oracle_dir.mkdir()
    oracle.rename(oracle_dir / "oaxes135.pt")
    done, out = _run(tmp_path, records, verdicts, oracle_dir)
    assert done.returncode == 0, done.stderr
    assert _packet(out)["oracle"]["key"] == "axes_attn"


def test_per_layer_axes_pairs_uses_within_request_differences():
    from jdewitt.spanfit import per_layer_axes

    # Group means point at (-0.71,-0.71); both within-request differences point at
    # (+0.71,+0.71). pairs=True must return the latter, pairs=False the former.
    records = [
        ({"prompt_id": "a", "label": "bad", "frame": "none"}, torch.tensor([[1.0, 0.0]])),
        ({"prompt_id": "a", "label": "deliver", "frame": "seed"}, torch.tensor([[1.0, 1.0]])),
        ({"prompt_id": "b", "label": "bad", "frame": "none"}, torch.tensor([[0.0, 1.0]])),
        ({"prompt_id": "b", "label": "deliver", "frame": "seed"}, torch.tensor([[1.0, 1.0]])),
    ]
    paired = per_layer_axes(records, "bad", "deliver", pairs=True)
    grouped = per_layer_axes(records, "bad", "deliver")
    assert torch.allclose(paired, torch.tensor([[2 ** -0.5, 2 ** -0.5]]), atol=1e-6)
    assert torch.allclose(grouped, torch.tensor([[-2 ** -0.5, -2 ** -0.5]]), atol=1e-6)


def test_per_layer_axes_pairs_needs_prompt_id_to_match_sides():
    from jdewitt.spanfit import per_layer_axes

    records = [({"label": "bad"}, torch.tensor([[1.0, 0.0]])),
               ({"label": "deliver"}, torch.tensor([[1.0, 1.0]]))]
    with pytest.raises(SystemExit, match="prompt_id"):
        per_layer_axes(records, "bad", "deliver", pairs=True)
