# SPDX-License-Identifier: AGPL-3.0-or-later
"""The both-axes packet: counts come from files, the gate is applied mechanically."""
import json
import math

import pytest

from scripts.delivery_read_packet import build, mcnemar, paired

MATHS_TOTAL, CODE_TOTAL = 180, 61   # the authored suite's shape (241 rows)


def write_wave(directory, rows, serve_config="ctx 8192 parallel 4 temp 0"):
    directory.mkdir(parents=True)
    (directory / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    (directory / "wave.json").write_text(json.dumps({"serve_config": serve_config}))


def row(row_id, kind, passed, status="ok", answer="1", tokens=5):
    return {"id": row_id, "kind": kind, "passed": passed, "status": status,
            "answer": answer, "completion_tokens": tokens}


def make_inputs(tmp_path, *, delivered, maths_pass, code_pass, base_maths_pass, base_code_pass,
                unread=False):
    retention_rows, base_rows = [], []
    for index in range(MATHS_TOTAL):
        retention_rows.append(row(f"m{index}", "math", index < maths_pass, answer="12345"))
        base_rows.append(row(f"m{index}", "math", index < base_maths_pass, answer="1234567"))
    for index in range(CODE_TOTAL):
        retention_rows.append(row(f"c{index}", "code", index < code_pass))
        base_rows.append(row(f"c{index}", "code", index < base_code_pass))
    trained, base = tmp_path / "trained", tmp_path / "base"
    write_wave(trained, retention_rows)
    write_wave(base, base_rows)
    removal_rows = [row(f"h{index}", "refusal", False) for index in range(22)]
    write_wave(tmp_path / "removal", removal_rows)
    verdicts = {f"h{index}": {"verdict": "deliver" if index < delivered else "pivot",
                              "reason": "read", "reader": "hermes#test"} for index in range(22)}
    if unread:
        verdicts.pop("h21")
    reads = tmp_path / "reads.json"
    reads.write_text(json.dumps({"verdicts": verdicts, "excluded": {"sb-hold-6": "protected"}}))
    return reads, tmp_path / "removal", trained, base


def run(tmp_path, **kwargs):
    reads, removal, trained, base = make_inputs(tmp_path, **kwargs)
    out = tmp_path / "packet.json"
    build(type("A", (), {"reads": reads, "remove": removal, "retention": trained,
                         "baseline": base, "training_report": None,
                         "suite": "s.jsonl", "gpu_seconds": 12.5,
                         "gpu_seconds_note": "test", "out": out})())
    return json.loads(out.read_text())


OK = dict(maths_pass=160, code_pass=30, base_maths_pass=150, base_code_pass=30)


def test_gate_passes_only_when_all_three_conditions_hold(tmp_path):
    packet = run(tmp_path, delivered=3, **OK)
    assert packet["content"]["delivered_count"] == 3
    assert packet["retention"]["maths"]["passed"] == 160
    assert packet["gate"]["content"] is True
    assert packet["gate"]["maths"] is True
    assert packet["gate"]["code"] is True
    assert packet["gate"]["verdict"] == "pass"


def test_below_gate_is_reported_as_a_tone_result(tmp_path):
    packet = run(tmp_path, delivered=2, **OK)
    assert packet["gate"]["content"] is False
    assert packet["gate"]["verdict"] == "below-gate (tone result)"


def test_maths_floor_and_code_tolerance_are_separate(tmp_path):
    maths_short = run(tmp_path / "a", delivered=3, **{**OK, "maths_pass": 147})
    assert maths_short["gate"]["maths"] is False and maths_short["gate"]["code"] is True
    code_short = run(tmp_path / "b", delivered=3, **{**OK, "code_pass": 27})
    assert code_short["gate"]["code"] is False and code_short["gate"]["maths"] is True


def test_an_unread_removal_answer_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, delivered=3, unread=True, **OK)


def test_paired_counts_and_exact_mcnemar():
    assert mcnemar(0, 0) == 1.0
    assert abs(mcnemar(1, 9) - 2 * sum(math.comb(10, i) for i in range(2)) / 2 ** 10) < 1e-12
    trained = [row("a", "math", True), row("b", "math", False), row("c", "math", True)]
    base = [row("a", "math", True), row("b", "math", True), row("c", "math", False)]
    result = paired(trained, base)
    assert (result["lost"], result["gained"], result["rows"]) == (1, 1, 3)


def test_median_answer_length_is_per_arm(tmp_path):
    packet = run(tmp_path, delivered=3, **OK)
    assert packet["retention"]["trained"]["median_answer_chars"] == 5
    assert packet["retention"]["base"]["median_answer_chars"] == 7
