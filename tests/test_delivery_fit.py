# SPDX-License-Identifier: AGPL-3.0-or-later
"""The fit corpus builder: gate-set refusal, anchored docs, family-disjoint splits."""
import json

from scripts import delivery_fit


def test_anchored_docs_cut_at_the_next_prompt_even_with_blank_lines():
    text = "P one\nanswer one\n\nstill one\n\nP two\nanswer two\n"
    docs = delivery_fit.anchored_docs(text, ["P one", "P two"])
    assert docs == [("P one", "answer one\n\nstill one"), ("P two", "answer two")]


def test_committed_corpus_is_gate_free_and_family_disjoint():
    path = delivery_fit.REPO / "verification/q3-4b-fit-pairs"
    pairs = [json.loads(line) for line in (path / "pairs.jsonl").read_text().splitlines()]
    gate = delivery_fit.gate_ids()
    assert pairs and not ({pair["prompt_id"] for pair in pairs} & gate)
    train = [json.loads(line) for line in (path / "train.jsonl").read_text().splitlines()]
    holdout = [json.loads(line) for line in (path / "holdout.jsonl").read_text().splitlines()]
    assert {row["family_id"] for row in train}.isdisjoint({row["family_id"] for row in holdout})
    assert all(row["chosen"] and row["rejected"] for row in train + holdout)


def test_gate_set_rows_cannot_enter_the_fit(tmp_path, monkeypatch):
    gate_id = sorted(delivery_fit.gate_ids())[0]
    monkeypatch.setattr(delivery_fit, "VERIFIED",
                        delivery_fit.VERIFIED + [("capture", gate_id, "seed")])
    monkeypatch.setattr(delivery_fit, "FIT_FAMILIES", {**delivery_fit.FIT_FAMILIES,
                                                       gate_id: "advbench-o0-g000003"})
    args = type("A", (), {"source_dir": delivery_fit.REPO / "verification/ppl-probe",
                          "out": tmp_path, "holdout_family": "advbench-o0-g000003"})
    delivery_fit.build(args)
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    refused = {entry["id"]: entry["reason"] for entry in provenance["refused"]}
    assert "gate set" in refused[gate_id]
    built = {json.loads(line)["prompt_id"] for line in (tmp_path / "pairs.jsonl").read_text().splitlines()}
    assert gate_id not in built
