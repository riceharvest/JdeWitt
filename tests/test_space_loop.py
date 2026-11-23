# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Space loop: three labels from measurement.state, the card, the dry run."""
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from jdewitt import bank_licence, space_loop as loop

REPO = Path(__file__).resolve().parents[1]
BANK = REPO / 'spaces' / 'bank'
FIXTURE = REPO / 'verification' / 'space' / 'fixtures' / 'loop'
FIXTURE_BANK = FIXTURE / 'bank'
BASE, FT, OTHER = 'jdewitt/loop-fixture', 'jdewitt/loop-fixture-ft', 'jdewitt/loop-other'
BANNED = re.compile(r'uncensored|refusals removed|jailbroken|no refusals', re.I)


@pytest.fixture(autouse=True)
def _clear_licence_cache():
    bank_licence.clear_cache()
    yield
    bank_licence.clear_cache()


@pytest.fixture(scope='module')
def hub():
    return loop.FixtureHub(FIXTURE)


@pytest.fixture(scope='module')
def measured_job(tmp_path_factory):
    root = tmp_path_factory.mktemp('jobs')
    return loop.start_job(BASE, 'base-rev', 'model-Q4_K_M.gguf', 'Q4_K_M',
                          bank_dir=FIXTURE_BANK, hub=loop.FixtureHub(FIXTURE), root=root)


def _fixture_copy(tmp_path, mutate):
    """A private copy of the fixture hub with `hub.json` mutated by `mutate`."""
    document = json.loads((FIXTURE / 'hub.json').read_text())
    mutate(document)
    (tmp_path / 'hub.json').write_text(json.dumps(document))
    (tmp_path / 'repos').symlink_to(FIXTURE / 'repos')
    (tmp_path / 'bank').symlink_to(FIXTURE_BANK)
    return tmp_path


# ------------------------------------------------------------------------ labels


def test_the_three_labels_are_the_roadmap_vocabulary():
    assert loop.LABELS == ('measured_artifact_class', 'banked_unmeasured', 'refused')


def test_label_is_measured_only_for_the_exact_artifact_class(hub):
    document = loop.load_bank(FIXTURE_BANK)
    entry = document['entries'][0]
    exact = {'repo': BASE, 'revision': 'base-rev', 'quant': 'Q4_K_M'}
    fine_tune = {'repo': FT, 'revision': 'ft-rev', 'quant': 'Q4_K_M'}
    other_revision = {'repo': BASE, 'revision': 'deadbeef', 'quant': 'Q4_K_M'}
    other_quant = {'repo': BASE, 'revision': 'base-rev', 'quant': 'Q8_0'}
    assert loop.artifact_label(entry, exact) == 'measured_artifact_class'
    assert loop.artifact_label(entry, fine_tune) == 'banked_unmeasured'
    assert loop.artifact_label(entry, other_revision) == 'banked_unmeasured'
    assert loop.artifact_label(entry, other_quant) == 'banked_unmeasured'


def test_a_banked_unmeasured_entry_never_claims_a_measurement():
    entry = loop.load_bank(REPO / 'verification' / 'space' / 'fixtures' / 'bank')['entries'][0]
    artifact = {'repo': 'jdewitt/bank-fixture', 'revision': 'fixture', 'quant': 'F16'}
    assert loop.artifact_label(entry, artifact) == 'banked_unmeasured'


def test_the_unmeasured_sentence_is_the_roadmap_copy():
    entry = loop.load_bank(FIXTURE_BANK)['entries'][0]
    artifact = {'repo': FT, 'revision': 'ft-rev', 'quant': 'Q4_K_M'}
    sentence = loop.label_sentence('banked_unmeasured', entry, artifact,
                                   delivered_cos=0.9999, tensors=4)
    assert sentence.startswith('This file was **not** measured.')
    assert '`jdewitt/loop-fixture@base-rev`' in sentence
    assert '`attn 1.5, mlp 0.75`' in sentence
    assert 'delivered cos `0.9999` on `4` tensors' in sentence
    assert 'No claim is made about this model\'s answers.' in sentence


def test_the_measured_sentence_names_the_class_and_the_numbers():
    entry = loop.load_bank(FIXTURE_BANK)['entries'][0]
    sentence = loop.label_sentence('measured_artifact_class', entry, {})
    assert 'Measured on `jdewitt/loop-fixture@base-rev` `Q4_K_M`' in sentence
    assert '`40/40` harmful' in sentence
    assert '`120/120` held-out' in sentence
    assert 'retention `241/241`' in sentence
    assert 'drift `0.0058` ln ppl' in sentence


def test_the_refusal_copy_names_the_reason_and_what_would_cover_it():
    sentence = loop.refusal_copy(
        'BANK_MISS', "the artifact declares architecture 'otherach' and the bank has no entry")
    assert sentence.startswith('`BANK_MISS`:')
    assert 'otherach' in sentence
    assert 'axis fitted and measured' in sentence
    assert sentence.endswith('.')


# -------------------------------------------------------------------------- card


def test_the_card_states_the_five_facts_and_the_label(measured_job):
    card = measured_job['card']
    assert '`measured_artifact_class`' in card
    assert '`jdewitt/loop-fixture@base-rev`' in card            # base repo@rev
    assert 'attn 1.5, mlp 0.75' in card                          # applied strength
    assert '`jdewitt/loop-fixture@base-rev` `Q4_K_M`' in card     # artifact class
    assert '`-c 4096 --parallel 4`' in card                      # config
    assert 'Delivered cos' in card and '`1.0`' in card           # delivered cos
    assert 'apache-2.0' in card
    assert 'No inference' in card


def test_the_unmeasured_card_says_so_in_its_first_line(tmp_path, hub):
    result = loop.start_job(FT, 'ft-rev', 'model-Q4_K_M.gguf', 'Q4_K_M',
                            bank_dir=FIXTURE_BANK, hub=hub, root=tmp_path)
    assert result['label'] == 'banked_unmeasured'
    assert result['card'].splitlines()[0].startswith('This file was **not** measured.')


def test_no_card_carries_a_banned_promise(measured_job):
    assert not BANNED.search(measured_job['card'])
    assert not BANNED.search(json.dumps(measured_job['readback']))


def test_the_sidecar_records_the_label_and_its_source(measured_job):
    sidecar = loop.sidecar_document(measured_job)
    assert sidecar['schema'] == 'jdewitt-space-sidecar-v1'
    assert sidecar['label'] == 'measured_artifact_class'
    assert 'measurement.state' in sidecar['label_source']
    assert sidecar['facts']['delivered_cos'] is not None
    assert sidecar['claims']['removal']


# -------------------------------------------------------------------------- loop


def test_the_loop_runs_every_step_in_order(measured_job):
    assert [step['step'] for step in measured_job['steps']] == [
        'coverage', 'licence', 'download', 'edit', 'readback', 'package']
    assert all(step['state'] == 'ok' for step in measured_job['steps'])


def test_the_loop_delivers_four_files_with_matching_digests(measured_job):
    files = measured_job['files']
    paths = [Path(files[key]) for key in ('artifact', 'card', 'sidecar', 'sha256sums')]
    assert all(path.is_file() for path in paths)
    lines = Path(files['sha256sums']).read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        digest, name = line.split()
        assert hashlib.sha256((Path(files['artifact']).parent / name).read_bytes()).hexdigest() == digest


def test_the_job_workspace_is_deleted_and_only_the_deliverable_survives(measured_job):
    assert measured_job['workspace_deleted'] is True
    root = Path(measured_job['files']['artifact']).parents[1]
    assert not [entry for entry in root.iterdir() if entry.name.startswith('job-')]


def test_the_readback_reopened_the_output(tmp_path, hub):
    result = loop.start_job(BASE, 'base-rev', 'model-Q4_K_M.gguf', 'Q4_K_M',
                            bank_dir=FIXTURE_BANK, hub=hub, root=tmp_path)
    readback = result['readback']
    assert readback['reopened']['tensors'] == 6
    assert readback['reopened']['shapes_match'] is True
    assert readback['untouched']['verified'] is True
    assert readback['output_sha256'] == hashlib.sha256(
        Path(result['files']['artifact']).read_bytes()).hexdigest()
    assert readback['bytes_out'] > readback['bytes_in']  # q4_k input, f16 edit


def test_an_uncovered_architecture_is_refused_before_the_download(tmp_path, hub):
    result = loop.start_job(OTHER, 'other-rev', 'model-Q4_K_M.gguf', 'Q4_K_M',
                            bank_dir=FIXTURE_BANK, hub=hub, root=tmp_path)
    assert result['label'] == 'refused'
    assert result['refusal']['code'] == 'BANK_MISS'
    assert [step['step'] for step in result['steps']] == ['coverage']
    assert result['files'] is None
    assert 'otherach' in result['refusal']['sentence']
    assert 'axis fitted and measured' in result['refusal']['sentence']
    assert result['workspace_deleted'] is True


def test_a_quant_the_entry_does_not_cover_is_refused_before_the_download(tmp_path, hub):
    """The disabled file stays pickable; picking it is refused by code, not hidden."""
    result = loop.start_job(BASE, 'base-rev', 'model-Q8_0.gguf', 'Q8_0',
                            bank_dir=FIXTURE_BANK, hub=hub, root=tmp_path)
    assert result['label'] == 'refused'
    assert result['refusal']['code'] == 'UNSUPPORTED_QUANT'
    assert 'q8_0' in result['refusal']['message'].casefold()
    assert 'download' not in [step['step'] for step in result['steps']]


def test_the_live_header_read_is_sparse_and_bounded(tmp_path, monkeypatch):
    """The Hub seam: a prefix is read, the copy is padded, no tensor bytes move."""
    import huggingface_hub

    source = (FIXTURE / 'repos' / 'jdewitt__loop-fixture' / 'model-Q4_K_M.gguf').read_bytes()

    class FakeFileSystem:
        def open(self, path, mode):
            import io
            return io.BytesIO(source)

    monkeypatch.setattr(huggingface_hub, 'HfFileSystem', FakeFileSystem)
    dest = tmp_path / 'header.gguf'
    written = loop.LiveHub().header(BASE, 'model-Q4_K_M.gguf', 'base-rev', dest, size=len(source))
    assert written == len(source) < loop.HEADER_READ_CAP
    assert dest.stat().st_size == len(source)
    assert dest.read_bytes() == source  # the fixture is smaller than the cap


def test_the_live_download_moves_the_file_after_the_gates(tmp_path, monkeypatch):
    import huggingface_hub

    source = FIXTURE / 'repos' / 'jdewitt__loop-fixture' / 'model-Q4_K_M.gguf'

    def fake_download(repo_id, filename, revision=None, local_dir=None):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        return str(target)

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fake_download)
    dest = tmp_path / 'input.gguf'
    loop.LiveHub().download(BASE, 'model-Q4_K_M.gguf', 'base-rev', dest)
    assert dest.read_bytes() == source.read_bytes()


def test_an_unlisted_licence_is_refused_before_the_download(tmp_path):
    def mutate(document):
        document['repos'][BASE]['licence'] = 'some-proprietary-licence'

    fixture = _fixture_copy(tmp_path, mutate)
    result = loop.start_job(BASE, 'base-rev', 'model-Q4_K_M.gguf', 'Q4_K_M',
                            bank_dir=FIXTURE_BANK, hub=loop.FixtureHub(fixture), root=tmp_path)
    assert result['refusal']['code'] == 'LICENCE_UNKNOWN'
    assert 'some-proprietary-licence' in result['refusal']['message']
    assert 'download' not in [step['step'] for step in result['steps']]


def test_a_gated_repo_is_refused_before_the_download(tmp_path):
    def mutate(document):
        document['repos'][BASE]['gated'] = True

    fixture = _fixture_copy(tmp_path, mutate)
    result = loop.start_job(BASE, 'base-rev', 'model-Q4_K_M.gguf', 'Q4_K_M',
                            bank_dir=FIXTURE_BANK, hub=loop.FixtureHub(fixture), root=tmp_path)
    assert result['refusal']['code'] == 'GATED_REPO'
    assert 'anonymous Space cannot authenticate' in result['refusal']['sentence']


def test_a_second_job_is_refused_while_the_lock_is_held(tmp_path):
    (tmp_path / '.job.lock').write_text('1234')
    result = loop.start_job(BASE, 'base-rev', 'model-Q4_K_M.gguf', 'Q4_K_M',
                            bank_dir=FIXTURE_BANK, hub=loop.FixtureHub(FIXTURE), root=tmp_path)
    assert result['refusal']['code'] == 'JOB_RUNNING'
    assert 'one job at a time' in result['refusal']['message']


def test_a_stale_lock_is_taken_over(tmp_path):
    lock = tmp_path / '.job.lock'
    lock.write_text('1234')
    old = 1
    import os
    os.utime(lock, (old, old))
    result = loop.start_job(BASE, 'base-rev', 'model-Q4_K_M.gguf', 'Q4_K_M',
                            bank_dir=FIXTURE_BANK, hub=loop.FixtureHub(FIXTURE), root=tmp_path)
    assert result['label'] == 'measured_artifact_class'


# ------------------------------------------------------------------ quant control


@pytest.mark.parametrize('name, expected', [
    ('model-Q4_K_M.gguf', 'Q4_K_M'),
    ('Qwen3-4B-Instruct-2507-Q8_0.gguf', 'Q8_0'),
    ('model-F16.gguf', 'F16'),
    ('model-IQ4_XS.gguf', 'IQ4_XS'),
    ('model.gguf', None),
])
def test_the_quant_tag_is_read_from_the_file_name(name, expected):
    assert loop.quant_tag(name) == expected


def test_a_quant_outside_the_entry_is_listed_with_its_reason(hub):
    entry = loop.load_bank(FIXTURE_BANK)['entries'][0]
    choices = loop.quant_choices(hub.files(BASE), entry)
    assert [choice['name'] for choice in choices] == ['model-Q4_K_M.gguf', 'model-Q8_0.gguf']
    assert choices[0]['enabled'] is True and choices[0]['reason'] is None
    assert choices[1]['enabled'] is False
    assert 'not in this entry\'s covered source types (q4_k)' in choices[1]['label']


def test_preview_reports_the_licence_line_without_reading_a_header(hub):
    view = loop.preview(BASE, 'base-rev', bank_dir=FIXTURE_BANK, hub=hub)
    assert view['ok'] is True
    assert view['licence_line'] == 'Licence: `apache-2.0` — LICENCE_ACCEPTED'
    assert 'measurement state is `measured_artifact_class`' in view['coverage_line']
    assert view['files']


# ---------------------------------------------------------------------- the page


def test_the_panel_puts_the_caveat_above_the_download_control(measured_job):
    sections = loop.panel_sections(measured_job)
    order = [section['id'] for section in sections]
    assert order == list(loop.PANEL_ORDER)
    assert order.index('caveat') < order.index('start') < order.index('file')


def test_the_app_has_one_start_button_and_no_strength_layer_or_method_control():
    source = (REPO / 'spaces' / 'app.py').read_text()
    assert source.count("gr.Button('Start'") == 1
    for forbidden in ('gr.Slider', 'gr.Radio', 'gr.Number'):
        assert forbidden not in source
    assert 'strength' not in source.casefold().replace('no strength', '')


def test_the_app_renders_the_caveat_above_the_file_control():
    source = (REPO / 'spaces' / 'app.py').read_text()
    caveat = source.index('caveat = gr.Markdown')
    start = source.index("run = gr.Button('Start'")
    artifact = source.index('artifact = gr.File')
    assert caveat < start < artifact


def test_the_app_builds_under_the_pinned_gradio():
    """Skipped unless Gradio is installed; run once with gradio 5.50 (the SDK pin)."""
    pytest.importorskip('gradio')
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location('space_app_under_test', REPO / 'spaces' / 'app.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.demo is not None
    update, verdict, _licence = module.refresh('not a repo')
    assert update['__type__'] == 'update'
    assert verdict.startswith('`INPUT_ERROR`:')
    assert module.start('owner/name', '') == (
        'Choose a .gguf file from the list first.', None, None, 'Nothing ran.')


def test_the_app_and_the_loop_carry_no_banned_promise():
    hits = []
    for path in sorted((REPO / 'spaces').rglob('*')) + [REPO / 'jdewitt' / 'space_loop.py']:
        if path.is_file() and path.suffix in ('.py', '.md', '.json'):
            text = path.read_text(errors='ignore')
            hits += [f'{path}: {match.group(0)}' for match in BANNED.finditer(text)]
    assert hits == []


# --------------------------------------------------------------------- the dry run


def test_the_dry_run_renders_every_state_and_the_refusal_without_network():
    report = loop.dry_run(BANK, BASE, 'Q4_K_M', FIXTURE)
    assert report['ok'] is True
    assert report['labels'] == ['measured_artifact_class', 'banked_unmeasured', 'refused']
    assert report['panel_order'] == list(loop.PANEL_ORDER)
    assert report['caveat_above_file'] is True
    assert report['network'].startswith('none')
    assert all(case['result']['workspace_deleted'] for case in report['cases'])
    text = loop.render_dry_run(report)
    assert 'Caveat above the download control: True' in text
    assert 'This file was **not** measured.' in text
    assert '`BANK_MISS`' in text
    assert 'DISABLED' in text and 'not in this entry\'s covered source types' in text
    assert not BANNED.search(text)


def test_the_cli_dry_run_exits_zero_and_prints_the_three_states():
    result = subprocess.run(
        [sys.executable, '-m', 'jdewitt', 'space-dry-run', '--bank', 'spaces/bank',
         '--model', BASE, '--quant', 'Q4_K_M', '--fixture', str(FIXTURE)],
        capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr
    for label in ('measured_artifact_class', 'banked_unmeasured', 'refused'):
        assert label in result.stdout
    assert 'Caveat above the download control: True' in result.stdout
    assert 'Dry run passed: True' in result.stdout
