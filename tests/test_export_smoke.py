"""export-smoke: the header half and the verdict contract, no llama.cpp needed."""
import json
import subprocess
import sys
from pathlib import Path

from jdewitt.export_smoke import export_smoke
from jdewitt.ggufio import write_gguf
from jdewitt.schemas import EMIT_SCHEMAS, enforce

ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path, name='m.gguf'):
    path = tmp_path / name
    write_gguf(path, {'general.architecture': 'pytestarch', 'general.name': 'pytest'},
               [('t.weight', [4], 0, b'\x00' * 16)])
    return path


def test_header_half_is_measured_but_never_servable(tmp_path):
    artifact = _fixture(tmp_path)
    out = tmp_path / 'smoke.json'
    result = export_smoke(artifact, out=out, run_server=False)
    assert result['artifact']['architecture'] == 'pytestarch'
    assert result['artifact']['tensors'] == 1
    assert result['artifact']['dtypes'] == {'F32': {'tensors': 1, 'bytes': 16}}
    assert result['artifact']['sha256'] and result['artifact']['bytes'] == artifact.stat().st_size
    assert result['argv'] == [] and result['props_identity'] == {} and result['completion'] == {}
    assert result['verdict']['servable'] == 'unmeasured'
    assert json.loads(out.read_text()) == result


def test_truncated_file_fails_with_the_header_error(tmp_path):
    artifact = _fixture(tmp_path)
    truncated = tmp_path / 'truncated.gguf'
    truncated.write_bytes(artifact.read_bytes()[:32])  # magic+version+counts, no KV region
    result = export_smoke(truncated, out=tmp_path / 'smoke.json', run_server=False)
    assert result['verdict']['servable'] is False
    assert 'header unreadable' in result['verdict']['reason']
    assert 'header_error' in result['artifact']


def test_writer_satisfies_the_published_command_schema(tmp_path):
    result = export_smoke(_fixture(tmp_path), out=tmp_path / 'smoke.json', run_server=False)
    assert EMIT_SCHEMAS['export-smoke'] == 'export-smoke.command.json'
    enforce('export-smoke', result)  # raises if the writer drifts from the schema


def test_commands_manifest_lists_export_smoke_with_its_flags():
    manifest = json.loads(subprocess.run([sys.executable, '-m', 'jdewitt', 'commands'], cwd=ROOT,
                                         capture_output=True, text=True, check=True).stdout)
    row = next(item for item in manifest['commands'] if item['command'] == 'export-smoke')
    flags = {flag for argument in row['arguments'] for flag in argument['flags']}
    assert {'--gguf', '--server', '--ctx', '--max-tokens', '--out', '--server-arg', '--no-server'} <= flags
    assert row['schema_enforced'] is True
