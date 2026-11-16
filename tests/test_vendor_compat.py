import json
import stat
import pytest
from b70_ablator.compat import compatible_source, repair_export_metadata
from b70_ablator.release import load_mode_flags


def source(tmp_path):
    root=tmp_path/'original'
    root.mkdir()
    (root/'config.json').write_text(json.dumps({'model_type':'spark2_5','tie_word_embeddings':True}))
    (root/'modeling_spark.py').write_text('class Spark2_5ForCausalLM:\n    _tied_weights_keys = {"lm_head.weight": "lm_head.weight"}\n')
    (root/'model.safetensors').write_bytes(b'fixture for file-link integrity, not model weights')
    return root


def test_compatibility_changes_only_copied_vendor_metadata(tmp_path):
    original=source(tmp_path)
    before=(original/'modeling_spark.py').read_bytes()
    with compatible_source(original) as copied:
        assert copied!=original
        assert '"model.embedding.weight"' in (copied/'modeling_spark.py').read_text()
        assert (copied/'model.safetensors').is_symlink()
        assert (copied/'model.safetensors').resolve()==(original/'model.safetensors').resolve()
        assert (original/'modeling_spark.py').read_bytes()==before
    assert not copied.exists()
    assert original.exists()


def test_exported_vendor_code_is_portable_under_current_transformers(tmp_path):
    original=source(tmp_path)
    result=repair_export_metadata(original)
    assert result==['spark_tied_embeddings']
    assert '"model.embedding.weight"' in (original/'modeling_spark.py').read_text()
    assert repair_export_metadata(original)==[]


def fake_server(tmp_path,help_text,name='llama-server'):
    path=tmp_path/name
    path.write_text('#!/bin/sh\ncat <<\'HELP\'\n'+help_text+'\nHELP\n')
    path.chmod(path.stat().st_mode|stat.S_IEXEC)
    return path


def test_load_mode_follows_the_servers_own_flag_spelling(tmp_path):
    modern=fake_server(tmp_path,'-lm,   --load-mode MODE                 model loading mode (default: auto)\n                                        - none: no special loading mode','llama-server-modern')
    legacy=fake_server(tmp_path,'--no-mmap                               do not memory-map model\n--mlock','llama-server-legacy')
    assert load_mode_flags(modern)==['--load-mode','none']
    assert load_mode_flags(legacy)==['--no-mmap']


def test_load_mode_refuses_to_guess_when_neither_flag_is_advertised(tmp_path):
    with pytest.raises(RuntimeError,match='refusing to guess'):
        load_mode_flags(fake_server(tmp_path,'--model FNAME\n--ctx-size N'))
