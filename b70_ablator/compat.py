# SPDX-License-Identifier: AGPL-3.0-or-later
"""Known vendor ABI fixes in private metadata copies, never the user's originals."""
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


def repair_export_metadata(directory):
    directory=Path(directory)
    config=json.loads((directory/'config.json').read_text())
    if config.get('model_type')!='spark2_5' or not config.get('tie_word_embeddings'):
        return []
    path=directory/'modeling_spark.py'
    source=path.read_text()
    broken='{"lm_head.weight": "lm_head.weight"}'
    fixed='{"lm_head.weight": "model.embedding.weight"}'
    if fixed in source:
        return []
    if source.count(broken)!=1:
        raise ValueError('Unrecognized Spark tied-weight declaration; compatibility adapter must be reviewed')
    source=source.replace(broken,fixed)
    source='# B70-ablator compatibility fix: tie the output head to the real input embedding.\n'+source
    path.write_text(source)
    return ['spark_tied_embeddings']


@contextmanager
def compatible_source(source):
    source=Path(source)
    if not source.is_dir():
        yield source
        return
    config=json.loads((source/'config.json').read_text())
    if config.get('model_type')!='spark2_5' or not config.get('tie_word_embeddings'):
        yield source
        return
    # tempfile honors the caller's TMPDIR; this host points it at managed scratch.
    with tempfile.TemporaryDirectory(prefix='b70-vendor-compat-') as temporary:
        copied=Path(temporary)
        for path in source.iterdir():
            if not path.is_file() or path.name.startswith('.'):
                continue
            if path.suffix in {'.safetensors','.bin','.pt','.pth'}:
                (copied/path.name).symlink_to(path.resolve())
            else:
                shutil.copy2(path,copied/path.name)
        repair_export_metadata(copied)
        yield copied
