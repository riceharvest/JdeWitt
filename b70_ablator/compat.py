# SPDX-License-Identifier: AGPL-3.0-or-later
"""Known vendor ABI fixes in private metadata copies, never the user's originals."""
import json
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .answers import atomic_text


# transformers 5.x only understands a dict-shaped _tied_weights_keys (it calls .keys(),
# .values() and .items() on it), but Spark-X2.6's remote code still ships the 4.x list form
# and dies in post_init with AttributeError: 'list' object has no attribute 'keys'.
# The 4.x list form meant 'tie this target to the input embedding', which here is
# model.embedding.weight -- the same target the self-referential dict fix below produces.
_TIED_LIST=re.compile(r'(?m)^(\s*)_tied_weights_keys\s*=\s*\[\s*"lm_head\.weight"\s*,?\s*\]\s*(.*)$')

# transformers 5.x renamed create_causal_mask's second parameter from input_embeds to
# inputs_embeds, but Spark-X2.6 builds its mask kwargs under the old name.
_MASK_KWARG='"input_embeds": inputs_embeds,'
_MASK_KWARG_FIXED='"inputs_embeds": inputs_embeds,'
# ... and dropped cache_position entirely from both mask factories' signatures; the cache
# length now comes from past_key_values, while the vendor still feeds it in.
_MASK_CACHE_POSITION='"cache_position": cache_position,'


def repair_export_metadata(directory):
    directory=Path(directory)
    config=json.loads((directory/'config.json').read_text())
    if config.get('model_type')!='spark2_5' or not config.get('tie_word_embeddings'):
        return []
    path=directory/'modeling_spark.py'
    source=path.read_text()
    repairs=[]
    broken='{"lm_head.weight": "lm_head.weight"}'
    fixed='{"lm_head.weight": "model.embedding.weight"}'
    if fixed not in source:
        matches=_TIED_LIST.findall(source)
        if len(matches)==1 and broken not in source:
            source=_TIED_LIST.sub(
                lambda m:m.group(1)+'_tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}'+m.group(2),
                source,count=1)
            source='# B70-ablator compatibility fix: declare the tied output head in the dict form ' \
                   'transformers 5.x requires, pointing at the real input embedding.\n'+source
            repairs.append('spark_tied_weights_dict_form')
        elif source.count(broken)==1:
            source=source.replace(broken,fixed)
            source='# B70-ablator compatibility fix: tie the output head to the real input embedding.\n'+source
            repairs.append('spark_tied_embeddings')
        else:
            raise ValueError('Unrecognized Spark tied-weight declaration; compatibility adapter must be reviewed')
    if source.count(_MASK_KWARG)==1 and _MASK_KWARG_FIXED not in source:
        source=source.replace(_MASK_KWARG,_MASK_KWARG_FIXED)
        source='# B70-ablator compatibility fix: transformers 5.x renamed create_causal_mask input_embeds to inputs_embeds.\n'+source
        repairs.append('spark_mask_kwargs_inputs_embeds')
    elif source.count(_MASK_KWARG)>1:
        raise ValueError('Unrecognized Spark causal-mask declaration; compatibility adapter must be reviewed')
    if source.count(_MASK_CACHE_POSITION)==1:
        source=source.replace(_MASK_CACHE_POSITION,'')
        source='# B70-ablator compatibility fix: transformers 5.x mask factories no longer take cache_position.\n'+source
        repairs.append('spark_mask_kwargs_no_cache_position')
    elif source.count(_MASK_CACHE_POSITION)>1:
        raise ValueError('Unrecognized Spark cache_position usage; compatibility adapter must be reviewed')
    if repairs:
        atomic_text(path,source)
    return repairs


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
