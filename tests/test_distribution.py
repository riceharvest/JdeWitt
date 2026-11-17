"""Exercise the built distribution, not editable-install resource paths."""
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def test_wheel_contains_runtime_judge_assets(tmp_path):
    root=Path(__file__).resolve().parents[1]
    subprocess.run(['uv','build','--wheel','--out-dir',str(tmp_path/'dist')],cwd=root,check=True,capture_output=True,text=True)
    wheel,=(tmp_path/'dist').glob('*.whl')
    required={'jdewitt/assets/'+name for name in (
        'rubric-system.txt','evidence-system.txt','style-system.txt','judge-calibration.json')}
    with zipfile.ZipFile(wheel) as archive:
        assert required <= set(archive.namelist()), 'Built wheel drops runtime judge assets'
        archive.extractall(tmp_path/'installed')
    env=dict(os.environ,PYTHONPATH=str(tmp_path/'installed'))
    result=subprocess.run([sys.executable,'-c',
        'import jdewitt.grounded_judge as m; '
        'assert m.RUBRIC_SYSTEM and m.EVIDENCE_SYSTEM; '
        'print(m.__file__)'],cwd=tmp_path,env=env,text=True,capture_output=True,check=True)
    assert str(tmp_path/'installed') in result.stdout
