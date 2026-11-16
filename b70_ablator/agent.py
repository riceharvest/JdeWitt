# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only agent operations. No model imports, device access or network calls."""
from collections import Counter
import json
from pathlib import Path
import sqlite3


def validate_run(path):
    from .search import RunConfig
    from .validation import validate_run_inputs
    from .architectures import ARCHITECTURES
    config=RunConfig.from_json(json.loads(Path(path).read_text()))
    _,report=validate_run_inputs(config)
    for field in ('converter','quantizer','converter_python','llama_server'):
        if not Path(getattr(config,field)).is_file():
            raise FileNotFoundError(f'Missing {field}: {getattr(config,field)}')
    if config.imatrix and not Path(config.imatrix).is_file():
        raise FileNotFoundError(config.imatrix)
    source=Path(config.model)
    if source.is_dir():
        model_config=json.loads((source/'config.json').read_text())
        architecture=model_config.get('model_type')
        if architecture not in ARCHITECTURES:
            raise ValueError(f'No architecture adapter for {architecture!r}')
        weights=list(source.glob('*.safetensors'))
        if not weights:
            raise ValueError('Original safetensors are required for streaming export')
        report['source']={'architecture':architecture,'weight_bytes':sum(p.stat().st_size for p in weights)}
    else:
        report['source']={'model':config.model,'remote_metadata_checked':False}
    if config.scoring_mode=='judge':
        for field in ('judge_artifact','calibration'):
            if not Path(getattr(config,field)).is_file():
                raise FileNotFoundError(getattr(config,field))
    return report


def run_status(directory):
    root=Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    trials=[]
    path=root/'study.db'
    if path.exists():
        # mode=ro forbids accidental database creation or Optuna schema migration.
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=1) as connection:
            connection.execute('PRAGMA query_only=ON')
            rows=connection.execute('''SELECT t.number,t.state,v.value,a.value_json FROM trials t
                JOIN studies s ON s.study_id=t.study_id
                LEFT JOIN trial_values v ON v.trial_id=t.trial_id AND v.objective=0
                LEFT JOIN trial_user_attributes a ON a.trial_id=t.trial_id AND a.key='feasible'
                WHERE s.study_name='b70-ablator' ORDER BY t.number''').fetchall()
        trials=[{'number':n,'state':state,'value':value,'feasible':json.loads(feasible) is True if feasible else False}
                for n,state,value,feasible in rows]
    candidates=[t for t in trials if t['state']=='COMPLETE' and t['feasible'] and t['value'] is not None]
    return {'output':str(root),'trial_counts':dict(Counter(t['state'] for t in trials)),
            'trials':trials,'best_feasible':min(candidates,key=lambda t:t['value']) if candidates else None,
            'release_present':(root/'release').is_dir(),'release_verified':False,
            'note':'RUNNING is an unfinished resumable trial, not proof of a live process. Use verify-release for bundle integrity.'}
