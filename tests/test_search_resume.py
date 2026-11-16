"""Exercise real Optuna persistence with explicit deterministic objective fixtures."""
import optuna
import pytest
from b70_ablator.search import build_study, execute_trials


@pytest.mark.parametrize('startup', [20, 2])
def test_restart_matches_uninterrupted_future_suggestions(tmp_path, startup):
    def objective(t):
        x = t.suggest_float('x', 0., 1.)
        y = t.suggest_float('y', .01, 1., log=True)
        return x + y
    identity = {'fixture': 'sampler-continuity'}
    continuous = build_study(tmp_path/'continuous.db', identity, 7, startup)
    execute_trials(continuous, objective, 9)
    resumed = build_study(tmp_path/'resumed.db', identity, 7, startup)
    execute_trials(resumed, objective, 4)
    resumed = build_study(tmp_path/'resumed.db', identity, 7, startup)
    execute_trials(resumed, objective, 9)
    assert [t.params for t in continuous.trials] == [t.params for t in resumed.trials]


def test_restart_half_suggested_trial_preserves_later_suggestions(tmp_path):
    identity = {'fixture': 'partial-sampler'}
    def objective(t):
        return t.suggest_float('x', 0., 1.) + t.suggest_float('y', 0., 1.)
    continuous = build_study(tmp_path/'continuous.db', identity, 7, 20)
    execute_trials(continuous, objective, 5)
    resumed = build_study(tmp_path/'resumed.db', identity, 7, 20)
    execute_trials(resumed, objective, 2)
    def interrupted(t):
        t.suggest_float('x', 0., 1.)
        raise ConnectionError('after first parameter')
    with pytest.raises(ConnectionError):
        execute_trials(resumed, interrupted, 5)
    resumed = build_study(tmp_path/'resumed.db', identity, 7, 20)
    execute_trials(resumed, objective, 5)
    assert [t.params for t in continuous.trials] == [t.params for t in resumed.trials]


def test_sampler_contract_rejects_changed_seed_and_legacy_trials(tmp_path):
    path = tmp_path/'study.db'
    study = build_study(path, {'fixture':'contract'}, 7, 20)
    execute_trials(study, lambda t:t.suggest_float('x', 0., 1.), 1)
    with pytest.raises(ValueError, match='Sampler contract'):
        build_study(path, {'fixture':'contract'}, 8, 20)
    legacy = optuna.create_study(study_name='b70-ablator', storage='sqlite:///'+str(tmp_path/'legacy.db'))
    legacy.optimize(lambda t:t.suggest_float('x', 0., 1.), n_trials=1)
    with pytest.raises(ValueError, match='legacy'):
        build_study(tmp_path/'legacy.db', {'fixture':'legacy'}, 7, 20)


def test_interrupted_attempt_resumes_same_trial_and_same_parameters(tmp_path):
    path=tmp_path/'study.db'
    study=build_study(path,{'fixture':'resume'},7,1)
    seen=[]
    def interrupted(trial):
        strength=trial.suggest_float('fixture_strength',0.,1.)
        seen.append((trial.number,strength))
        raise ConnectionError('fixture judge interrupted')
    with pytest.raises(ConnectionError):
        execute_trials(study,interrupted,1)
    assert len(study.trials)==1 and study.trials[0].state==optuna.trial.TrialState.RUNNING
    resumed=build_study(path,{'fixture':'resume'},7,1)
    def repaired(trial):
        strength=trial.suggest_float('fixture_strength',0.,1.)
        seen.append((trial.number,strength))
        return strength
    execute_trials(resumed,repaired,1)
    assert seen[0]==seen[1]
    assert len(resumed.trials)==1 and resumed.trials[0].state==optuna.trial.TrialState.COMPLETE


def test_pending_warm_starts_do_not_consume_execution_budget(tmp_path):
    study=build_study(tmp_path/'study.db',{'fixture':'warm-start'},1,1)
    for strength in (.2,.4,.6):
        study.enqueue_trial({'fixture_strength':strength})
    execute_trials(study,lambda t:t.suggest_float('fixture_strength',0.,1.),2)
    assert [t.state for t in study.trials]==[optuna.trial.TrialState.COMPLETE,optuna.trial.TrialState.COMPLETE,optuna.trial.TrialState.WAITING]
    assert [t.value for t in study.trials[:2]]==[.2,.4]


def test_rejected_trials_are_completed_as_pruned_and_not_retried(tmp_path):
    study=build_study(tmp_path/'study.db',{'fixture':'pruning'},1,1)
    calls=[]
    def objective(trial):
        calls.append(trial.number)
        raise optuna.TrialPruned('fixture damage gate')
    execute_trials(study,objective,2)
    execute_trials(study,objective,2)
    assert calls==[0,1]
    assert all(t.state==optuna.trial.TrialState.PRUNED for t in study.trials)
