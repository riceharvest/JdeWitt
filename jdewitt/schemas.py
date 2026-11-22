# SPDX-License-Identifier: AGPL-3.0-or-later
"""Published JSON Schemas for on-disk artifacts and a dependency-free shape validator.

The schemas describe the versioned contracts this tool writes, so a third party
can check an artifact offline. They mirror the writers in release.py and
answers.py; when a writer version bumps, both the const version and the schema
bump together or nothing is silent.
"""
import json
from pathlib import Path

SCHEMAS = {
    'files.json': {
        'description': 'Integrity index written beside every release bundle.',
        'type': 'object',
        'required': ['version', 'release_identity', 'files', 'record_digest'],
        'properties': {
            'version': {'const': 1},
            'release_identity': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
            'files': {'type': 'object', 'additionalProperties': {
                'type': 'object', 'required': ['sha256', 'bytes'],
                'properties': {'sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
                               'bytes': {'type': 'integer'}}}},
            'record_digest': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}},
    },
    'evidence.json': {
        'description': 'Chained claims ledger of a release.',
        'type': 'object',
        'required': ['version', 'release_identity', 'entries', 'ledger_digest'],
        'properties': {
            'version': {'const': 1},
            'release_identity': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
            'entries': {'type': 'array', 'items': {
                'type': 'object',
                'required': ['seq', 'claim', 'evidence', 'verified_by', 'at', 'previous_digest', 'digest'],
                'properties': {'seq': {'type': 'integer'}, 'claim': {'type': 'string'},
                               'evidence': {'type': 'string'}, 'verified_by': {'type': 'string'},
                               'at': {'type': 'string'}, 'previous_digest': {'type': 'string'},
                               'digest': {'type': 'string'}}}},
            'ledger_digest': {'type': 'string'}},
    },
    'signature.json': {
        'description': 'Ed25519 signature over a release reproduction index.',
        'type': 'object',
        'required': ['version', 'algorithm', 'public_key_pem', 'payload', 'signature_base64'],
        'properties': {'version': {'const': 1}, 'algorithm': {'const': 'ed25519'},
                       'public_key_pem': {'type': 'string'}, 'payload': {'type': 'object'},
                       'signature_base64': {'type': 'string'}}},
    'scoring-manifest.json': {
        'description': 'Identity binding a scoring directory to one generation and one scorer.',
        'type': 'object',
        'required': ['version', 'suite_digest', 'generation_digest', 'identity', 'sandbox_image', 'scoring_digest'],
        'properties': {'version': {'const': 2}, 'suite_digest': {'type': 'string'},
                       'generation_digest': {'type': 'string'}, 'identity': {'type': 'object'},
                       'sandbox_image': {'type': 'string'}, 'scoring_digest': {'type': 'string'}}},
    'adapter.json': {
        'description': 'R8 label shipped beside a LoRA adapter: base binding, adapter identity, '
                       'the merged twin fidelity record and BOTH delta_br readings.',
        'type': 'object',
        'required': ['version', 'kind', 'adapter', 'base', 'merged_twin', 'adapter_delta_br',
                     'merged_delta_br', 'adapter_to_merged_ratio', 'ranking_valid_on_this_artifact',
                     'ranking_validity_ratio_floor', 'merged_effect_status',
                     'curve_is_not_merged_bytes', 'r8_note', 'claim_limit'],
        'properties': {
            'version': {'const': 1},
            'kind': {'const': 'adapter-card'},
            'adapter': {'type': 'object',
                        'required': ['sha256', 'bytes', 'rank', 'tensor_entries',
                                     'edited_tensors', 'scale', 'bound_base_sha256'],
                        'properties': {'sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
                                       'bound_base_sha256': {'type': 'string'}},
                        'additionalProperties': True},
            'base': {'type': 'object',
                     'required': ['sha256', 'arch', 'tensor_count'],
                     'properties': {'sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
                                    'arch': {'type': 'string'},
                                    'tensor_count': {'type': 'integer'}}},
            'merged_twin': {'type': 'object',
                            'required': ['artifact_sha256', 'edited_tensors', 'untouched_tensors',
                                         'edit_fingerprint', 'delivered'],
                            'properties': {'artifact_sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
                                           'edited_tensors': {'type': 'integer'},
                                           'untouched_tensors': {'type': 'integer'},
                                           'delivered': {'type': 'object', 'required': ['tensors']}}},
            'adapter_delta_br': {'type': 'number', 'nullable': True},
            'merged_delta_br': {'type': 'number', 'nullable': True},
            'adapter_to_merged_ratio': {'type': 'number', 'nullable': True},
            'ranking_valid_on_this_artifact': {'type': 'boolean'},
            'ranking_validity_ratio_floor': {'type': 'number'},
            'merged_effect_status': {'type': 'string'},
            'curve_is_not_merged_bytes': {'const': True},
            'r8_note': {'type': 'string'},
            'claim_limit': {'type': 'string'}}},
    'packet.json': {
        'description': 'Phase 7 certification packet: one arm, both axes, one serve config each, '
                       'built from saved waves with every removal number carrying both wording '
                       'inventories and its shape verdicts.',
        'type': 'object',
        'required': ['schema_version', 'kind', 'lineage', 'arm', 'artifact_sha256', 'base_sha256',
                     'artifact_identity_digest', 'removal', 'shape', 'retention',
                     'held_out_families', 'drift', 'realized_fidelity', 'configs',
                     'pinned_cert_argv', 'pinned_cert_match', 'next_command', 'measured'],
        'properties': {
            'schema_version': {'const': 1},
            'kind': {'const': 'release-packet'},
            'lineage': {'type': 'string'},
            'arm': {'type': 'object',
                    'required': ['edited_tensors', 'chosen_type', 'realized_cos', 'byte_cost'],
                    'properties': {'edited_tensors': {'type': 'integer'},
                                   'chosen_type': {'type': 'string'},
                                   'realized_cos': {'type': 'number'},
                                   'byte_cost': {'type': 'integer', 'nullable': True},
                                   'label': {'type': 'string'}}},
            'artifact_sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$', 'nullable': True},
            'base_sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$', 'nullable': True},
            'artifact_identity_digest': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
            'removal': {'type': 'object', 'required': ['harmful40', 'held_out_120']},
            'shape': {'type': 'object', 'required': ['audits', 'verdict_counts']},
            'retention': {'type': 'object', 'required': ['math', 'code'],
                          'properties': {'math': {'type': 'object',
                                                  'required': ['strict', 'paired_sign_flip_p_two_sided',
                                                               'median_answer_chars',
                                                               'base_median_answer_chars']},
                                         'code': {'type': 'object',
                                                  'required': ['strict', 'paired_sign_flip_p_two_sided',
                                                               'median_answer_chars',
                                                               'base_median_answer_chars']}}},
            'held_out_families': {'type': 'integer'},
            'drift': {'type': 'object', 'required': ['control_delta_ln_ppl', 'budget', 'inside_budget']},
            'realized_fidelity': {'type': 'object', 'required': ['min_cos', 'mean_cos', 'sampled']},
            'configs': {'type': 'object', 'required': ['removal', 'retention', 'held_out']},
            'measured_axes': {'type': 'object'},
            'pinned_cert_argv': {'type': 'string'},
            'pinned_cert_match': {'type': 'object',
                                  'required': ['removal', 'retention', 'held_out'],
                                  'properties': {'removal': {'type': 'boolean'},
                                                 'retention': {'type': 'boolean'},
                                                 'held_out': {'type': 'boolean'}}},
            'next_command': {'type': 'string'},
            'measured': {'const': True}},
    },
    'matrix.json': {
        'description': 'Phase 7 release matrix: one row per registered lineage with a closed-vocabulary '
                       'status, every packet number, the loss flags and the recompute command, and a '
                       'complete flag over the five required lineages.',
        'type': 'object',
        'required': ['schema_version', 'kind', 'manifest', 'registered', 'complete', 'certified',
                     'counts', 'rows', 'losses', 'refusals', 'next_command', 'note'],
        'properties': {
            'schema_version': {'const': 1},
            'kind': {'const': 'release-matrix'},
            'manifest': {'type': 'string'},
            'registered': {'type': 'object'},
            'complete': {'type': 'boolean'},
            'certified': {'type': 'integer'},
            'counts': {'type': 'object'},
            'rows': {'type': 'array'},
            'spares': {'type': 'array'},
            'loss_ledger': {'type': 'object'},
            'losses': {'type': 'object'},
            'head_to_head': {'type': 'array'},
            'refusals': {'type': 'array'},
            'valid': {'type': 'boolean'},
            'next_command': {'type': 'string'},
            'note': {'type': 'string'}},
    },
    'twin.json': {
        'description': 'Codec-realized record of the merged twin shipped beside the adapter: '
                       'artifact identity, edited/untouched counts, the edit fingerprint, the '
                       'per-tensor delivered cos/SNR with the codec type that realized them, and '
                       'a ledger that says which numbers are codec-realized and which are '
                       'behaviour claims (and where each was measured).',
        'type': 'object',
        'required': ['version', 'kind', 'artifact', 'base', 'tensor_count', 'edited_tensors',
                     'untouched_tensors', 'edit_fingerprint', 'edit_type_per_tensor', 'delivered',
                     'behaviour', 'measurement_ledger'],
        'properties': {
            'version': {'const': 1},
            'kind': {'const': 'merged-twin'},
            'artifact': {'type': 'object', 'required': ['file', 'sha256', 'bytes'],
                         'properties': {'file': {'type': 'string'},
                                        'sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
                                        'bytes': {'type': 'integer'}}},
            'base': {'type': 'object', 'required': ['sha256'],
                     'properties': {'sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}}},
            'tensor_count': {'type': 'integer'},
            'edited_tensors': {'type': 'integer'},
            'untouched_tensors': {'type': 'integer'},
            'edit_fingerprint': {'type': 'object'},
            'edit_type_per_tensor': {'type': 'object',
                                     'additionalProperties': {'type': 'string'}},
            'delivered': {
                'type': 'object',
                'required': ['measurement_class', 'measured_where', 'tensors'],
                'properties': {
                    'measurement_class': {'const': 'codec-realized'},
                    'measured_where': {'type': 'string'},
                    'tensors': {'type': 'array', 'items': {
                        'type': 'object',
                        'required': ['name', 'cos', 'edited_type', 'measurement_class'],
                        'properties': {'name': {'type': 'string'},
                                       'cos': {'type': 'number'},
                                       'snr': {'type': 'number', 'nullable': True},
                                       'edited_type': {'type': 'string'},
                                       'edit': {'type': 'object'},
                                       'measurement_class': {'const': 'codec-realized'}}}}}},
            'behaviour': {
                'type': 'object',
                'required': ['status', 'measurement_class', 'measured_where', 'serve_argv',
                             'argv_recorded', 'curve_is_not_merged_bytes', 'adapter_delta_br',
                             'merged_delta_br', 'ranking_valid_on_this_artifact'],
                'properties': {'status': {'type': 'string'},
                               'measurement_class': {'const': 'behaviour-claim'},
                               'measured_where': {'type': 'string'},
                               'serve_argv': {'type': 'array', 'nullable': True,
                                              'items': {'type': 'string'}},
                               'argv_recorded': {'type': 'boolean'},
                               'curve_is_not_merged_bytes': {'const': True},
                               'adapter_delta_br': {'type': 'number', 'nullable': True},
                               'merged_delta_br': {'type': 'number', 'nullable': True},
                               'adapter_to_merged_ratio': {'type': 'number', 'nullable': True},
                               'ranking_valid_on_this_artifact': {'type': 'boolean'},
                               'ranking_validity_ratio_floor': {'type': 'number'}}},
            'measurement_ledger': {
                'type': 'array',
                'items': {'type': 'object', 'required': ['numbers', 'class', 'basis', 'where'],
                          'properties': {'numbers': {'type': 'array', 'items': {'type': 'string'}},
                                         'class': {'type': 'string'},
                                         'basis': {'type': 'string'},
                                         'where': {'type': 'string'}}}}}},
    'plan-card.json': {
        'description': 'Manifest of the digest-pinned plan bundle shipped as plan-card.tar '
                       '(jdewitt plan-bundle): one key, the signed meta and the payload digest. '
                       'The format const mirrors jdewitt.registry.BUNDLE_FORMAT.',
        'type': 'object',
        'required': ['format', 'key', 'meta', 'payload_sha256'],
        'properties': {
            'format': {'const': 'b70ablator-plan-bundle-v1'},
            'key': {'type': 'string'},
            'payload_sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
            'meta': {'type': 'object', 'required': ['meta_digest', 'payload_sha256'],
                     'properties': {
                         'meta_digest': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
                         'payload_sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}}}}},
    'run-status.command.json': {
        'description': 'run-status output contract.',
        'type': 'object',
        'required': ['output', 'trial_counts', 'trials', 'cancel_requested', 'release_present', 'note'],
        'properties': {'output': {'type': 'string'}, 'trial_counts': {'type': 'object'},
                       'trials': {'type': 'array'}, 'note': {'type': 'string'}}},
    'studies.command.json': {
        'description': 'studies output contract.',
        'type': 'object',
        'required': ['root', 'study_count', 'studies', 'note'],
        'properties': {'root': {'type': 'string'}, 'study_count': {'type': 'integer'},
                       'studies': {'type': 'array'}, 'note': {'type': 'string'}}},
    'describe-run.command.json': {
        'description': 'describe-run output contract.',
        'type': 'object',
        'required': ['config', 'status', 'suites', 'budget_estimates_upper_bounds', 'note'],
        'properties': {'config': {'type': 'string'}, 'status': {'type': 'string'},
                       'suites': {'type': 'object'}, 'split_family_collisions': {'type': 'array'},
                       'budget_estimates_upper_bounds': {'type': 'object'}, 'note': {'type': 'string'}}},
    'validate-auto.command.json': {
        'description': 'validate-auto / abliterate dry-run output contract.',
        'type': 'object',
        'required': ['status', 'trial_budget', 'candidates', 'models_loaded', 'output_created'],
        'properties': {'status': {'const': 'metadata-valid'}, 'trial_budget': {'type': 'integer'},
                       'candidates': {'type': 'array'}, 'holdout_policy': {'type': 'string'}}},
    'methods.command.json': {
        'description': 'methods output contract.',
        'type': 'object',
        'required': ['scope', 'methods', 'edit_surface']},
    'architectures.command.json': {
        'description': 'architectures output contract.',
        'type': 'object',
        'required': ['scope', 'architectures']},
    'score-compare.command.json': {
        'description': 'score-compare output contract.',
        'type': 'object',
        'required': ['suite', 'before', 'after', 'pass_on', 'shared_prompts', 'flip_count',
                     'flips', 'list_cap', 'before_summary', 'after_summary', 'note'],
        'properties': {'effect_size_cohens_h': {'type': 'number'}, 'suite': {'type': 'string'}, 'before': {'type': 'string'},
                       'after': {'type': 'string'}, 'pass_on': {'type': 'string'},
                       'shared_prompts': {'type': 'integer'}, 'flip_count': {'type': 'integer'},
                       'flips': {'type': 'array'}, 'list_cap': {'type': 'integer'},
                       'before_summary': {'type': 'object'}, 'after_summary': {'type': 'object'},
                       'family_bootstrap': {'type': 'object'}, 'paired_permutation': {'type': 'object'},
                       'note': {'type': 'string'}}},
    'judge-ensemble.command.json': {
        'description': 'judge-ensemble output contract.',
        'type': 'object',
        'required': ['suite', 'directories', 'min_agreement', 'prompts_scored_by_any_judge',
                     'consensus_count', 'contested_count', 'consensus', 'contested', 'note'],
        'properties': {'suite': {'type': 'string'}, 'directories': {'type': 'object'},
                       'consensus_count': {'type': 'integer'}, 'contested_count': {'type': 'integer'},
                       'consensus': {'type': 'array'}, 'contested': {'type': 'array'},
                       'note': {'type': 'string'}}},
    'calibration-report.command.json': {
        'description': 'calibration-report output contract.',
        'type': 'object',
        'required': ['scope', 'suite', 'rows_in_suite', 'rows_with_consensus', 'per_judge',
                     'pairwise_agreement', 'note'],
        'properties': {'scope': {'type': 'string'}, 'suite': {'type': 'string'},
                       'rows_in_suite': {'type': 'integer'}, 'rows_with_consensus': {'type': 'integer'},
                       'per_judge': {'type': 'object'}, 'pairwise_agreement': {'type': 'object'},
                       'authored_ledger': {'type': 'object'}, 'note': {'type': 'string'}}},
    'direction-export.command.json': {
        'description': 'direction-export output contract.',
        'type': 'object',
        'required': ['output', 'plan_key', 'format', 'tensors', 'metadata_entries',
                     'data_base', 'file_size', 'note'],
        'properties': {'output': {'type': 'string'}, 'plan_key': {'type': 'string'},
                       'format': {'type': 'string', 'const': 'direction-gguf-v1'},
                       'tensors': {'type': 'integer'}, 'metadata_entries': {'type': 'integer'},
                       'data_base': {'type': 'integer'}, 'file_size': {'type': 'integer'},
                       'note': {'type': 'string'}}},
    'gguf-delta-pack.command.json': {
        'description': 'gguf-delta-pack output contract.',
        'type': 'object',
        'required': ['output', 'tensors', 'metadata_entries', 'data_base', 'file_size'],
        'properties': {'output': {'type': 'string'}, 'tensors': {'type': 'integer'},
                       'metadata_entries': {'type': 'integer'},
                       'data_base': {'type': 'integer'}, 'file_size': {'type': 'integer'}}},
    'suites-compare.command.json': {
        'description': 'suites-compare output contract.',
        'type': 'object',
        'required': ['families', 'tests_in_family', 'holm_min_adjusted_p', 'note'],
        'properties': {'families': {'type': 'array'}, 'tests_in_family': {'type': 'integer'},
                       'holm_min_adjusted_p': {'type': 'number'}, 'note': {'type': 'string'}}},
    'studies-compare.command.json': {
        'description': 'studies-compare output contract.',
        'type': 'object',
        'required': ['databases', 'param_stats', 'note'],
        'properties': {'databases': {'type': 'array'}, 'param_stats': {'type': 'object'},
                       'note': {'type': 'string'}}},
    'quants-audit.command.json': {
        'description': 'quants-audit output contract.',
        'type': 'object',
        'required': ['file', 'dtypes_audited', 'per_dtype', 'decode_only_dtypes',
                     'port_decode_only_dtypes', 'unsupported_dtypes', 'note'],
        'properties': {'file': {'type': 'string'}, 'dtypes_audited': {'type': 'array'},
                       'per_dtype': {'type': 'object'}, 'decode_only_dtypes': {'type': 'object'},
                       'port_decode_only_dtypes': {'type': 'object'},
                       'unsupported_dtypes': {'type': 'object'}, 'note': {'type': 'string'}}},
    'export-smoke.command.json': {
        'description': 'export-smoke output contract: one artifact, the recorded serve argv, '
                       'what the runtime returned and the serve verdict.',
        'type': 'object',
        'required': ['artifact', 'argv', 'props_identity', 'completion', 'verdict', 'cost_seconds'],
        'properties': {
            'artifact': {'type': 'object',
                         'required': ['path', 'sha256', 'bytes', 'architecture', 'tensors', 'dtypes'],
                         'properties': {'path': {'type': 'string'},
                                        'sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
                                        'bytes': {'type': 'integer'},
                                        'architecture': {'type': 'string'},
                                        # Unreadable headers (truncated files) are a
                                        # legitimate verdict, so these stay untyped.
                                        'tensors': {}, 'dtypes': {'type': 'object'}}},
            'argv': {'type': 'array', 'items': {'type': 'string'}},
            'props_identity': {'type': 'object'},
            'completion': {'type': 'object'},
            'verdict': {'type': 'object', 'required': ['servable', 'reason'],
                        'properties': {'servable': {}, 'reason': {'type': 'string'}}},
            'cost_seconds': {'type': 'number'}}},
    'verdict-audit.command.json': {
        'description': 'verdict-audit output contract.',
        'type': 'object',
        'required': ['directory', 'rows_total', 'compared', 'agreed', 'agreement_rate',
                     'skipped_states', 'dangerous_disagreements', 'dangerous_count',
                     'other_disagreement_kinds', 'other_disagreement_examples',
                     'check_method', 'note'],
        'properties': {'directory': {'type': 'string'}, 'rows_total': {'type': 'integer'},
                       'compared': {'type': 'integer'}, 'agreed': {'type': 'integer'},
                       'skipped_states': {'type': 'object'},
                       'dangerous_disagreements': {'type': 'array'}, 'dangerous_count': {'type': 'integer'},
                       'other_disagreement_kinds': {'type': 'object'},
                       'other_disagreement_examples': {'type': 'array'},
                       'check_method': {'type': 'string',
                                        'const': 'explicit-wording-v1'},
                       'note': {'type': 'string'}}},
    'suite-stats.command.json': {
        'description': 'suite-stats output contract.',
        'type': 'object',
        'required': ['suite', 'prompts', 'kinds', 'languages', 'families', 'note'],
        'properties': {'suite': {'type': 'string'}, 'prompts': {'type': 'integer'},
                       'kinds': {'type': 'object'}, 'languages': {'type': 'object'},
                       'families': {'type': 'object'}}},
    'suite-diff.command.json': {
        'description': 'suite-diff output contract.',
        'type': 'object',
        'required': ['before', 'after', 'digest_before', 'digest_after', 'counts', 'added',
                     'removed', 'changed', 'note'],
        'properties': {'before': {'type': 'string'}, 'after': {'type': 'string'},
                       'digest_before': {'type': 'string'}, 'digest_after': {'type': 'string'},
                       'counts': {'type': 'object'}, 'added': {'type': 'array'},
                       'removed': {'type': 'array'}, 'changed': {'type': 'array'}}},
    'study-metrics.command.json': {
        'description': 'study-metrics output contract.',
        'type': 'object',
        'required': ['study', 'trials', 'best', 'state_counts', 'note'],
        'properties': {'study': {'type': 'string'}, 'trials': {'type': 'integer'},
                       'best': {'type': 'object'}, 'state_counts': {'type': 'object'},
                       'note': {'type': 'string'}}},
    'family-decay.command.json': {
        'description': 'family-decay output contract.',
        'type': 'object',
        'required': ['fitted', 'held', 'global_decay', 'novel_families', 'shared_families', 'note'],
        'properties': {'fitted': {'type': 'object'}, 'held': {'type': 'object'},
                       'novel_families': {'type': 'array'}, 'shared_families': {'type': 'array'},
                       'warning': {'type': 'string'}, 'note': {'type': 'string'}}},
    'run-events.command.json': {
        'description': 'run-events output contract.',
        'type': 'object', 'required': ['events'],
        'properties': {'events': {'type': 'array', 'items': {'type': 'object', 'required': ['kind']}}}},
    'plan-list.command.json': {
        'description': 'plan-list output contract (array of plan rows).',
        'type': 'array', 'items': {'type': 'object', 'required': ['key', 'measured']}},
    'hub-census.command.json': {
        'description': 'hub-census output contract.',
        'type': 'object',
        'required': ['scope', 'window', 'census_at', 'quarter', 'repositories_checked',
                     'model_types_seen', 'adapter_worklist', 'untyped_repositories',
                     'supported_model_types', 'unresolved_gaps', 'author_errors',
                     'derived_artifact_repositories'],
        'properties': {'scope': {'type': 'string'}, 'window': {'type': 'object'},
                       'census_at': {'type': 'string', 'pattern': r'^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'},
                       'quarter': {'type': 'string', 'pattern': r'^[0-9]{4}-Q[1-4]$'},
                       'repositories_checked': {'type': 'integer'},
                       'derived_artifact_repositories': {'type': 'integer'},
                       'model_types_seen': {'type': 'object'},
                       'adapter_worklist': {'type': 'array', 'items': {
                           'type': 'object',
                           'required': ['model_type', 'downloads', 'example_models'],
                           'properties': {'model_type': {'type': 'string'},
                                          'downloads': {'type': 'integer'},
                                          'example_models': {'type': 'array'}}}},
                       'untyped_repositories': {'type': 'array', 'items': {
                           'type': 'object',
                           'required': ['label', 'downloads', 'example_models'],
                           'properties': {'label': {'type': 'string'},
                                          'downloads': {'type': 'integer'},
                                          'example_models': {'type': 'array'}}}},
                       'supported_model_types': {'type': 'array'}, 'unresolved_gaps': {'type': 'array'},
                       'author_errors': {'type': 'array'}}},
    'run-ls.command.json': {
        'description': 'run-ls output contract.',
        'type': 'object',
        'required': ['output', 'entries', 'unexpected_entries', 'note'],
        'properties': {'output': {'type': 'string'}, 'entries': {'type': 'array'},
                       'unexpected_entries': {'type': 'array'}, 'note': {'type': 'string'}}},
    'release-diff.command.json': {
        'description': 'release-diff output contract.',
        'type': 'object',
        'required': ['left', 'right', 'release_identities', 'files', 'claims',
                     'ledger_digests', 'signatures', 'note'],
        'properties': {'left': {'type': 'string'}, 'right': {'type': 'string'},
                       'release_identities': {'type': 'object'}, 'files': {'type': 'object'},
                       'claims': {'type': 'object'}, 'ledger_digests': {'type': 'object'},
                       'signatures': {'type': 'object'}, 'note': {'type': 'string'}}},
    'gguf-tensor-diff.command.json': {
        'description': 'gguf-tensor-diff output contract.',
        'type': 'object',
        'required': ['before', 'after', 'tensor_count_before', 'tensor_count_after',
                     'only_in_before', 'only_in_after', 'summary', 'rel_rms',
                     'differing_tensors', 'note'],
        'properties': {'before': {'type': 'string'}, 'after': {'type': 'string'},
                       'tensor_count_before': {'type': 'integer'},
                       'tensor_count_after': {'type': 'integer'},
                       'only_in_before': {'type': 'array'}, 'only_in_after': {'type': 'array'},
                       'summary': {'type': 'object'},
                       'rel_rms': {'type': 'object'},
                       'differing_tensors': {'type': 'array'}, 'note': {'type': 'string'}}},
    'safetensors-diff.command.json': {
        'description': 'safetensors-diff output contract.',
        'type': 'object',
        'required': ['before', 'after', 'tensor_count_before', 'tensor_count_after',
                     'only_in_before', 'only_in_after', 'summary', 'rel_rms',
                     'differing_tensors', 'note'],
        'properties': {'before': {'type': 'string'}, 'after': {'type': 'string'},
                       'tensor_count_before': {'type': 'integer'},
                       'tensor_count_after': {'type': 'integer'},
                       'only_in_before': {'type': 'array'}, 'only_in_after': {'type': 'array'},
                       'summary': {'type': 'object'},
                       'rel_rms': {'type': 'object'},
                       'differing_tensors': {'type': 'array'}, 'note': {'type': 'string'}}},
    'pt-diff.command.json': {
        'description': 'pt-diff output contract.',
        'type': 'object',
        'required': ['before', 'after', 'states', 'summary', 'differing',
                     'missing_in_after', 'missing_in_before', 'note'],
        'properties': {'before': {'type': 'string'}, 'after': {'type': 'string'},
                       'states': {'type': 'object'}, 'summary': {'type': 'object'},
                       'differing': {'type': 'array'},
                       'missing_in_after': {'type': 'array'},
                       'missing_in_before': {'type': 'array'},
                       'note': {'type': 'string'}}},
    'fp8-edit.command.json': {
        'description': 'fp8-edit output contract.',
        'type': 'object',
        'required': ['input', 'output', 'format', 'tensor', 'scales', 'elements',
                     'scale_mode', 'axis', 'block', 'note'],
        'properties': {'input': {'type': 'string'}, 'output': {'type': 'string'},
                       'format': {'type': 'string'}, 'tensor': {'type': 'string'},
                       'scales': {'type': 'string'}, 'elements': {'type': 'integer'},
                       'scale_mode': {'type': 'string'},
                       'axis': {}, 'block': {}, 'note': {'type': 'string'}}},
    'fp4-edit.command.json': {
        'description': 'fp4-edit output contract.',
        'type': 'object',
        'required': ['input', 'output', 'format', 'tensor', 'scales', 'elements',
                     'block', 'global_scale', 'note'],
        'properties': {'input': {'type': 'string'}, 'output': {'type': 'string'},
                       'format': {'type': 'string'}, 'tensor': {'type': 'string'},
                       'scales': {'type': 'string'}, 'elements': {'type': 'integer'},
                       'block': {'type': 'integer'}, 'global_scale': {},
                       'note': {'type': 'string'}}},
    'nf4-edit.command.json': {
        'description': 'nf4-edit output contract.',
        'type': 'object',
        'required': ['input', 'output', 'tensor', 'scales', 'elements', 'block',
                     'scale_dtype', 'note'],
        'properties': {'input': {'type': 'string'}, 'output': {'type': 'string'},
                       'tensor': {'type': 'string'}, 'scales': {'type': 'string'},
                       'elements': {'type': 'integer'}, 'block': {'type': 'integer'},
                       'scale_dtype': {'type': 'string'}, 'note': {'type': 'string'}}},
    'run-doctor.command.json': {
        'description': 'run-doctor output contract.',
        'type': 'object',
        'required': ['output', 'healthy', 'problems', 'trial_counts',
                     'run_lock_held', 'cancel_requested', 'note'],
        'properties': {'output': {'type': 'string'}, 'healthy': {'type': 'boolean'},
                       'problems': {'type': 'array', 'items': {
                           'type': 'object',
                           'required': ['code', 'detail', 'action'],
                           'properties': {'code': {'type': 'string'},
                                          'detail': {'type': 'string'},
                                          'action': {'type': 'string'}}}},
                       'trial_counts': {}, 'run_lock_held': {},
                       'cancel_requested': {'type': 'boolean'},
                       'note': {'type': 'string'}}},
    }

EMIT_SCHEMAS = {
    'run-status': 'run-status.command.json',
    'studies': 'studies.command.json',
    'describe-run': 'describe-run.command.json',
    'validate-auto': 'validate-auto.command.json',
    'abliterate': 'validate-auto.command.json',
    'methods': 'methods.command.json',
    'architectures': 'architectures.command.json',
    'score-compare': 'score-compare.command.json',
    'judge-ensemble': 'judge-ensemble.command.json',
    'calibration-report': 'calibration-report.command.json',
    'verdict-audit': 'verdict-audit.command.json',
    'direction-export': 'direction-export.command.json',
    'gguf-delta-pack': 'gguf-delta-pack.command.json',
    'suites-compare': 'suites-compare.command.json',
    'studies-compare': 'studies-compare.command.json',
    'quants-audit': 'quants-audit.command.json',
    'export-smoke': 'export-smoke.command.json',
    'adapter-card': 'adapter.json',
    'release-certify': 'packet.json',
    'release-matrix': 'matrix.json',
    'suite-stats': 'suite-stats.command.json',
    'suite-diff': 'suite-diff.command.json',
    'study-metrics': 'study-metrics.command.json',
    'family-decay': 'family-decay.command.json',
    'run-events': 'run-events.command.json',
    'plan-list': 'plan-list.command.json',
    'hub-census': 'hub-census.command.json',
    'run-ls': 'run-ls.command.json',
    'run-doctor': 'run-doctor.command.json',
    'release-diff': 'release-diff.command.json',
    'gguf-tensor-diff': 'gguf-tensor-diff.command.json',
    'safetensors-diff': 'safetensors-diff.command.json',
    'pt-diff': 'pt-diff.command.json',
    'fp8-edit': 'fp8-edit.command.json',
    'fp4-edit': 'fp4-edit.command.json',
    'nf4-edit': 'nf4-edit.command.json',
}


def enforce(command, value):
    """Command-output contract enforcement at emit time; raise if the writer drifted."""
    name = EMIT_SCHEMAS[command]
    problems = validate_against(value, SCHEMAS[name])
    if problems:
        raise ValueError(f'{command} output violates published schema {name}: '
                         + '; '.join(problems[:3]) + (f' ({len(problems)} total)' if len(problems) > 3 else ''))


def validate_against(value, schema, path='$'):
    """Small validator covering the subset used above; returns a list of problems."""
    problems = []
    if value is None and schema.get('nullable'):
        return problems
    kind = schema.get('type')
    if kind == 'object':
        if not isinstance(value, dict):
            return [f'{path}: expected object']
        for key in schema.get('required', []):
            if key not in value:
                problems.append(f'{path}.{key}: required and missing')
        for key, item in value.items():
            rule = schema.get('properties', {}).get(key, schema.get('additionalProperties'))
            if rule is not None and not isinstance(rule, bool):
                problems.extend(validate_against(item, rule, f'{path}.{key}'))
    elif kind == 'array':
        if not isinstance(value, list):
            return [f'{path}: expected array']
        rule = schema.get('items')
        if rule is not None:
            for index, item in enumerate(value):
                problems.extend(validate_against(item, rule, f'{path}[{index}]'))
    elif kind == 'string':
        if not isinstance(value, str):
            problems.append(f'{path}: expected string')
        elif 'pattern' in schema:
            import re
            if not re.fullmatch(schema['pattern'], value):
                problems.append(f'{path}: does not match {schema["pattern"]}')
    elif kind == 'integer':
        if type(value) is not int:
            problems.append(f'{path}: expected integer')
    elif kind == 'number':
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f'{path}: expected number')
    elif kind == 'boolean':
        if not isinstance(value, bool):
            problems.append(f'{path}: expected boolean')
    if 'const' in schema and value != schema['const']:
        problems.append(f'{path}: expected constant {schema["const"]!r}, got {value!r}')
    return problems


def infer_schema_name(path):
    path = Path(path)
    if path.name in SCHEMAS:
        return path.name
    # An artifact persisted by a command keeps that command's name on its directory,
    # so `verification/hub-census/2026-11-22.json` is checked without repeating --name.
    emitted = EMIT_SCHEMAS.get(path.parent.name)
    if emitted:
        return emitted
    raise ValueError(f'No published schema for {path.name!r}; available: {sorted(SCHEMAS)}')


def _twin_problems(document):
    """Facts a shape check cannot express for the twin record shipped beside an adapter.

    "valid" must mean the record is internally consistent and says which of its
    numbers are codec-realized versus behaviour claims: the tensor counts have to
    add up, every delivered row has to name the codec type that realized it, the
    behaviour block has to be either MEASURED with numbers or explicitly UNMEASURED
    with none, a claimed serve argv has to be present, and the ledger has to carry
    both measurement classes.
    """
    problems = []
    edited = document.get('edited_tensors')
    untouched = document.get('untouched_tensors')
    total = document.get('tensor_count')
    if all(type(value) is int for value in (edited, untouched, total)):
        if edited + untouched != total:
            problems.append(f'TWIN_TENSOR_COUNTS_DISAGREE: {edited} edited + {untouched} untouched '
                            f'!= tensor_count {total}')
    elif not all(type(value) is int for value in (edited, untouched, total)):
        problems.append('TWIN_TENSOR_COUNTS_DISAGREE: edited_tensors, untouched_tensors and '
                        'tensor_count must all be integers')
    delivered = document.get('delivered') if isinstance(document.get('delivered'), dict) else {}
    raw_rows = delivered.get('tensors')
    rows = raw_rows if isinstance(raw_rows, list) else []
    raw_types = document.get('edit_type_per_tensor')
    edit_types = raw_types if isinstance(raw_types, dict) else {}
    if type(edited) is int and len(rows) > edited:
        problems.append(f'TWIN_DELIVERED_OVER_EDITED: {len(rows)} delivered rows for {edited} '
                        f'edited tensors')
    for row in rows:
        if not isinstance(row, dict):
            problems.append('TWIN_DELIVERED_ROW_INVALID: delivered.tensors carries a non-object row')
            break
        name = row.get('name')
        if isinstance(name, str) and name not in edit_types:
            problems.append(f'TWIN_EDIT_TYPE_MISSING: {name} has no entry in '
                            f'edit_type_per_tensor, so the codec that realized it is unnamed')
            break
    behaviour = document.get('behaviour') if isinstance(document.get('behaviour'), dict) else {}
    status = behaviour.get('status')
    merged = behaviour.get('merged_delta_br')
    adapter = behaviour.get('adapter_delta_br')
    ratio = behaviour.get('adapter_to_merged_ratio')
    ranking = behaviour.get('ranking_valid_on_this_artifact')
    floor = behaviour.get('ranking_validity_ratio_floor')
    if status not in ('MEASURED', 'UNMEASURED'):
        problems.append(f'BEHAVIOUR_EFFECT_INCONSISTENT: status is {status!r}; it must be '
                        f'MEASURED or UNMEASURED')
    elif status == 'MEASURED':
        if not isinstance(merged, (int, float)) or isinstance(merged, bool):
            problems.append('BEHAVIOUR_EFFECT_INCONSISTENT: MEASURED with no merged_delta_br')
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
            problems.append('BEHAVIOUR_EFFECT_INCONSISTENT: MEASURED with no '
                            'adapter_to_merged_ratio')
        if isinstance(ratio, (int, float)) and not isinstance(ratio, bool):
            expected = isinstance(floor, (int, float)) and ratio >= floor
            if ranking is not expected:
                problems.append(f'BEHAVIOUR_EFFECT_INCONSISTENT: ranking_valid_on_this_artifact '
                                f'is {ranking!r} but ratio {ratio} against floor {floor} implies '
                                f'{expected}')
        if isinstance(merged, (int, float)) and isinstance(adapter, (int, float)) and adapter:
            implied = round(merged / adapter, 3)
            if not isinstance(ratio, (int, float)) or abs(ratio - implied) > 0.002:
                problems.append(f'BEHAVIOUR_EFFECT_INCONSISTENT: merged {merged} / adapter '
                                f'{adapter} implies ratio {implied}, record has {ratio}; the '
                                f'merged field may hold the adapter number')
    else:
        if merged is not None or ratio is not None:
            problems.append('BEHAVIOUR_EFFECT_INCONSISTENT: UNMEASURED but the record still carries '
                            'merged_delta_br or a ratio')
        if ranking is not False:
            problems.append('BEHAVIOUR_EFFECT_INCONSISTENT: UNMEASURED cannot rank as valid')
    if behaviour.get('argv_recorded') is True and not behaviour.get('serve_argv'):
        problems.append('SERVE_ARGV_MISSING: argv_recorded is true but serve_argv is empty')
    classes = {entry.get('class') for entry in document.get('measurement_ledger') or []
               if isinstance(entry, dict)}
    for required in ('codec-realized', 'behaviour-claim'):
        if required not in classes:
            problems.append(f'MEASUREMENT_CLASS_UNLABELLED: measurement_ledger has no '
                            f'{required!r} entry, so the reader cannot tell which numbers are which')
    return problems


def _plan_card_problems(document):
    """The manifest beside plan-card.tar has to describe itself consistently."""
    from .data import digest
    problems = []
    try:
        from .registry import BUNDLE_FORMAT
    except Exception:  # registry is light, but a schema check must never need torch
        BUNDLE_FORMAT = 'b70ablator-plan-bundle-v1'
    if document.get('format') != BUNDLE_FORMAT:
        problems.append(f'PLAN_CARD_STRUCTURE: format {document.get("format")!r} is not '
                        f'{BUNDLE_FORMAT!r}')
    meta = document.get('meta') if isinstance(document.get('meta'), dict) else {}
    payload = document.get('payload_sha256')
    if meta.get('payload_sha256') != payload:
        problems.append('PLAN_CARD_PAYLOAD_MISMATCH: manifest payload_sha256 disagrees with the '
                        'signed meta payload_sha256')
    content = {key: value for key, value in meta.items() if key != 'meta_digest'}
    if meta.get('meta_digest') != digest(content):
        problems.append('PLAN_CARD_META_DIGEST_MISMATCH: signed meta fails its own meta_digest')
    return problems


def _artifact_problems(name, document):
    """Facts that must hold for an artifact whose schema is shape-only.

    adapter.json is the R8 label, so "valid" has to mean more than "the keys are
    there": the merged field must not hold an adapter number and the curve claim
    must be present and true (jdewitt.adapter_card.verify_adapter_card). twin.json
    labels codec-realized numbers against behaviour claims, and plan-card.json is
    the manifest of a digest-pinned bundle, so both get the same treatment.
    """
    if name == 'adapter.json':
        from .adapter_card import verify_adapter_card
        return verify_adapter_card(document)
    if name == 'twin.json':
        return _twin_problems(document)
    if name == 'plan-card.json':
        return _plan_card_problems(document)
    return []


def validate_artifact(path, name=None):
    """Check one artifact against its published schema. File reads only."""
    path = Path(path)
    name = name or infer_schema_name(path)
    if name not in SCHEMAS:
        raise ValueError(f'No published schema for {name!r}; available: {sorted(SCHEMAS)}')
    document = json.loads(path.read_text())
    problems = validate_against(document, SCHEMAS[name]) + _artifact_problems(name, document)
    notes = {
        'adapter.json': 'Shape, version and the R8 card semantics: curve claim, merged-field '
                        'provenance and ratio/ranking agreement.',
        'twin.json': 'Shape, version and the codec-realized/behaviour split: tensor counts, per-row '
                     'codec type, behaviour labelling and both measurement-ledger classes.',
        'plan-card.json': 'Shape, version and the plan-card manifest semantics: signed meta digest '
                          'and payload digest agreement. The bundle bytes are re-hashed by '
                          'release-assemble, which reads the tar.',
    }
    note = notes.get(name, 'Shape and version only; digests and semantics are checked by release '
                           'verify commands.')
    return {'file': str(path.resolve()), 'schema': name, 'valid': not problems,
            'problems': problems, 'note': note}


def verify_all(directory):
    """Check every published artifact found in a directory against its schema. Reads only."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    artifact_names = [name for name in SCHEMAS if not name.endswith('.command.json')]
    checked, missing = [], []
    for name in sorted(artifact_names):
        path = directory / name
        if not path.is_file():
            missing.append(name)
            continue
        document = json.loads(path.read_text())
        problems = validate_against(document, SCHEMAS[name]) + _artifact_problems(name, document)
        checked.append({'file': str(path), 'schema': name, 'valid': not problems, 'problems': problems[:5]})
    return {'directory': str(directory.resolve()), 'checked': checked, 'missing_schemas': missing,
            'valid': all(row['valid'] for row in checked),
            'note': 'Shape and version only for release artifacts; adapter.json also gets its R8 '
                    'card semantics, twin.json its codec-realized/behaviour split, and '
                    'plan-card.json its signed-meta/payload agreement. Digests and claim chains '
                    'are checked by release verify commands.'}
