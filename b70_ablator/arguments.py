# SPDX-License-Identifier: AGPL-3.0-or-later
"""Argument definitions must remain free of model/runtime imports."""
from pathlib import Path


def add_arguments(sub):
    generate = sub.add_parser('generate', help='Save model answers, resuming missing prompts')
    for key in ('model', 'server', 'suite', 'output'):
        generate.add_argument('--'+key, type=Path, required=True)
    for key in ('max-tokens', 'ctx-size'):
        generate.add_argument('--'+key, type=int, required=True)
    for key in ('refusal-floor','batch-max-sequences','batch-token-budget'):
        generate.add_argument('--'+key,type=int)
    generate.add_argument('--allow-secondary',action='store_true',help='serve on a non-B70 Vulkan device as an explicitly opt-in second-rank test backend; recorded as rank=secondary and never a certification')
    score = sub.add_parser('score', help='Score saved answers independently')
    score.add_argument('--mode', dest='scoring_mode', choices=['judge','refusal-check'], default='judge')
    for key in ('answers','suite','output'):
        score.add_argument('--'+key, type=Path, required=True)
    for key in ('server','judge-artifact','calibration'):
        score.add_argument('--'+key, type=Path)
    for key in ('judge-max-tokens','ctx-size'):
        score.add_argument('--'+key, type=int)
    for key in ('judge-min-confidence','judge-min-accuracy'):
        score.add_argument('--'+key, type=float)
    score.add_argument('--sandbox-image', default='python:3.12-slim')
    score.add_argument('--retry-invalid', action='store_true')
    status = sub.add_parser('status', help='Read persisted job progress without loading models')
    status.add_argument('--answers', type=Path, required=True)
    status.add_argument('--scores', type=Path)
