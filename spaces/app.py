# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Space's Gradio surface: one repo, one quant, one Start button, one file.

Every answer and every byte on this page comes from `jdewitt.space_loop`, which is
importable without Gradio and without torch, so the loop is tested on a CPU and this
file stays thin. The control surface is deliberately small: a repo id, the bank's
short list of covered bases, a quant dropdown built from the repo's `*.gguf` list,
the licence line, the coverage verdict, and one Start button. There is no strength,
layer or method control — the bank's measured arm is the only one applied (§5.3).

The quant dropdown keeps every file the repo lists. A source type the resolved
entry does not cover is shown with its reason in the label and refused by code if
picked anyway; hiding it would make the picker lie about coverage (§7.1).

The caveat comes from the measurement state, not from this file, and it is rendered
above the download control: `PANEL_ORDER` in `space_loop` is the order used below.
"""
import json

import gradio as gr

from jdewitt import space_loop

BANK = space_loop.default_bank()
BASES = space_loop.covered_bases(BANK)
TITLE = 'Banked edit — coverage, licence, one file'

INTRO = (
    'This Space answers one question for one file: does the direction bank cover '
    'this GGUF, at the quant you hold? It reads your file\'s header, checks the '
    'licence on the model card, and — when both pass — applies the bank\'s measured '
    'edit to a copy and gives you that copy with its card. No inference runs here: '
    'the requirements carry no transformers, no llama-server and no tokenizer.\n\n'
    'The label on the result is generated from the bank entry\'s measurement state. '
    '`measured_artifact_class` means your exact repo@revision and quant were measured; '
    '`banked_unmeasured` means the direction was fitted elsewhere and only codec '
    'fidelity is claimed; `refused` names why nothing was edited.'
)

NO_CAVEAT_YET = (
    'Nothing has been edited yet. When a job finishes, this line carries the label\'s '
    'own sentence before the download control.'
)


def _quant_update(view):
    choices = [(choice['label'], choice['name']) for choice in view['quant_control']]
    enabled = [choice['name'] for choice in view['quant_control'] if choice['enabled']]
    return gr.update(choices=choices, value=(enabled[0] if enabled else None))


def refresh(repo):
    """Repo id -> (quant choices, coverage verdict, licence line). Metadata only."""
    repo = (repo or '').strip()
    if not repo:
        empty = gr.update(choices=[], value=None)
        return (empty, 'Coverage: enter owner/name (the shipped bank has no base until '
                       'Phase 1 seeds it).', 'Licence: not read yet.')
    view = space_loop.preview(repo, None, bank_dir=BANK)
    verdict = view['coverage_line']
    if view.get('refusal'):
        verdict = space_loop.refusal_copy(view['refusal']['code'], view['refusal']['message'])
    return _quant_update(view), verdict, view['licence_line']


def pick_base(base):
    """The covered-bases list writes the repo field, then refreshes the rest."""
    quant_update, verdict, licence_line = refresh(base)
    return (base or ''), quant_update, verdict, licence_line


# ZeroGPU boots a Space only when one `@spaces.GPU` function is declared, and the
# free tier's cap is 120 s per call. The loop is CPU numpy end to end: it decodes,
# projects and re-encodes 42 tensors in `bank_apply` and never touches the GPU, so a
# job does not call the decorated function at all. That is the honest shape — the
# alternative, wrapping the CPU edit in a GPU call because the edit "needs" a GPU,
# queues the visitor behind a card the edit cannot use and charges the quota for
# waiting (§4). The run record states how many GPU calls there were: none.
GPU_CALL_SECONDS = 120

try:  # the ZeroGPU image carries `spaces`; a CPU run does not
    import spaces

    @spaces.GPU(duration=GPU_CALL_SECONDS)
    def gpu_declaration():
        """Satisfies ZeroGPU's startup check; no job calls this function."""
        return {'declared': True, 'requested_seconds': GPU_CALL_SECONDS}

    HOSTING = ('ZeroGPU hardware; one @spaces.GPU function is declared for the '
               'startup check and no job calls it, so a job spends 0 GPU seconds')
except ImportError:  # local or CPU run
    gpu_declaration = None
    HOSTING = 'no `spaces` package: CPU-only process, a job spends 0 GPU seconds'


def _record(result):
    """One JSON line into the Space's own log: the numbers the run card reads.

    The card quotes a Space runtime only from the Space's log, so the loop's own
    result — per-stage seconds, the download rate, the container's cores and RSS
    high-water mark, the delivered bytes and sha256 — is printed here, not
    reconstructed from the page. One line, one prefix, the whole document.
    """
    print('jdewitt-live-run ' + json.dumps(result, sort_keys=True, default=str),
          flush=True)


def start(repo, filename):
    """One job: the loop runs, the caveat and the delivered files come back."""
    if not (filename or '').strip():
        return ('Choose a .gguf file from the list first.', None, None, 'Nothing ran.')
    print('jdewitt-live-run-start ' + json.dumps(
        {'repo': (repo or '').strip(), 'file': filename.strip(), 'hosting': HOSTING,
         'host': space_loop.host_probe()}, sort_keys=True, default=str), flush=True)
    result = space_loop.start_job((repo or '').strip(), None, filename.strip(),
                                  space_loop.quant_tag(filename), bank_dir=BANK)
    _record(result)
    sections = {section['id']: section for section in space_loop.panel_sections(result)}
    caveat = sections['caveat'].get('body') or ''
    status = (f"Label `{result['label']}` · {result['wall_seconds']} s · "
              f"{HOSTING} · workspace deleted: {result['workspace_deleted']}")
    if result['label'] == 'refused':
        return caveat, None, None, status
    return (caveat, result['files']['artifact'],
            [result['files']['card'], result['files']['sidecar'], result['files']['sha256sums']],
            status)


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f'# {TITLE}\n\n{INTRO}')
    with gr.Row():
        repo = gr.Textbox(label='Model repository', placeholder='owner/name',
                          value=(BASES[0]['repo'] if BASES else ''))
        bases = gr.Dropdown(label='or pick a covered base',
                            choices=[(base['repo'], base['repo']) for base in BASES],
                            value=(BASES[0]['repo'] if BASES else None))
    # A visitor who pastes a file name is allowed to: the loop validates the name and
    # the header, and a cold session has no list until `refresh` fills one.
    quant = gr.Dropdown(label='Quant file', choices=[], value=None,
                        allow_custom_value=True)
    licence = gr.Markdown('Licence: not read yet.')
    coverage = gr.Markdown('Coverage: not checked yet.')
    caveat = gr.Markdown(NO_CAVEAT_YET)
    run = gr.Button('Start', variant='primary')
    artifact = gr.File(label='Download the edited GGUF', interactive=False)
    extras = gr.File(label='Card, sidecar and SHA256SUMS', file_count='multiple',
                     interactive=False)
    status = gr.Markdown('')

    bases.change(pick_base, inputs=[bases], outputs=[repo, quant, coverage, licence])
    repo.submit(refresh, inputs=[repo], outputs=[quant, coverage, licence])
    demo.load(refresh, inputs=[repo], outputs=[quant, coverage, licence])
    run.click(start, inputs=[repo, quant], outputs=[caveat, artifact, extras, status])

# One job at a time. The queue is the first lock; `space_loop._job_lock` is the
# second, so two worker processes cannot both hold a 50 GB workspace.
demo.queue(default_concurrency_limit=1)

if __name__ == '__main__':
    demo.launch()
