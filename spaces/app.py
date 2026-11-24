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
import queue as queue_module
import threading
from pathlib import Path

import gradio as gr

from jdewitt import space_deliver, space_loop

BANK = space_loop.default_bank()
BASES = space_loop.covered_bases(BANK)
TITLE = 'Banked edit — coverage, licence, one file'
# v1.1 is behind a flag: without `JW_OAUTH_PUSH` the page keeps the direct
# download and never asks for a scope (§8).
PUSH = space_deliver.enabled()

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

try:  # the ZeroGPU image carries `spaces`; a CPU run does not (or has another package)
    import spaces

    if not hasattr(spaces, 'GPU'):
        raise ImportError('this `spaces` module has no GPU decorator')

    @spaces.GPU(duration=GPU_CALL_SECONDS)
    def gpu_declaration():
        """Satisfies ZeroGPU's startup check; no job calls this function."""
        return {'declared': True, 'requested_seconds': GPU_CALL_SECONDS}

    HOSTING = ('ZeroGPU hardware; one @spaces.GPU function is declared for the '
               'startup check and no job calls it, so a job spends 0 GPU seconds')
except ImportError:  # local or CPU run
    gpu_declaration = None
    HOSTING = 'no `spaces.GPU`: CPU-only process, a job spends 0 GPU seconds'


def _record(result):
    """One JSON line into the Space's own log: the numbers the run card reads.

    The card quotes a Space runtime only from the Space's log, so the loop's own
    result — per-stage seconds, the download rate, the container's cores and RSS
    high-water mark, the delivered bytes and sha256 — is printed here, not
    reconstructed from the page. One line, one prefix, the whole document.
    """
    print('jdewitt-live-run ' + json.dumps(result, sort_keys=True, default=str),
          flush=True)


def _fmt_bytes(count):
    count = float(count)
    for unit in ('B', 'kB', 'MB', 'GB', 'TB'):
        if count < 1000 or unit == 'TB':
            return f'{count:.0f} {unit}' if unit in ('B', 'kB') else f'{count:.1f} {unit}'
        count /= 1000


def _fmt_seconds(seconds):
    if seconds is None:
        return '--:--'
    seconds = int(round(seconds))
    return f'{seconds // 60}:{seconds % 60:02d}'


def publish_line(snapshot):
    """One line from the bytes actually moved: file, progress, rate, remaining."""
    percent = (100.0 * snapshot['bytes_done'] / snapshot['bytes_total']
               if snapshot['bytes_total'] else 0.0)
    return (f"Uploading `{snapshot['file']}` ({snapshot['file_index']}/"
            f"{snapshot['file_count']}) · {_fmt_bytes(snapshot['file_done'])} of "
            f"{_fmt_bytes(snapshot['file_total'])} · {_fmt_bytes(snapshot['bytes_done'])} of "
            f"{_fmt_bytes(snapshot['bytes_total'])} total ({percent:.1f}%) · "
            f"**{_fmt_bytes(snapshot['rate_bytes_per_second'])}/s** · "
            f"remaining {_fmt_seconds(snapshot['eta_seconds'])}")


def cancel_upload():
    """The page's Cancel button: stop the PUT and let the worker delete the repo."""
    repo = space_deliver.cancel_current()
    if repo is None:
        return 'No upload is running.'
    return f'Cancelling; `{repo}` is deleted before the job returns.'


def _deliver(files, repo, quant, token, cancel, on_progress):
    return space_deliver.deliver(files, token=token, base_repo=repo, quant=quant,
                                 cancel=cancel, on_progress=on_progress)


def start(repo, filename, publish, oauth_token: gr.OAuthToken | None = None):
    """One job: the loop runs, the caveat and the delivered files come back.

    A generator, because an upload has to show the bytes it moved: each snapshot
    the delivery reports is one `publish_status` line, and the final line carries
    the repo URL and the commit sha. Without the flag or without a sign-in the
    direct download is unchanged and the publish line says why nothing was pushed.
    """
    if not (filename or '').strip():
        yield 'Choose a .gguf file from the list first.', None, None, 'Nothing ran.', ''
        return
    print('jdewitt-live-run-start ' + json.dumps(
        {'repo': (repo or '').strip(), 'file': filename.strip(), 'hosting': HOSTING,
         'publish': bool(publish), 'oauth': bool(space_deliver.token_text(oauth_token)),
         'host': space_loop.host_probe()}, sort_keys=True, default=str), flush=True)
    result = space_loop.start_job((repo or '').strip(), None, filename.strip(),
                                  space_loop.quant_tag(filename), bank_dir=BANK)
    _record(result)
    sections = {section['id']: section for section in space_loop.panel_sections(result)}
    caveat = sections['caveat'].get('body') or ''
    status = (f"Label `{result['label']}` · {result['wall_seconds']} s · "
              f"{HOSTING} · workspace deleted: {result['workspace_deleted']}")
    if result['label'] == 'refused':
        yield caveat, None, None, status, ''
        return
    extras = [result['files']['card'], result['files']['sidecar'],
              result['files']['sha256sums']]
    yield caveat, result['files']['artifact'], extras, status, ''
    if not publish:
        return
    if not space_deliver.enabled():
        yield (caveat, result['files']['artifact'], extras, status,
               'Publishing is off on this deployment: `JW_OAUTH_PUSH` is not set, so the '
               'file above is the delivery. Nothing was uploaded.')
        return
    token = space_deliver.token_text(oauth_token)
    if not token:
        yield (caveat, result['files']['artifact'], extras, status,
               'Sign in with the Hub account that should own the file, then run again; '
               'nothing was uploaded.')
        return
    files = {Path(result['files']['artifact']).name: result['files']['artifact'],
             'card.md': result['files']['card'],
             'sidecar.json': result['files']['sidecar']}
    if result['files'].get('revert'):
        files['.revert.npz'] = result['files']['revert']
    queue = queue_module.Queue()
    box = {}

    def worker():
        try:
            box['result'] = _deliver(files, result['job']['repo'],
                                     result['artifact'].get('quant') or space_loop.quant_tag(filename),
                                     token, threading.Event(), queue.put)
        except Exception as error:  # the page shows the typed code, never a stack
            box['error'] = error

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    yield caveat, result['files']['artifact'], extras, status, 'Starting the upload…'
    while thread.is_alive() or not queue.empty():
        try:
            snapshot = queue.get(timeout=0.5)
        except queue_module.Empty:
            continue
        yield caveat, result['files']['artifact'], extras, status, publish_line(snapshot)
    thread.join(timeout=30)
    delivery = box.get('result')
    if delivery is None:
        error = box.get('error')
        code = getattr(error, 'code', 'UPLOAD_FAILED')
        message = getattr(error, 'message', str(error))
        yield (caveat, result['files']['artifact'], extras, status,
               f'**{code}** — {message}')
        return
    print('jdewitt-delivery ' + json.dumps(delivery, sort_keys=True, default=str),
          flush=True)
    summary = ' · '.join(f"{_fmt_bytes(f['total'])} at {_fmt_bytes(f['bytes_per_second'])}/s"
                         for f in delivery['files'])
    yield (caveat, result['files']['artifact'], extras, status,
           f"Delivered to **{delivery['repo_url']}** · commit `{delivery['commit_sha']}` · "
           f"{_fmt_bytes(delivery['bytes'])} in {delivery['wall_seconds']} s "
           f"({_fmt_bytes(delivery['bytes_per_second'])}/s) · {summary}")



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
    # v1.1 controls exist only when the flag is on: `gr.LoginButton` makes Gradio
    # attach the OAuth routes at startup, which needs `gradio[oauth]`. With the flag
    # off the page is the v1 page and nothing asks the visitor for a scope.
    publish = gr.Checkbox(
        label='Create jdewitt-<base>-<quant> in my Hub account and upload the result',
        value=False, visible=PUSH)
    if PUSH:
        login = gr.LoginButton()
        cancel = gr.Button('Cancel upload')
    publish_status = gr.Markdown('', visible=PUSH)
    artifact = gr.File(label='Download the edited GGUF', interactive=False)
    extras = gr.File(label='Card, sidecar and SHA256SUMS', file_count='multiple',
                     interactive=False)
    status = gr.Markdown('')

    bases.change(pick_base, inputs=[bases], outputs=[repo, quant, coverage, licence])
    repo.submit(refresh, inputs=[repo], outputs=[quant, coverage, licence])
    demo.load(refresh, inputs=[repo], outputs=[quant, coverage, licence])
    run.click(start, inputs=[repo, quant, publish],
              outputs=[caveat, artifact, extras, status, publish_status])
    if PUSH:
        cancel.click(cancel_upload, inputs=None, outputs=[publish_status])

# One job at a time. The queue is the first lock; `space_loop._job_lock` is the
# second, so two worker processes cannot both hold a 50 GB workspace.
demo.queue(default_concurrency_limit=1)

if __name__ == '__main__':
    demo.launch()
