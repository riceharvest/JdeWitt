# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat rendering independent of device/model loading."""
from datetime import datetime, timezone
from jinja2.exceptions import TemplateError


def render_chat(tokenizer, prompt, system='You are a helpful assistant.', enable_thinking=False):
    options = {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': enable_thinking,
               'strftime_now': lambda fmt: datetime(2026, 1, 1, tzinfo=timezone.utc).strftime(fmt)}
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]
    try:
        return tokenizer.apply_chat_template(messages, **options)
    except TemplateError as error:
        # Retry ONLY the known system-role rejection, not arbitrary template errors.
        text = str(error).lower()
        if 'system' not in text or not any(x in text for x in ('not support', 'unsupported', 'not allowed')):
            raise
        content = system + '\n\n' + prompt if system else prompt
        return tokenizer.apply_chat_template([{'role': 'user', 'content': content}], **options)
