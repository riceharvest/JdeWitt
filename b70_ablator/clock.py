# SPDX-License-Identifier: AGPL-3.0-or-later
"""One clock for artifacts, so a run can be reproduced with a pinned stamp.

Set B70_ABLATOR_CLOCK to an epoch float or an ISO-8601 instant and every
artifact timestamp written through here resolves to that instant.
"""
import datetime
import os
import time


def now_iso():
    override = os.environ.get('B70_ABLATOR_CLOCK')
    epoch = None
    if override is not None:
        try:
            epoch = float(override)
        except ValueError:
            parsed = datetime.datetime.fromisoformat(override.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.timezone.utc)
            epoch = parsed.timestamp()
    else:
        epoch = time.time()
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(epoch))
