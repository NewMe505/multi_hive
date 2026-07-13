"""
console.py — the shared Rich console, and the Windows UTF-8 bootstrap.

Every panel and status line in this project uses emoji (🐝 🚨 ✅). Windows
defaults its stdio encoding to cp1252, which has no codepoint for any of them,
so the first console.print() of the banner raised UnicodeEncodeError and killed
the process before a single node ran.

Reconfiguring the streams has to happen *before* the Console is constructed —
Rich samples the stream's encoding when it decides how to render — which is why
this lives in its own module that everything else imports the console from,
rather than being a call someone has to remember to make first.

errors="replace" rather than "strict": a console glyph is never worth crashing
a sprint over. A terminal too old to draw an emoji prints a placeholder instead.
"""
from __future__ import annotations

import contextlib
import sys

from rich.console import Console


def _enable_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # Detached or already-wrapped stream: nothing to do, and not worth
        # failing over.
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")


_enable_utf8_output()

console = Console()
