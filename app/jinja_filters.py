"""Jinja2 template filters."""

from __future__ import annotations

import re

from markupsafe import Markup, escape

# Numbers: comma-separated thousands, decimals, optional % — avoid matching inside HTML
_NUM_RE = re.compile(
    r"(?<![\w/.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)(%?)(?![\w.])"
)


def highlight_numerics(value: object) -> Markup:
    """Escape text and wrap numeric tokens in a styled span."""
    if value is None:
        return Markup("")
    s = escape(str(value))

    def repl(m: re.Match[str]) -> str:
        return (
            f'<span class="insight-number fw-semibold text-primary">'
            f"{m.group(1)}{m.group(2)}</span>"
        )

    return Markup(_NUM_RE.sub(repl, str(s)))
