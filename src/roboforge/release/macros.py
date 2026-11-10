"""Read the paper's generated result macros as plain text (DECISIONS D-103).

`paper/generated/numbers.tex` defines one macro per value used in the paper, each line of the
form ``\\expandafter\\def\\csname rf@<key>\\endcsname{<value>}``. The values are LaTeX
fragments (``\\ensuremath{-}41.3``, ``1,077,255``, ``separated (\\ensuremath{+})``). This module
parses them and converts the few LaTeX constructs they use into Unicode text for Markdown; any
other construct is an error, so a README never shows half-converted TeX.
"""

from __future__ import annotations

import re
from pathlib import Path

from roboforge.paths import REPO_ROOT

NUMBERS_TEX = REPO_ROOT / "paper" / "generated" / "numbers.tex"

_LINE = re.compile(r"^\\expandafter\\def\\csname rf@(?P<key>[^\\]+)\\endcsname\{(?P<val>.*)\}$")
# The only LaTeX constructs that numbers.tex values may contain, mapped to plain text.
_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\ensuremath{-}", "−"),
    (r"\ensuremath{+}", "+"),
    (r"\ensuremath{\pm}", "±"),
    (r"\,", " "),
    (r"\%", "%"),
)


def parse_macros(text: str) -> dict[str, str]:
    """Return ``{key: raw LaTeX value}`` for every macro line; duplicate keys are an error."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if m is None:
            continue
        key = m.group("key")
        if key in out:
            raise ValueError(f"duplicate macro key {key!r}")
        out[key] = m.group("val")
    if not out:
        raise ValueError("no rf@ macro found")
    return out


def to_text(value: str) -> str:
    """Convert one macro value to plain text, refusing any LaTeX it does not know."""
    s = value
    for tex, plain in _REPLACEMENTS:
        s = s.replace(tex, plain)
    if "\\" in s or "{" in s or "}" in s:
        raise ValueError(f"unconverted LaTeX in macro value {value!r}")
    return s


def load_macros(path: Path = NUMBERS_TEX) -> dict[str, str]:
    """Load ``numbers.tex`` and return ``{key: plain-text value}``."""
    raw = parse_macros(path.read_text(encoding="utf-8"))
    return {k: to_text(v) for k, v in raw.items() if "\\" not in v or _convertible(v)}


def _convertible(value: str) -> bool:
    try:
        to_text(value)
    except ValueError:
        return False
    return True


class Macros:
    """Strict accessor: an unknown key raises, like ``\\rf{}`` in the paper."""

    def __init__(self, values: dict[str, str]) -> None:
        self._v = values

    def __getitem__(self, key: str) -> str:
        if key not in self._v:
            raise KeyError(f"undefined result key {key!r} (run make paper-assets)")
        return self._v[key]

    def __contains__(self, key: object) -> bool:
        return key in self._v
