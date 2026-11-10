"""Checks of the paper's LaTeX sources (DECISIONS D-094).

1. Every ``\\rf{key}`` used anywhere under ``paper/`` is defined in the generated
   ``numbers.tex``.
2. No bare number appears in the prose of ``paper/sections/*.tex``: after removing comments,
   macro arguments that hold keys, references, citations, file paths (``\\file{}``) and code
   (``\\code{}``), every remaining digit must belong to an allowed identifier (hypothesis,
   ablation, approval, decision, issue and ledger IDs, "3D", model and sampler names, document
   section signs). Inside math only the structural digits 0, 1 and 2 are allowed (``A_0``,
   ``1 - \\bar\\alpha_t``); design constants such as horizons use ``\\rf{}`` there too.
3. No placeholder (TBD, TODO, XXX, ``??``) appears.
"""

from __future__ import annotations

import re
from pathlib import Path

from roboforge.paths import REPO_ROOT

PAPER = "paper"
NUMBERS = "paper/generated/numbers.tex"

# Commands whose (first) argument is removed before the scan: keys, labels, references,
# citations, file paths, code, includes, environments, colours and lengths.
ARG_COMMANDS = ["rf", "cite", "citet", "citep", "citealp", "ref", "eqref", "autoref", "label",
                "input", "include", "includegraphics", "url", "file", "code", "begin", "end",
                "hyperref", "href", "textsuperscript", "bibliography", "bibliographystyle",
                "setlength", "vspace", "hspace", "rotatebox", "resizebox", "scalebox",
                "newcommand", "renewcommand", "usepackage", "documentclass"]

ALLOWED_TOKEN = re.compile(
    r"^(?:H[1-5]|H2-c|A[1-3]|A-0[1-4]|KI-\d{3}|D-\d{3}|E\d{1,2}-\d{2}|[123]D|ResNet-18|"
    r"DDIM-10|DDPM-100|L4|M1|SHA-256|v0\.1|v1\.5|x86_64|aarch64|arm64|"
    r"MuJoCo|OSMesa|EGL|CGL|BC-MLP|BC-RNN|U-Net|DP3|DP3-inspired|float32|Apache-2\.0|[ABC]-\d{1,2})$")

STRUCTURAL = [
    r"Sessions?~?\d+(?:--\d+)?",   # project session numbers (identifiers)
    r"Level[~ ]?[12]",             # study names
    r"§\s?\d+(?:\.\d+)*",          # RESEARCH_PROTOCOL / MASTER_SPEC section signs
    r"\\S\s?\d+(?:\.\d+)*",
]

PLACEHOLDER = re.compile(r"\bTBD\b|\bTODO\b|\bXXX\b|\?\?")


def _strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def _remove_command_args(tex: str) -> str:
    for cmd in ARG_COMMANDS:
        # \cmd[opt]{arg} with one level of nested braces in arg.
        tex = re.sub(r"\\" + cmd + r"\*?(?:\[[^\]]*\])?\{(?:[^{}]|\{[^{}]*\})*\}", " ", tex)
    return tex


def defined_keys(root: Path = REPO_ROOT) -> set[str]:
    """Keys defined in the generated numbers.tex."""
    txt = (root / NUMBERS).read_text()
    return set(re.findall(r"\\csname rf@([^\\]+)\\endcsname", txt))


def used_keys(root: Path = REPO_ROOT) -> dict[str, set[str]]:
    """``\\rf{key}`` uses per LaTeX file under paper/ (generated files excluded)."""
    out: dict[str, set[str]] = {}
    for p in sorted((root / PAPER).rglob("*.tex")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith("paper/generated/"):
            continue
        keys = set(re.findall(r"\\rf\{([^}]*)\}", _strip_comments(p.read_text())))
        if keys:
            out[rel] = keys
    return out


def bare_numbers(tex: str) -> list[str]:
    """Digits in prose that are not an allowed identifier; returns offending snippets."""
    t = _remove_command_args(_strip_comments(tex))
    bad: list[str] = []
    # Math: only the structural digits 0, 1 and 2, standing alone.
    for m in re.finditer(r"\$([^$]*)\$|\\\((.*?)\\\)|\\\[(.*?)\\\]", t, re.S):
        body = next(g for g in m.groups() if g is not None)
        for num in re.findall(r"\d+(?:\.\d+)?", body):
            if num not in {"0", "1", "2"}:
                bad.append(f"math: ${body.strip()}$")
                break
    t = re.sub(r"\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]", " ", t, flags=re.S)
    for pat in STRUCTURAL:
        t = re.sub(pat, " ", t)
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-_]*", t):
        tok = tok.rstrip(".-")
        if re.search(r"\d", tok) and not ALLOWED_TOKEN.match(tok):
            bad.append(tok)
    return bad


def lint_paper(root: Path = REPO_ROOT) -> list[str]:
    """All problems of the paper's LaTeX sources (empty when clean)."""
    problems: list[str] = []
    if not (root / NUMBERS).is_file():
        return [f"{NUMBERS} missing: run scripts/paper_assets.py build"]
    keys = defined_keys(root)
    for rel, used in used_keys(root).items():
        for k in sorted(used - keys):
            problems.append(f"{rel}: \\rf{{{k}}} is not defined in {NUMBERS}")
    for p in sorted((root / PAPER / "sections").glob("*.tex")):
        rel = p.relative_to(root).as_posix()
        txt = p.read_text()
        for tok in bare_numbers(txt):
            problems.append(f"{rel}: bare number '{tok}' (use \\rf{{}} or an allowed identifier)")
        for m in PLACEHOLDER.finditer(_strip_comments(txt)):
            problems.append(f"{rel}: placeholder '{m.group(0)}'")
    return problems
