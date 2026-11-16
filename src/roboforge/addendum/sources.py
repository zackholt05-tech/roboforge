"""The three extension results the v1.1 addendum may read, pinned by SHA-256 (D-133).

A file whose bytes differ from its pin stops every build, so the addendum, the README block
and the Observatory's extension findings can never be generated from an edited or regenerated
result. Each study keeps its own label (D-105, D-133).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT


@dataclass(frozen=True)
class Source:
    """One pinned extension result: key, repository path, SHA-256 and the study's label."""

    key: str
    path: str
    sha256: str
    label: str


SOURCES: dict[str, Source] = {s.key: s for s in (
    Source("x1", "reports/session13/x1_clamp/results.json",
           "ad50368645b96fb87489310243535f97694b1e47a2ff993915b039b2199a5ee7",
           "exploratory, after the freeze"),
    Source("x3", "reports/session14/x3/results.json",
           "4299a14930790bfdb2f91d7a3216c9bd7f6db1fa2171469a29bfed3c5c5b71d1",
           "separate replication"),
    Source("x2", "reports/session17/x2/results.json",
           "30c01f551333feafba70bfa4c9c6a414eda691f179800022d3871776be2df238",
           "X2 (Can MH)"),
)}


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def verify(root: Path = REPO_ROOT) -> list[str]:
    """Problems with the pinned files (empty when every file equals its pin)."""
    problems = []
    for s in SOURCES.values():
        p = root / s.path
        if not p.is_file():
            problems.append(f"{s.path}: missing")
        elif sha256_bytes(p.read_bytes()) != s.sha256:
            problems.append(f"{s.path}: SHA-256 differs from the pin {s.sha256}")
    return problems


def load(key: str, root: Path = REPO_ROOT) -> dict[str, Any]:
    """The parsed JSON of one pinned source, refused unless its bytes equal the pin."""
    s = SOURCES[key]
    data = (root / s.path).read_bytes()
    if sha256_bytes(data) != s.sha256:
        raise RuntimeError(f"{s.path}: SHA-256 differs from the pin {s.sha256} (D-133)")
    return json.loads(data)


def load_all(root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """All three pinned sources, after checking every pin first."""
    problems = verify(root)
    if problems:
        raise RuntimeError("extension results differ from their pins:\n" + "\n".join(problems))
    return {k: load(k, root) for k in SOURCES}
