"""Filesystem locations used across RoboForge.

All dataset paths resolve through :func:`data_root`, which honors the
``ROBOFORGE_DATA_ROOT`` environment variable so the same code works on a laptop,
in CI (where no data exists) and inside a remote container with a mounted volume.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Return the root directory for datasets (``$ROBOFORGE_DATA_ROOT`` or ``<repo>/data``)."""
    env = os.environ.get("ROBOFORGE_DATA_ROOT")
    return Path(env).expanduser().resolve() if env else REPO_ROOT / "data"


def raw_data_dir() -> Path:
    """Return the directory holding unmodified third-party downloads (gitignored)."""
    return data_root() / "raw"


def manifests_dir() -> Path:
    """Return the directory of committed, small dataset manifests (always inside the repo)."""
    return REPO_ROOT / "data" / "manifests"


def configs_dir() -> Path:
    """Return the committed configuration directory."""
    return REPO_ROOT / "configs"


def feasibility_reports_dir() -> Path:
    """Return the directory for Session 0 feasibility evidence."""
    return REPO_ROOT / "reports" / "feasibility"


EVIDENCE_SESSION_ENV = "ROBOFORGE_EVIDENCE_SESSION"


def evidence_session() -> int:
    """The session whose evidence a command writes: ``$ROBOFORGE_EVIDENCE_SESSION``.

    There is deliberately no default. Evidence tools shared across sessions derive their
    output directory from this value (DECISIONS D-039), so they can never write into an
    earlier session's committed evidence by accident. The container wrapper mounts
    ``reports/session<N>`` writable for the same ``N``.
    """
    raw = os.environ.get(EVIDENCE_SESSION_ENV, "")
    if not raw.isdigit():
        raise RuntimeError(f"set {EVIDENCE_SESSION_ENV}=<session number> (or pass an explicit "
                           f"output directory); got {raw!r}")
    return int(raw)


def evidence_reports_dir(*parts: str) -> Path:
    """``reports/session<N>/<parts>`` for the current :func:`evidence_session`."""
    return REPO_ROOT / "reports" / f"session{evidence_session()}" / Path(*parts)


def repo_relative(path: Path | str) -> str:
    """Return ``path`` as a POSIX string relative to the repo root, or absolute if outside it.

    Evidence reports record file locations with this function. Paths inside the repository
    stay portable (``reports/feasibility/frames/x.png``). An output directory outside it, such
    as ``/out`` mounted into a container, is recorded as an absolute path instead of failing.
    """
    # Lexical first: a directory mounted through a symlink inside the repository (a Modal
    # Volume at /app/runs links to /__modal/volumes/<id>) keeps its repository path.
    lexical = Path(os.path.abspath(path))
    try:
        return lexical.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        pass
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()
