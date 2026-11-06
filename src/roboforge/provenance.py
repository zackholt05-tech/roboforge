"""Capture software/hardware provenance for run manifests and evidence reports.

The returned dictionaries are embedded in every generated report so a reader can
tell exactly which code revision, interpreter, package versions and rendering
backend produced a given number or image.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT

TRACKED_PACKAGES: tuple[str, ...] = (
    "mujoco",
    "robosuite",
    "robomimic",
    "torch",
    "torchvision",
    "numpy",
    "h5py",
    "diffusers",
    "transformers",
    "huggingface-hub",
    "numba",
    "scipy",
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file, streaming in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


EVIDENCE_DIRS: tuple[str, ...] = ("reports", "logs")


def git_revision(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Return ``{"sha": ..., "dirty": ...}`` for the repo, or nulls if git is unavailable.

    ``dirty`` reflects uncommitted changes to code/configs/docs only: generated evidence
    under :data:`EVIDENCE_DIRS` is excluded so that writing one report does not mark the
    next report as produced from a modified tree.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        excludes = [f":(exclude){d}" for d in EVIDENCE_DIRS]
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", *excludes],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout
        return {"sha": sha, "dirty": bool(status.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": None, "dirty": None}


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Return installed versions of the tracked packages (``None`` when not installed)."""
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_snapshot() -> dict[str, Any]:
    """Return a JSON-serializable description of the current execution environment."""
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "git": git_revision(),
        "packages": package_versions(),
    }
