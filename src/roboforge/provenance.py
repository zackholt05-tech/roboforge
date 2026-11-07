"""Capture software/hardware provenance for run manifests and evidence reports.

The returned dictionaries are embedded in every generated report so a reader can
tell exactly which code revision, interpreter, package versions and rendering
backend produced a given number or image.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
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


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` deterministically: sorted keys, no whitespace, ASCII, repr floats.

    Two equal JSON-compatible objects always give the same string, so its hash identifies a
    spec, split or statistics file independently of key order or formatting.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def sha256_json(obj: Any) -> str:
    """Return the hex SHA-256 of :func:`canonical_json` of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("ascii")).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as stable, human-readable JSON (sorted keys, 1-space indent, final newline).

    Committed manifests are written with this function so that rebuilding them from the same
    inputs yields byte-identical files.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=True,
                               allow_nan=False) + "\n")


EVIDENCE_DIRS: tuple[str, ...] = ("reports", "logs", "runs")


def git_revision(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Return ``{"sha": ..., "dirty": ...}`` for the repo, or nulls if git is unavailable.

    ``dirty`` reflects uncommitted changes to code/configs/docs only: generated evidence
    under :data:`EVIDENCE_DIRS` is excluded so that writing one report does not mark the
    next report as produced from a modified tree. Inside the evaluation container, which has
    no ``.git``, the wrapper script passes the host revision in ``ROBOFORGE_GIT_SHA`` and
    ``ROBOFORGE_GIT_DIRTY``; a Modal job receives the launch plan's revision the same way,
    with ``ROBOFORGE_GIT_SOURCE`` naming where it came from.
    """
    host_sha = os.environ.get("ROBOFORGE_GIT_SHA")
    if host_sha:  # inside a container: the revision of the host tree that launched it
        return {"sha": host_sha, "dirty": os.environ.get("ROBOFORGE_GIT_DIRTY") == "1",
                "source": os.environ.get("ROBOFORGE_GIT_SOURCE",
                                         "host working tree (infra/containers/run_sim_cpu.sh)")}
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


def _cpu_model() -> str | None:
    """Best-effort CPU model string (``sysctl`` on macOS, ``/proc/cpuinfo`` on Linux)."""
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        info = Path("/proc/cpuinfo").read_text()
        for key in ("model name", "CPU part"):
            for line in info.splitlines():
                if line.lower().startswith(key.lower()):
                    return f"{key}: {line.split(':', 1)[1].strip()}"
    except (OSError, subprocess.CalledProcessError):
        return None
    return None


def evaluation_platform() -> dict[str, Any]:
    """Identity of the platform a rollout ran on (DECISIONS D-017).

    Physics is not bit-identical across platforms (KI-013), so every rollout records this and
    results from different ``platform_id`` values are never pooled. Inside the canonical
    container the wrapper ``infra/containers/run_sim_cpu.sh`` exports the image reference and
    ID, which a process cannot otherwise discover about itself.
    """
    image_id = os.environ.get("ROBOFORGE_CONTAINER_IMAGE_ID")
    pid = f"{platform.system().lower()}-{platform.machine()}"
    return {"platform_id": pid + (f"-container-{image_id.split(':')[-1][:12]}" if image_id
                                  else "-native"),
            "system": platform.system(), "machine": platform.machine(),
            "platform": platform.platform(), "python": sys.version.split()[0],
            "cpu": _cpu_model(),
            "container_image": os.environ.get("ROBOFORGE_CONTAINER_IMAGE"),
            "container_image_id": image_id,
            "packages": package_versions(("mujoco", "robosuite", "numpy", "torch"))}


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
