"""Host-wide limits shared by every orchestrator (Gate C, DECISIONS D-051, D-069).

Each orchestrator (``scripts/run_suite.py``, ``scripts/run_evals.py``,
``scripts/run_heldout.py``) enforces its own ``--max-eval``; when two run at once, their
limits would add up. :class:`ContainerSlots` is one host-wide pool of evaluation-container
slots (default 4) that they all draw from: a slot is a file
``logs/container_slots/slot<i>`` created atomically (``O_CREAT | O_EXCL``) holding the PIDs
of the orchestrator and of the launched container client. A slot whose PIDs are all dead
(an orchestrator that crashed) is reclaimed. :func:`disk_ok` is the free-space check made
before every launch.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from roboforge.paths import REPO_ROOT

MAX_CONTAINERS = 4
MIN_FREE_GB = 4.0
SLOTS_DIR = REPO_ROOT / "logs" / "container_slots"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ContainerSlots:
    """The host-wide pool of :data:`MAX_CONTAINERS` evaluation-container slots (each
    orchestrator still applies its own, possibly lower, ``--max-eval``)."""

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = Path(directory or SLOTS_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _stale(self, path: Path) -> bool:
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):  # being written right now, or left half-written
            try:
                return time.time() - path.stat().st_mtime > 60
            except OSError:
                return False
        return not any(_alive(int(p)) for p in rec.get("pids", []))

    def acquire(self, key: str) -> Path | None:
        """A free slot for ``key`` (held by this process), or ``None`` when all are taken."""
        for i in range(MAX_CONTAINERS):
            path = self.dir / f"slot{i}"
            for _ in range(2):
                try:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                except FileExistsError:
                    if self._stale(path):
                        path.unlink(missing_ok=True)
                        continue
                    break
                with os.fdopen(fd, "w") as f:
                    json.dump({"key": key, "pids": [os.getpid()],
                               "utc": datetime.now(tz=UTC).isoformat()}, f)
                return path
        return None

    def held(self) -> int:
        """Slots currently taken (by any orchestrator)."""
        return sum(1 for i in range(MAX_CONTAINERS) if (self.dir / f"slot{i}").exists())

    def attach(self, path: Path, pid: int) -> None:
        """Record the launched container client's PID in the slot."""
        rec = json.loads(path.read_text())
        rec["pids"] = [*rec["pids"], int(pid)]
        path.write_text(json.dumps(rec))

    @staticmethod
    def release(path: Path | None) -> None:
        """Free a slot."""
        if path is not None:
            path.unlink(missing_ok=True)


def free_gb(path: Path = REPO_ROOT) -> float:
    """Free disk space of the repository's volume, in GiB."""
    return shutil.disk_usage(path).free / 2**30


def disk_ok(min_gb: float = MIN_FREE_GB, path: Path = REPO_ROOT) -> bool:
    """Whether at least ``min_gb`` GiB are free (checked before every launch)."""
    return free_gb(path) >= min_gb
