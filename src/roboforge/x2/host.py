"""Host guards for X2-b orchestration on the 16 GB macOS host (KI-035, KI-008, D-113, D-125).

The same measurements as the X2-a pilot's orchestrator (``scripts/x2_pilot.py``, left
unchanged): free disk of the repository volume, the macOS swap rate from ``vm_stat`` counters,
swap use, load and container memory. :class:`HostGuard` adds what the pilot lacked
(D-125): a baseline sample before the first launch, at most one launch per sample (so each
launch's effect is measured before the next), and a hard disk floor at which running
trainings are stopped with a resume checkpoint.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from typing import Any

from roboforge.paths import REPO_ROOT


def now_utc() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(tz=UTC).isoformat()


def tree_clean() -> bool:
    """No change outside reports/, logs/ and runs/ (the container's dirty-flag rule)."""
    out = subprocess.run(["git", "status", "--porcelain", "--", ".", ":(exclude)reports",
                          ":(exclude)logs", ":(exclude)runs"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout
    return not out.strip()


def free_gib() -> float:
    """Free disk space of the repository volume, GiB."""
    return shutil.disk_usage(REPO_ROOT).free / 2**30


def swap_counters() -> tuple[int, int]:
    """macOS cumulative (swap-ins, swap-outs) from ``vm_stat``."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout

    def get(key: str) -> int:
        return int(re.search(rf"{key}:\s+(\d+)", out).group(1))

    return get("Swapins"), get("Swapouts")


def host_sample(prev: dict[str, Any] | None) -> dict[str, Any]:
    """Swap rate since ``prev`` (pages/s), swap use, load, free disk and container memory."""
    si, so = swap_counters()
    t = time.monotonic()
    rate = None
    if prev is not None and t > prev["t"]:
        rate = ((si - prev["si"]) + (so - prev["so"])) / (t - prev["t"])
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                          text=True).stdout.strip()
    stats = subprocess.run(["docker", "stats", "--no-stream", "--format",
                            "{{.Name}} {{.MemUsage}} {{.CPUPerc}}"], capture_output=True,
                           text=True).stdout.strip().splitlines()
    return {"t": t, "si": si, "so": so, "utc": now_utc(), "swap_pages_per_s": rate,
            "swapusage": swap, "loadavg": os.getloadavg(), "free_gib": round(free_gib(), 2),
            "containers": stats}


class HostGuard:
    """Samples the host every ``every_s`` seconds and decides whether a launch may happen."""

    def __init__(self, limits: dict[str, Any]) -> None:
        self.limits = limits
        self.every_s = float(limits["sample_every_s"])
        self.prev: dict[str, Any] | None = None
        self.last: dict[str, Any] | None = None
        self.last_launch_t = -1.0

    def due(self) -> bool:
        """True when the next sample is due (or none has a rate yet)."""
        return self.last is None or time.monotonic() - self.last["t"] >= self.every_s

    def sample(self) -> dict[str, Any]:
        """Take a sample; the first call only sets the baseline counters (rate None)."""
        s = host_sample(self.prev)
        self.prev = s
        if s["swap_pages_per_s"] is not None:
            self.last = s
        return s

    def hold_reason(self) -> str | None:
        """Why no launch may happen now, or None. A launch needs a rate measured after the
        previous launch, a swap rate at or below the limit, the disk floor and a clean tree."""
        if self.last is None or self.last["t"] <= self.last_launch_t:
            return "await_sample"
        if float(self.last["swap_pages_per_s"]) > float(self.limits["swap_pages_per_s"]):
            return "swap"
        if free_gib() < float(self.limits["min_free_gib"]):
            return "disk"
        if not tree_clean():
            return "dirty_tree"
        return None

    def launched(self) -> None:
        """Record a launch: the next one waits for a fresh sample."""
        self.last_launch_t = time.monotonic()

    def hard_stop(self) -> bool:
        """Free disk below the hard floor: stop trainings, launch nothing."""
        return free_gib() < float(self.limits["hard_stop_free_gib"])


def strip_counters(s: dict[str, Any]) -> dict[str, Any]:
    """A sample without its internal counters, for ``host.jsonl``."""
    return {k: v for k, v in s.items() if k not in ("t", "si", "so")}
