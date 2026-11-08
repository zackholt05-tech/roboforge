"""Host-wide container slots and the disk check (src/roboforge/slots.py, D-069)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from roboforge import slots as sl


def test_at_most_four_slots_across_pools(tmp_path: Path) -> None:
    a, b = sl.ContainerSlots(tmp_path), sl.ContainerSlots(tmp_path)  # two orchestrators
    got = [a.acquire("a0"), b.acquire("b0"), a.acquire("a1"), b.acquire("b1")]
    assert all(got) and len(set(got)) == 4 and a.held() == 4
    assert a.acquire("a2") is None and b.acquire("b2") is None
    sl.ContainerSlots.release(got[1])
    again = b.acquire("b3")
    assert again == got[1] and json.loads(again.read_text())["key"] == "b3"


def test_a_slot_held_only_by_dead_processes_is_reclaimed(tmp_path: Path) -> None:
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                          capture_output=True, text=True, check=True).stdout.strip()
    for i in range(4):
        (tmp_path / f"slot{i}").write_text(json.dumps({"key": "old", "pids": [int(dead)]}))
    pool = sl.ContainerSlots(tmp_path)
    s = pool.acquire("new")
    assert s is not None and json.loads(s.read_text())["key"] == "new"
    pool.attach(s, 12345)
    assert json.loads(s.read_text())["pids"][-1] == 12345


def test_disk_check() -> None:
    assert sl.disk_ok(0.0) and not sl.disk_ok(1e12)
