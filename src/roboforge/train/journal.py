"""Run-directory journal that makes a training run resumable and interrupt-safe (Session 5).

A training run may be interrupted by a signal (local Ctrl-C, a Modal preemption, which sends
SIGINT and then SIGKILL 30 s later), by an internal wall-clock deadline set below a remote
job's timeout, or by a hard kill with no warning. The journal records enough on disk, as the
run progresses, to continue it later from the last verified checkpoint. Continuing is
bitwise identical to never stopping on the same platform (``tests/unit/test_resume.py``).

Files in ``runs/<run_id>/`` besides the ones listed in :mod:`roboforge.train.run`:

* ``launch.json``: written once, with exclusive create, when the run directory is created.
  It holds the run ID and the **launch ID**, a nonce of the launch that created the run. A
  restart of the *same* launch (a Modal preemption restarts the function on the same input)
  resumes. A different launch of the same run ID is refused, so a duplicate launch can never
  overwrite or interleave with an existing run.
* ``checkpoints.jsonl``: one line per cadence checkpoint (``checkpoints/step_<N>.pt``),
  appended only after the atomic rename, with its SHA-256. Only these checkpoints enter the
  run manifest and therefore checkpoint selection (RESEARCH_PROTOCOL §7).
* ``resume/step_<N>.pt`` and ``resume.jsonl``: off-cadence checkpoints written when a stop is
  requested between cadence steps. They exist only to continue training. The manifest lists
  them under ``resume_checkpoints``, never under ``checkpoints``.
* ``segments.jsonl``: a ``start`` line when a segment (one process lifetime) begins and an
  ``end`` line when it stops (``complete``, ``interrupted`` or ``aborted``). A start without an
  end is a segment that was killed; the next resume records it as ``lost``.
* ``progress.json``: a heartbeat, replaced atomically at every log row: step, cumulative wall
  and CPU seconds, segment and UTC time.
* ``orphans/<segment>/``: files that no index references, found when resuming (a
  ``.pt.tmp`` from a kill during a write, or a checkpoint renamed into place but killed before
  its index line). They are moved aside, never deleted and never used.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboforge.provenance import sha256_file, write_json

LAUNCH = "launch.json"
CKPT_INDEX = "checkpoints.jsonl"
RESUME_DIR = "resume"
RESUME_INDEX = "resume.jsonl"
SEGMENTS = "segments.jsonl"
PROGRESS = "progress.json"
ORPHANS = "orphans"
METRICS = "metrics.jsonl"
CONFIG = "config.json"
MANIFEST = "manifest.json"
LAUNCH_SCHEMA = "roboforge.train_launch/1"


class RunExistsError(FileExistsError):
    """The run directory exists and belongs to a different launch (duplicate launch)."""


class ResumeError(RuntimeError):
    """A run cannot be resumed (complete, failed, foreign config or corrupted state)."""


def utc_now() -> str:
    """Current UTC time, ISO 8601."""
    return datetime.now(tz=UTC).isoformat()


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


@dataclass(frozen=True)
class ResumePoint:
    """The verified checkpoint a resumed segment starts from."""

    step: int
    path: Path
    sha256: str
    kind: str  # "checkpoint" (cadence) or "resume" (off-cadence)


class RunJournal:
    """Append-only bookkeeping of one run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)

    # ------------------------------------------------------------------ creation and identity
    @classmethod
    def create(cls, run_dir: Path, run_id: str, config: dict[str, Any],
               launch_id: str, launch_info: dict[str, Any] | None = None) -> RunJournal:
        """Create a new run directory (refuses an existing one) with its config and lock."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)  # never reuse a run directory
        j = cls(run_dir)
        write_json(run_dir / CONFIG, config)
        with (run_dir / LAUNCH).open("x") as fh:
            fh.write(json.dumps({"schema": LAUNCH_SCHEMA, "run_id": run_id,
                                 "launch_id": launch_id, "created_utc": utc_now(),
                                 "info": launch_info or {}}, sort_keys=True, indent=1) + "\n")
        return j

    def launch(self) -> dict[str, Any]:
        """The run's ``launch.json``."""
        p = self.run_dir / LAUNCH
        if not p.is_file():
            raise ResumeError(f"{self.run_dir}: no {LAUNCH}; not a resumable run directory")
        return json.loads(p.read_text())

    def check_config(self, config: dict[str, Any]) -> None:
        """Refuse to continue a run with a config other than the one it was created with."""
        stored = json.loads((self.run_dir / CONFIG).read_text())
        if stored != config:
            raise ResumeError(f"{self.run_dir}: stored config differs from the requested one")

    def has_manifest(self) -> bool:
        """Whether the run has ended (complete or failed): it is then immutable."""
        return (self.run_dir / MANIFEST).is_file()

    # ------------------------------------------------------------------ JSON-lines indexes
    def append(self, name: str, row: dict[str, Any]) -> None:
        """Append one JSON line and fsync it."""
        with (self.run_dir / name).open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def read(self, name: str) -> list[dict[str, Any]]:
        """All lines of a JSON-lines index (empty if absent)."""
        p = self.run_dir / name
        if not p.is_file():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def register_checkpoint(self, step: int, rel_path: str, sha256: str, segment: int,
                            *, resume: bool = False) -> None:
        """Index a checkpoint after its atomic write (cadence or resume)."""
        self.append(RESUME_INDEX if resume else CKPT_INDEX,
                    {"step": int(step), "path": rel_path, "sha256": sha256,
                     "segment": int(segment), "saved_utc": utc_now()})

    def checkpoints(self) -> list[dict[str, Any]]:
        """Cadence checkpoints in step order (``step``, ``path``, ``sha256``)."""
        rows = sorted(self.read(CKPT_INDEX), key=lambda r: r["step"])
        steps = [r["step"] for r in rows]
        if len(set(steps)) != len(steps):
            raise ResumeError(f"{self.run_dir}: duplicate cadence checkpoint steps {steps}")
        return rows

    def resume_checkpoints(self) -> list[dict[str, Any]]:
        """Off-cadence resume checkpoints in step order."""
        return sorted(self.read(RESUME_INDEX), key=lambda r: r["step"])

    # ------------------------------------------------------------------ resume
    def latest_resume_point(self, repo_root: Path) -> ResumePoint | None:
        """The indexed checkpoint with the highest step, after verifying its SHA-256.

        A mismatch raises: an indexed checkpoint is written atomically before it is indexed,
        so a mismatch means corrupted storage, and silently falling back to an older file
        would hide it.
        """
        cands = ([(r, "checkpoint") for r in self.checkpoints()]
                 + [(r, "resume") for r in self.resume_checkpoints()])
        if not cands:
            return None
        row, kind = max(cands, key=lambda c: c[0]["step"])
        path = Path(repo_root) / row["path"]
        if not path.is_file():
            raise ResumeError(f"indexed checkpoint {row['path']} is missing")
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise ResumeError(f"indexed checkpoint {row['path']}: SHA-256 {actual} != "
                              f"{row['sha256']}")
        return ResumePoint(int(row["step"]), path, row["sha256"], kind)

    def quarantine_orphans(self, repo_root: Path, segment: int) -> list[str]:
        """Move checkpoint files no index references (and ``.tmp`` files) to ``orphans/``."""
        indexed = {(Path(repo_root) / r["path"]).resolve()
                   for r in self.checkpoints() + self.resume_checkpoints()}
        moved: list[str] = []
        for sub in ("checkpoints", RESUME_DIR):
            d = self.run_dir / sub
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.resolve() in indexed:
                    continue
                dest = self.run_dir / ORPHANS / f"segment{segment:03d}" / sub / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dest))
                moved.append(f"{sub}/{f.name}")
        for f in sorted(self.run_dir.glob("*.tmp")):  # an interrupted atomic JSON write
            dest = self.run_dir / ORPHANS / f"segment{segment:03d}" / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest))
            moved.append(f.name)
        return moved

    def truncate_metrics(self, step: int) -> int:
        """Keep only metric rows with ``step <= step`` (rows written after the resume point
        are regenerated by the resumed segment); return the number of rows dropped."""
        p = self.run_dir / METRICS
        if not p.is_file():
            return 0
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        keep, dropped = [], 0
        for ln in lines:
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:  # a line torn by a kill mid-write
                dropped += 1
                continue
            if int(row["step"]) <= step:
                keep.append(ln)
            else:
                dropped += 1
        _write_atomic(p, "".join(k + "\n" for k in keep))
        return dropped

    # ------------------------------------------------------------------ segments and heartbeat
    def segments(self) -> list[dict[str, Any]]:
        """All segment events in order."""
        return self.read(SEGMENTS)

    def open_segments(self) -> list[dict[str, Any]]:
        """Start events with no matching end or lost event (killed segments)."""
        closed = {e["segment"] for e in self.segments() if e["event"] in ("end", "lost")}
        return [e for e in self.segments() if e["event"] == "start"
                and e["segment"] not in closed]

    def next_segment(self) -> int:
        """Index of the next segment."""
        return 1 + max((e["segment"] for e in self.segments()), default=-1)

    def heartbeat(self, step: int, wall_s: float, cpu_s: float, segment: int) -> None:
        """Replace ``progress.json`` atomically."""
        _write_atomic(self.run_dir / PROGRESS,
                      json.dumps({"step": int(step), "wall_s_total": float(wall_s),
                                  "cpu_s_total": float(cpu_s), "segment": int(segment),
                                  "utc": utc_now()}, sort_keys=True) + "\n")

    def progress(self) -> dict[str, Any] | None:
        """The last heartbeat, if any."""
        p = self.run_dir / PROGRESS
        return json.loads(p.read_text()) if p.is_file() else None

    def close_lost_segments(self, resume_step: int) -> list[dict[str, Any]]:
        """Record every open segment as ``lost``, with what its last heartbeat shows."""
        hb = self.progress()
        lost = []
        for e in self.open_segments():
            row = {"event": "lost", "segment": e["segment"], "launch_id": e["launch_id"],
                   "recorded_utc": utc_now(), "resume_step": int(resume_step),
                   "last_heartbeat": hb if hb and hb["segment"] == e["segment"] else None}
            self.append(SEGMENTS, row)
            lost.append(row)
        return lost

    def totals(self) -> dict[str, float]:
        """Wall and CPU seconds summed over ended segments plus lost segments' heartbeats
        (time a killed segment spent after its last heartbeat is not known locally)."""
        wall = cpu = 0.0
        ev = self.segments()
        for e in ev:
            if e["event"] == "end":
                wall += e["wall_s"]
                cpu += e["cpu_s"]
            elif e["event"] == "lost" and e.get("last_heartbeat"):
                starts = [s for s in ev if s["event"] == "start" and s["segment"] == e["segment"]]
                base = starts[0]
                wall += e["last_heartbeat"]["wall_s_total"] - base["wall_s_before"]
                cpu += e["last_heartbeat"]["cpu_s_total"] - base["cpu_s_before"]
        return {"wall_s": wall, "cpu_s": cpu}


__all__ = ["CKPT_INDEX", "LAUNCH", "MANIFEST", "METRICS", "RESUME_DIR", "RESUME_INDEX",
           "SEGMENTS", "ResumeError", "ResumePoint", "RunExistsError", "RunJournal", "utc_now"]
