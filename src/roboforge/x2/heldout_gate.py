"""X2-c on the host: the output scan, the gate and the isolation snapshot (DECISIONS D-130).

Host side only (it asks git). Used by ``scripts/x2c_run.py`` before any X2-c evaluation
launches and by ``scripts/analyze_x2c.py`` before the analysis.

**The output scan** (:func:`existing_heldout_outputs`) finds anything that looks like an X2
held-out evaluation anywhere in ``reports/``, ``logs/`` and ``runs/``: a file or directory
whose name carries the held-out evaluation-ID tag or a held-out state ID, and any JSON file
under ``reports/`` or ``logs/`` that is an evaluation summary of the held-out suite.

**The gate** (:func:`check_gate`): a clean tree; the X2 freeze committed unmodified and
parsed; no file it pins changed; the Level 1 frozen code unchanged; the preregistration and
the X2-b test configuration equal to the freeze's pins; the held-out suite's files committed
unmodified and equal to the freeze's content hash; every frozen checkpoint file re-hashed. At
the first launch no held-out output may exist anywhere. A later launch (a resumed batch, or
the one engineering rerun) needs a recorded decision, and every existing output must then
belong to one of the 12 frozen jobs.

**The isolation snapshot** (:func:`isolation_snapshot`): the top-level listing of ``runs/``
and digests (names and sizes) of ``runs/x2``, ``runs/x3`` and the frozen Level 1 held-out
record directories, before and after the evaluations.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import sha256_file, tracked_and_unmodified
from roboforge.x2 import heldout as xh
from roboforge.x2 import states as x2s
from roboforge.x2.host import tree_clean

RUNS_SCAN_DEPTH = 3
DECISIONS_REL = "docs/DECISIONS.md"


class X2cGateError(RuntimeError):
    """A precondition of the X2-c evaluations or analysis does not hold."""


def eval_tag() -> str:
    """The suite tag every held-out evaluation ID carries (frozen evaluation-ID pattern)."""
    return f"__{x2s.HELDOUT_SUITE}_step"


def state_prefix() -> str:
    """The prefix of every held-out state ID (record directories are named by state ID)."""
    return f"{x2s.suite_task()}_{x2s.HELDOUT_SUITE}_"


def _named(name: str) -> bool:
    return eval_tag() in name or name.startswith(state_prefix())


def _walk(root: Path, max_depth: int | None) -> list[Path]:
    out, stack = [], [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(os.scandir(d))
        except (FileNotFoundError, NotADirectoryError):
            continue
        for e in entries:
            p = Path(e.path)
            out.append(p)
            if e.is_dir(follow_symlinks=False) and (max_depth is None or depth + 1 < max_depth):
                stack.append((p, depth + 1))
    return out


def existing_heldout_outputs(repo: Path = REPO_ROOT) -> list[str]:
    """Repository-relative paths of every held-out output found (empty = none)."""
    found = set()
    for root, depth in (("reports", None), ("logs", None), ("runs", RUNS_SCAN_DEPTH)):
        for p in _walk(repo / root, depth):
            if _named(p.name):
                found.add(p.relative_to(repo).as_posix())
    for root in ("reports", "logs"):
        for p in (repo / root).rglob("*.json"):
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError, UnicodeDecodeError):
                continue
            if (isinstance(d, dict) and d.get("suite") == x2s.HELDOUT_SUITE
                    and isinstance(d.get("episodes"), list)):
                found.add(p.relative_to(repo).as_posix())
    return sorted(found)


def job_paths(job: dict[str, Any], session: int) -> dict[str, str]:
    """Where one frozen job's outputs live (repository-relative)."""
    eid = job["eval_run_id"]
    return {"summary": f"reports/session{session}/{'/'.join(xh.SUMMARY_PARTS)}/{eid}.json",
            "records": f"runs/{xh.RUNS_SUBDIR[False]}/{eid}",
            "records_attempt2": f"runs/{xh.RUNS_SUBDIR[False]}/{xh.ATTEMPT2_SUBDIR}/{eid}",
            "log": f"logs/session{session}/x2/heldout/jobs/{eid}.log"}


def job_status(job: dict[str, Any], session: int, repo: Path = REPO_ROOT) -> str:
    """``done`` (summary written), ``todo`` (nothing), ``failed1`` (attempt-1 records without a
    summary: the one engineering rerun may run) or ``failed2`` (attempt 2 failed too)."""
    p = job_paths(job, session)
    if (repo / p["summary"]).exists():
        return "done"
    if (repo / p["records_attempt2"]).exists():
        return "failed2"
    if (repo / p["records"]).exists():
        return "failed1"
    return "todo"


def foreign_outputs(found: list[str], jobs: list[dict[str, Any]], session: int) -> list[str]:
    """Outputs that do not belong to any frozen job (anything in a record directory of a job
    belongs to it)."""
    mine = [v for j in jobs for v in job_paths(j, session).values()]
    return [f for f in found if not any(f == m or f.startswith(m + "/") for m in mine)]


def decision_recorded(decision: str) -> bool:
    """True if ``### <decision> `` is a heading of the committed, unmodified DECISIONS.md."""
    ok, _ = tracked_and_unmodified(DECISIONS_REL)
    if not ok or not xh.DECISION_ID.fullmatch(decision):
        return False
    text = subprocess.run(["git", "show", f"HEAD:{DECISIONS_REL}"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout
    return any(line.startswith(f"### {decision} ") for line in text.splitlines())


def check_inputs(freeze: dict[str, Any]) -> dict[str, Any]:
    """The pins of the freeze, re-checked: pinned code, the Level 1 frozen code, the
    preregistration, the test configuration, the held-out suite, the 12 checkpoint files."""
    from roboforge.eval import freeze as level1
    from roboforge.sim import eval_states as es
    from roboforge.train.run import read_manifest as read_train_manifest

    bad = xh.code_problems(freeze)
    if bad:
        raise X2cGateError(f"files pinned by the X2 freeze changed: {bad}")
    bad = level1.code_problems(level1.load_freeze())
    if bad:
        raise X2cGateError(f"Level 1 frozen code changed: {bad}")
    for rel, sha in ((xh.PREREGISTRATION, freeze["preregistration"]["sha256"]),
                     (xh.TEST_CONFIGURATION, xh.TEST_CONFIGURATION_SHA256)):
        ok, why = tracked_and_unmodified(rel)
        if not ok or sha256_file(REPO_ROOT / rel) != sha:
            raise X2cGateError(f"{rel}: not committed unmodified with SHA-256 {sha} ({why})")
    task, name = freeze["suite"]["task"], freeze["suite"]["name"]
    for rel in (repo_relative(es.manifest_path(task, name)),
                repo_relative(es.data_path(task, name))):
        ok, why = tracked_and_unmodified(rel)
        if not ok:
            raise X2cGateError(f"X2 held-out suite not usable: {why}")
    m = es.read_manifest(task, name)
    if m["content_sha256"] != freeze["suite"]["manifest_content_sha256"] or not \
            es.verify_data_file(task, name):
        raise X2cGateError(f"{task}/{name}: suite files differ from the X2 freeze")
    jobs = xh.frozen_jobs(freeze)
    for j in jobs:
        tm = read_train_manifest(REPO_ROOT / xh.TRAIN_RUNS / j["train_run_id"])
        ck = {int(c["step"]): c for c in tm["checkpoints"]}.get(j["step"])
        path = None if ck is None else Path(ck["path"])
        if (tm["status"] != "complete" or ck is None or ck["sha256"] != j["checkpoint_sha256"]
                or int(tm["config"]["seed"]) != j["seed"]):
            raise X2cGateError(f"{j['train_run_id']}: manifest does not hold the frozen "
                               f"checkpoint at step {j['step']}")
        path = path if path.is_absolute() else REPO_ROOT / path
        if sha256_file(path) != j["checkpoint_sha256"]:
            raise X2cGateError(f"{path}: SHA-256 differs from the X2 freeze")
    return {"suite": name, "content_sha256": m["content_sha256"], "jobs": jobs}


def check_gate(session: int, *, decision: str | None = None) -> dict[str, Any]:
    """The X2-c launch gate. Without ``decision`` no held-out output may exist anywhere;
    with a recorded ``decision`` every existing output must belong to a frozen job."""
    if not tree_clean():
        raise X2cGateError("the working tree is dirty (code, configs or docs)")
    try:
        freeze = xh.load_x2_freeze()
    except xh.X2FreezeError as e:
        raise X2cGateError(str(e)) from e
    blob = xh.freeze_blob()
    inputs = check_inputs(freeze)
    jobs = inputs["jobs"]
    want = f"reports/session{session}/{'/'.join(xh.SUMMARY_PARTS)}"
    if freeze["outputs"]["summaries"] != want:
        raise X2cGateError(f"the X2 freeze puts summaries in {freeze['outputs']['summaries']}, "
                           f"not {want} (ROBOFORGE_EVIDENCE_SESSION)")
    found = existing_heldout_outputs()
    if decision is None and found:
        raise X2cGateError(f"held-out outputs already exist (X2-c runs once): {found[:5]}")
    if decision is not None:
        if not decision_recorded(decision):
            raise X2cGateError(f"{decision} is not a heading of the committed {DECISIONS_REL}")
        foreign = foreign_outputs(found, jobs, session)
        if foreign:
            raise X2cGateError(f"held-out outputs that belong to no frozen job: {foreign[:5]}")
    status = {j["eval_run_id"]: job_status(j, session) for j in jobs}
    if any(v == "failed2" for v in status.values()):
        raise X2cGateError(f"an evaluation failed twice; no third attempt: {status}")
    return {"freeze_blob": blob, "freeze": freeze, "jobs": jobs, "status": status,
            "existing": found, "decision": decision, "suite": inputs["suite"],
            "content_sha256": inputs["content_sha256"]}


def _digest(paths: list[Path], repo: Path) -> dict[str, Any]:
    h, n = hashlib.sha256(), 0
    for f in sorted(paths):
        if f.is_file():
            h.update(f"{f.relative_to(repo).as_posix()}\t{f.stat().st_size}\n".encode())
            n += 1
    return {"n_files": n, "sha256": h.hexdigest()}


def isolation_snapshot(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """The ``runs/`` listing and digests of the frozen record trees (names and sizes): ``runs/x2``,
    ``runs/x3`` and the 26 frozen Level 1 held-out record directories (``runs/*__heldout*``)."""
    runs = repo / "runs"
    top = sorted(os.listdir(runs))
    frozen_dirs = sorted(runs.glob("*__heldout*"))
    frozen = [p for d in frozen_dirs for p in _walk(d, None)]
    return {"runs_top_level": {"n": len(top),
                               "sha256": hashlib.sha256("\n".join(top).encode()).hexdigest(),
                               "names": top},
            "runs_x2": _digest(_walk(runs / "x2", None), repo),
            "runs_x3": _digest(_walk(runs / "x3", None), repo),
            "frozen_level1_heldout": {**_digest(frozen, repo), "n_dirs": len(frozen_dirs)},
            "runs_x2c": sorted(os.listdir(runs / xh.RUNS_SUBDIR[False]))
            if (runs / xh.RUNS_SUBDIR[False]).is_dir() else []}


def isolation_problems(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Why ``after`` is not ``before`` plus only the X2-c record directory."""
    out = []
    new = sorted(set(after["runs_top_level"]["names"]) - set(before["runs_top_level"]["names"]))
    gone = sorted(set(before["runs_top_level"]["names"]) - set(after["runs_top_level"]["names"]))
    if gone or any(n != xh.RUNS_SUBDIR[False] for n in new):
        out.append(f"runs/: new {new}, removed {gone}")
    for key in ("runs_x2", "runs_x3", "frozen_level1_heldout"):
        if before[key] != after[key]:
            out.append(f"{key} changed: {before[key]} -> {after[key]}")
    return out
