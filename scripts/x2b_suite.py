"""X2-b suite: tuning, configuration choice, replicates and selection on X2 validation starts.

DECISIONS D-125 to D-127; ``docs/extensions/X2_PREREGISTRATION.md``. Development numbers only.
From a clean tree, this host orchestrator, resumable from its files at any point:

1. trains the planned runs (``scripts/x2b_train.py``; the X2-a pilot runs count as their
   trials' seed-0 runs and are not retrained, D-125): the tuning runs first (diffusion first),
   then, per family as soon as its four trials are selected, seeds 1 and 2 of the trial the
   frozen §6.3 rule chooses (``reports/session16/x2/suite/choices/<family>.json``);
2. evaluates every checkpoint of every run once on the 50 X2 validation starts in the
   canonical container (``scripts/x2b_eval.py``) as it is indexed, diffusion first;
3. respects the host limits of ``configs/extensions/x2b.yaml``
   (:class:`roboforge.x2.host.HostGuard`): at most 4 trainings, at most 5 jobs (1 container
   beside 4 trainings), one launch per host sample, holds on swap, disk floor and dirty
   tree; replicate trainings launch only when the free disk covers the remaining estimate
   plus the floor; below the hard floor the trainings stop with a resume checkpoint;
4. at the end writes ``reports/session16/x2/suite/{suite.json,tables.md,
   x2_test_configuration.json}`` (selections by the frozen §7 rule).

No evaluation is rerun automatically. A first SIGINT stops launches and asks the trainings to
stop with a resume checkpoint; running evaluations finish::

    ROBOFORGE_EVIDENCE_SESSION=16 python scripts/x2b_suite.py --dry-run
    ROBOFORGE_EVIDENCE_SESSION=16 python scripts/x2b_suite.py
    ROBOFORGE_EVIDENCE_SESSION=16 python scripts/x2b_suite.py --report
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT, evidence_session, repo_relative
from roboforge.provenance import git_revision, write_json
from roboforge.train.journal import RunJournal
from roboforge.x2 import suite as xs
from roboforge.x2.host import HostGuard, free_gib, host_sample, now_utc, strip_counters, tree_clean
from roboforge.x2.pilot import FAMILIES

POLL_S = 20
HOST_LOG_EVERY_S = 300
_STOP = {"requested": False}
sys.path.insert(0, str(Path(__file__).resolve().parent))
from x2b_report import write_report  # noqa: E402


def log_dir() -> Path:
    """``logs/session<N>/x2/suite``."""
    d = REPO_ROOT / "logs" / f"session{evidence_session()}" / "x2" / "suite"
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_dir() -> Path:
    """``reports/session16/x2/suite`` (the preregistered location)."""
    return REPO_ROOT / xs.x2b_config()["out_dir"]


def event(kind: str, **fields: Any) -> None:
    """Append one event to ``events.jsonl`` and echo it."""
    row = {"utc": now_utc(), "event": kind, **fields}
    with (log_dir() / "events.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def preflight() -> list[str]:
    """Reasons not to start (empty = go)."""
    cfg = xs.x2b_config()
    problems = []
    if repo_relative(REPO_ROOT / "reports" / f"session{evidence_session()}" / "x2" / "suite") \
            != cfg["out_dir"]:
        problems.append(f"ROBOFORGE_EVIDENCE_SESSION must give {cfg['out_dir']}")
    if not tree_clean():
        problems.append("working tree is dirty outside reports/logs/runs")
    for rel in (xs.CONFIG, cfg["plan"], "docs/extensions/X2_PREREGISTRATION.md",
                "configs/extensions/x2_pilot.yaml"):
        r = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO_ROOT,
                           capture_output=True)
        if r.returncode:
            problems.append(f"{rel} is not committed")
    if free_gib() < float(cfg["limits"]["min_free_gib"]):
        problems.append(f"free disk {free_gib():.1f} GiB below the floor")
    return problems


def indexed(run: xs.PlannedRun) -> dict[int, dict[str, Any]]:
    """Cadence checkpoints indexed so far by a run's journal."""
    if not (run.run_dir / "checkpoints.jsonl").exists():
        return {}
    return {int(r["step"]): r for r in RunJournal(run.run_dir).checkpoints()}


def choices_now(plan, summaries) -> dict[str, dict[str, Any]]:
    """Per family the §6.3 choice once available; written once, then read back."""
    out = {}
    for fam in FAMILIES:
        p = out_dir() / "choices" / f"{fam}.json"
        if p.exists():
            out[fam] = json.loads(p.read_text())
            continue
        rec = xs.family_choice(plan, fam, summaries)
        if rec is not None:
            write_json(p, rec)
            event("configuration_chosen", family=fam, trial=rec["chosen"]["trial"],
                  successes=rec["chosen"]["successes"],
                  replicates=[r.run_id for r in xs.replicate_runs(plan, fam,
                                                                  rec["chosen"]["trial"])])
            out[fam] = rec
    return out


def all_runs(tuning: list[xs.PlannedRun], replicates: dict[str, list[xs.PlannedRun]],
             plan, choices) -> list[xs.PlannedRun]:
    """Tuning runs, then the replicates of every chosen family (family order); ``replicates``
    caches the resolved replicate runs per family."""
    runs = list(tuning)
    for fam in FAMILIES:
        if fam in choices:
            if fam not in replicates:
                replicates[fam] = xs.replicate_runs(plan, fam,
                                                    int(choices[fam]["chosen"]["trial"]))
            runs += replicates[fam]
    return runs


def launch_train(run: xs.PlannedRun, resume: bool) -> subprocess.Popen:
    """Start one training under ``/usr/bin/time -l`` (peak RSS)."""
    f = (log_dir() / f"train_{run.run_id}.log").open("a")
    cmd = ["/usr/bin/time", "-l", sys.executable, "scripts/x2b_train.py", "--family",
           run.family, "--trial", str(run.trial), "--seed", str(run.seed)]
    if resume:
        cmd.append("--resume")
    return subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})


def launch_eval(run: xs.PlannedRun, step: int, sha: str) -> subprocess.Popen:
    """Start one container evaluation on the X2 validation starts."""
    f = (log_dir() / f"eval_{run.run_id}_step{step:06d}.log").open("a")
    cmd = ["infra/containers/run_sim_cpu.sh", "python", "scripts/x2b_eval.py", "--train-run",
           f"runs/x2/{run.run_id}", "--step", str(step), "--family", run.family,
           "--expected-sha256", sha]
    env = {**os.environ, "ROBOFORGE_EVIDENCE_SESSION": str(evidence_session())}
    return subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)


def remaining_need_gib(runs, summaries, unchosen: list[str]) -> float:
    """Disk estimate for every run not yet trained, every checkpoint not yet evaluated, and
    the 2 replicate runs (40 evaluations) of each family not yet chosen."""
    n_runs: dict[str, int] = {f: 2 for f in unchosen}
    n_evals = 40 * len(unchosen)
    for r in runs:
        done = indexed(r)
        if not (r.run_dir / "manifest.json").exists():
            n_runs[r.family] = n_runs.get(r.family, 0) + 1
        n_evals += 20 - sum((r.run_id, s) in summaries for s in done)
    return xs.remaining_disk_gib(n_runs, n_evals)


def orchestrate() -> int:
    """Run the suite until everything is selected (or SIGINT / a hard stop)."""
    cfg = xs.x2b_config()
    lim = cfg["limits"]
    problems = preflight()
    if problems:
        event("preflight_failed", problems=problems)
        return 2
    plan = xs.x2b_plan()
    event("start", git=git_revision(), limits=lim, pilot_mapping=xs.check_pilot_mapping(plan),
          free_gib=round(free_gib(), 2))
    guard = HostGuard(lim)
    guard.sample()
    tuning = xs.tuning_runs(plan)
    replicates: dict[str, list[xs.PlannedRun]] = {}
    cache: dict[Path, dict[str, Any]] = {}
    trains: dict[str, tuple[xs.PlannedRun, subprocess.Popen]] = {}
    evals: dict[tuple[str, int], subprocess.Popen] = {}
    failed: set[tuple[str, int]] = set()
    train_failed: set[str] = set()
    last_hold, last_train_hold, last_log = None, None, 0.0
    stop_sent = False
    signal.signal(signal.SIGINT, lambda *_: _STOP.update(requested=True))
    while True:
        if guard.due():
            s = guard.sample()
            rate = s["swap_pages_per_s"]
            if time.monotonic() - last_log >= HOST_LOG_EVERY_S or (
                    rate is not None and rate > float(lim["swap_pages_per_s"])):
                with (log_dir() / "host.jsonl").open("a") as f:
                    f.write(json.dumps(strip_counters(s)) + "\n")
                last_log = time.monotonic()
        if guard.hard_stop() and not _STOP["requested"]:
            event("hard_stop_disk", free_gib=round(free_gib(), 2))
            _STOP["requested"] = True
        if _STOP["requested"] and not stop_sent:
            event("stop_requested")
            for _, p in trains.values():
                if p.poll() is None:
                    p.send_signal(signal.SIGINT)
            stop_sent = True
        for rid, (_run, p) in list(trains.items()):
            if p.poll() is not None:
                event("train_ended", run_id=rid, exit=p.returncode)
                if p.returncode not in (0, 3):
                    train_failed.add(rid)
                del trains[rid]
        for key, p in list(evals.items()):
            if p.poll() is not None:
                event("eval_ended", run_id=key[0], step=key[1], exit=p.returncode)
                if p.returncode:
                    failed.add(key)
                del evals[key]
        summaries = xs.load_summaries(cache=cache)
        choices = choices_now(plan, summaries)
        runs = all_runs(tuning, replicates, plan, choices)
        pending_train = [r for r in runs if not (r.run_dir / "manifest.json").exists()
                         and r.run_id not in trains and r.run_id not in train_failed]
        queue = [(r, step, rec["sha256"]) for r in runs for step, rec in sorted(indexed(r).items())
                 if (r.run_id, step) not in summaries and (r.run_id, step) not in evals
                 and (r.run_id, step) not in failed]
        if not trains and not evals and not pending_train and not queue:
            break
        if _STOP["requested"]:
            if not trains and not evals:
                break
            time.sleep(POLL_S)
            continue
        hold = guard.hold_reason()
        launched = None
        if hold is None and pending_train and len(trains) < int(lim["max_train"]) \
                and len(trains) + len(evals) < int(lim["max_jobs"]):
            r = pending_train[0]
            train_hold = None
            if r.seed != plan.tuning_seed:
                unchosen = [f for f in FAMILIES if f not in choices]
                need = remaining_need_gib(runs, summaries, unchosen)
                if free_gib() < float(lim["min_free_gib"]) + need:
                    train_hold = f"disk_for_replicates (need {need:.1f} GiB + floor)"
                    if last_train_hold is None:
                        event("hold", reason=train_hold, free_gib=round(free_gib(), 2))
            last_train_hold = train_hold
            if train_hold is None:
                resume = r.run_dir.exists()
                trains[r.run_id] = (r, launch_train(r, resume))
                launched = ("train", r.run_id)
                event("train_launched", run_id=r.run_id, family=r.family, trial=r.trial,
                      seed=r.seed, resume=resume, running=len(trains))
        cap = min(int(lim["max_containers"]), int(lim["max_jobs"]) - len(trains))
        if launched is None and hold is None and queue and len(evals) < cap:
            r, step, sha = queue[0]
            evals[(r.run_id, step)] = launch_eval(r, step, sha)
            launched = ("eval", r.run_id)
            event("eval_launched", run_id=r.run_id, family=r.family, step=step,
                  running=len(evals), cap=cap, trainings=len(trains))
        if launched:
            guard.launched()
        elif hold not in (None, "await_sample") and hold != last_hold:
            event("hold", reason=hold, free_gib=round(free_gib(), 2),
                  swap_pages_per_s=guard.last and guard.last["swap_pages_per_s"])
        last_hold = hold
        time.sleep(POLL_S)
    event("end", failed_evals=sorted(f"{a}@{b}" for a, b in failed),
          failed_trainings=sorted(train_failed), stopped=_STOP["requested"])
    report = write_report()
    ok = report["complete"] and not failed and not train_failed and not _STOP["requested"]
    return 0 if ok else 1


def dry_run() -> int:
    """Print the plan, the pilot mapping, counts and the disk estimate; launch nothing."""
    plan = xs.x2b_plan()
    summaries = xs.load_summaries()
    rows = [{"family": r.family, "trial": r.trial, "seed": r.seed, "run_id": r.run_id,
             "pilot": r.pilot, "trained": (r.run_dir / "manifest.json").exists(),
             "evaluated": sum(k[0] == r.run_id for k in summaries)}
            for r in xs.tuning_runs(plan)]
    need = remaining_need_gib(xs.tuning_runs(plan), summaries, list(FAMILIES))
    print(json.dumps({"plan": xs.x2b_config()["plan"], "tuning_runs": rows,
                      "pilot_mapping": xs.check_pilot_mapping(plan),
                      "summaries_found": len(summaries),
                      "disk_need_estimate_gib": round(need, 1),
                      "free_gib": round(free_gib(), 1),
                      "host": strip_counters(host_sample(None))}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="plan, mapping and estimates only")
    g.add_argument("--report", action="store_true", help="rewrite the report from the files")
    args = p.parse_args(argv)
    if args.dry_run:
        return dry_run()
    if args.report:
        write_report()
        return 0
    return orchestrate()


if __name__ == "__main__":
    sys.exit(main())
