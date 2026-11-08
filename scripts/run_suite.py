"""Run a main-suite plan on local CPU: train, evaluate, select, choose, replicate.

    caffeinate -i uv run --frozen python scripts/run_suite.py \
        --plan configs/experiments/level1_core_lift.yaml          # ROBOFORGE_EVIDENCE_SESSION=6
    ... --dry-run          # resolve and record the plan (trial -> run ID); launch nothing

The plan (``roboforge.train.suite``) is executed as a dependency graph, never by hand:

1. every tuning trial of every family trains with seed 0 (``scripts/train.py`` with the
   plan's budget and the trial's overrides);
2. each finished run's checkpoints are evaluated one per canonical container
   (``infra/containers/run_sim_cpu.sh python scripts/eval_policy.py --steps N``), all 50
   validation starts, never any other suite;
3. once all of a run's checkpoints are evaluated, ``scripts/select_checkpoint.py`` applies
   RESEARCH_PROTOCOL §7;
4. once all of a family's trials are selected, ``roboforge.eval.tuning`` applies §6.3 and
   writes ``<reports>/tuning/<family>_<task>.json``; the chosen trial is then trained with
   the replicate seeds, which go through steps 2-3 as well.

Concurrency follows Gate C (D-051): at most ``--max-train`` training processes (1 torch
thread each), ``--max-eval`` containers and ``--max-total`` processes in all. Launch
priority: the first family's training (the critical path), then evaluations, then other
training. Every launch requires a clean tree (code, configs and docs committed), unless
``--allow-dirty`` (rehearsals only). Failures are recorded and never retried
automatically; a partial evaluation directory stops the suite for a manual decision.
SIGINT/SIGTERM stop new launches, forward SIGINT to the training processes (each writes a
resume checkpoint, exit 3) and wait for running evaluations; ``--resume-interrupted`` then
continues those runs with ``scripts/train.py --resume``. Events go to
``<log-dir>/events.jsonl``, per-job output to ``<log-dir>/{train,eval,select}/``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboforge.eval.tuning import choose_configuration
from roboforge.paths import REPO_ROOT, evidence_reports_dir, evidence_session, repo_relative
from roboforge.provenance import git_revision, sha256_file
from roboforge.train.suite import RunSpec, SuitePlan, load_plan, plan_table

IN_SIM = "infra/containers/run_sim_cpu.sh"
EXIT_INTERRUPTED = 3


class SuiteError(RuntimeError):
    """A condition that needs a manual decision; the suite stops launching."""


@dataclass
class Job:
    """A running subprocess."""

    kind: str  # train | eval
    key: str
    proc: subprocess.Popen
    log: Path
    started: float = field(default_factory=time.time)
    signalled: bool = False


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Suite:
    """State and scheduling of one plan execution."""

    def __init__(self, plan: SuitePlan, args: argparse.Namespace) -> None:
        self.plan, self.a = plan, args
        self.reports = args.reports_dir
        self.logs = args.log_dir
        self.order = {f.name: i for i, f in enumerate(plan.families)}
        self.pending_train: list[RunSpec] = plan.tuning_runs()
        if plan.replicates_with_tuning:  # one trial per family: nothing to choose (D-068)
            self.pending_train += [s for f in plan.families
                                   for s in plan.replicate_runs(f.name, 1)]
        self.runs: dict[str, RunSpec] = {s.run_id: s for s in self.pending_train}
        self.chosen: dict[str, int] = {}
        self.jobs: list[Job] = []
        self.failed: list[dict[str, Any]] = []
        self.stopping = False
        self.env = {**os.environ, "ROBOFORGE_EVIDENCE_SESSION": str(evidence_session())}
        for spec in list(self.pending_train):  # a restart never retrains a finished run
            m = self.manifest(spec.run_id)
            if m is not None:
                self.pending_train.remove(spec)
                if m["status"] != "complete":
                    self.failed.append({"kind": "train", "key": spec.run_id,
                                        "status": m["status"], "found_at_start": True,
                                        "fatal": m["status"] != "failed"})

    # ------------------------------------------------------------------ bookkeeping
    def event(self, ev: str, /, **kw: Any) -> None:
        """Append one event to ``events.jsonl`` and echo it."""
        rec = {"utc": _now(), "event": ev, **kw}
        with (self.logs / "events.jsonl").open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        print(json.dumps(rec, sort_keys=True), flush=True)

    def manifest(self, rid: str) -> dict[str, Any] | None:
        p = REPO_ROOT / "runs" / rid / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else None

    def eval_summary(self, rid: str, step: int) -> dict[str, Any] | None:
        hits = sorted((self.reports / "validation").glob(f"{rid}__val_step{step:07d}__*.json"))
        good = [json.loads(h.read_text()) for h in hits]
        good = [s for s in good if s["eligible_for_selection"] and not s["partial"]
                and s["n_episodes"] == 50]
        if len(good) > 1:
            raise SuiteError(f"{rid} step {step}: {len(good)} eligible evaluations")
        return good[0] if good else None

    def selection_path(self, rid: str) -> Path:
        return self.reports / "selection" / f"{rid}.json"

    def running(self, kind: str | None = None) -> list[Job]:
        return [j for j in self.jobs if kind is None or j.kind == kind]

    # ------------------------------------------------------------------ launching
    def _launch(self, kind: str, key: str, argv: list[str], log: Path) -> None:
        if not self.a.allow_dirty and git_revision()["dirty"]:
            raise SuiteError("working tree is dirty (code/configs/docs): refusing to launch")
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = log.open("a")
        proc = subprocess.Popen(argv, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                env=self.env, start_new_session=True)
        self.jobs.append(Job(kind, key, proc, log))
        self.event("launch", kind=kind, key=key, argv=argv, pid=proc.pid,
                   log=repo_relative(log))

    def launch_train(self, spec: RunSpec) -> None:
        argv = [sys.executable, *spec.train_argv(self.plan.prefix, self.plan.budget)]
        run_dir = REPO_ROOT / "runs" / spec.run_id
        if run_dir.exists():
            if not self.a.resume_interrupted:
                raise SuiteError(f"{spec.run_id} exists without a manifest (interrupted?); "
                                 "rerun with --resume-interrupted to continue it")
            argv.append("--resume")
        self._launch("train", spec.run_id, argv, self.logs / "train" / f"{spec.run_id}.log")

    def launch_eval(self, rid: str, step: int) -> None:
        if list((REPO_ROOT / "runs").glob(f"{rid}__val_step{step:07d}__*")):
            raise SuiteError(f"{rid} step {step}: an evaluation directory exists without an "
                             "eligible summary (partial or failed evaluation)")
        argv = [IN_SIM, "python", "scripts/eval_policy.py", "--train-run", f"runs/{rid}",
                "--steps", str(step), "--reports-dir", repo_relative(self.reports)]
        self._launch("eval", f"{rid}@{step}", argv,
                     self.logs / "eval" / f"{rid}_step{step:07d}.log")

    def select(self, rid: str) -> None:
        log = self.logs / "select" / f"{rid}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, "scripts/select_checkpoint.py", "--train-run", f"runs/{rid}",
                "--reports-dir", repo_relative(self.reports)]
        with log.open("a") as fh:
            rc = subprocess.run(argv, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                env=self.env, check=False).returncode
        if rc:
            raise SuiteError(f"selection of {rid} failed (exit {rc}); see {log}")
        chosen = json.loads(self.selection_path(rid).read_text())["chosen"]
        self.event("selected", key=rid, step=chosen["step"], successes=chosen["successes"],
                   first_success_step_mean=chosen["first_success_step_mean"])

    # ------------------------------------------------------------------ state machine
    def reap(self) -> None:
        """Collect finished jobs and record their outcome."""
        for j in list(self.jobs):
            rc = j.proc.poll()
            if rc is None:
                continue
            self.jobs.remove(j)
            info = {"kind": j.kind, "key": j.key, "returncode": rc,
                    "wall_s": round(time.time() - j.started, 1)}
            if j.kind == "train":
                m = self.manifest(j.key)
                resumable = (REPO_ROOT / "runs" / j.key / "resume.jsonl").exists()
                status = m["status"] if m else (
                    "interrupted" if rc == EXIT_INTERRUPTED or resumable else "no manifest")
                info["status"] = status
                if status != "complete":
                    # A run that failed with a manifest (e.g. a non-finite loss) is a recorded
                    # outcome: the choice rule ranks it last. Anything else needs a decision.
                    self.failed.append({**info, "fatal": status != "failed"})
            elif rc:
                self.failed.append({**info, "fatal": True})
            self.event("finished", **info)

    def tick(self) -> bool:
        """Advance the graph by one step; return False when nothing is left to do."""
        self.reap()
        if any(f["fatal"] for f in self.failed) and not self.a.keep_going:
            self.stopping = True
        busy = {j.key for j in self.jobs}
        # Selection and tuning choices (synchronous, quick).
        complete = {}
        for rid in self.runs:
            m = self.manifest(rid)
            if m and m["status"] == "complete" and rid not in busy:
                complete[rid] = m
        eval_queue: list[tuple[int, int, str, int]] = []
        for i, (rid, m) in enumerate(complete.items()):
            steps = [c["step"] for c in m["checkpoints"]]
            missing = [s for s in steps if self.eval_summary(rid, s) is None]
            if missing:
                fam = self.runs[rid].family
                eval_queue += [(self.order[fam], i, rid, s) for s in missing
                               if f"{rid}@{s}" not in busy]
            elif not self.selection_path(rid).exists() and not self.stopping:
                self.select(rid)
        for fam in self.plan.families:
            if fam.name not in self.chosen and not self.stopping:
                self.maybe_choose(fam.name, complete)
        if self.stopping:
            return bool(self.jobs)
        # Launches, in priority order, within the Gate C limits.
        critical = self.plan.families[0].name
        trains = sorted(self.pending_train, key=lambda s: (s.family != critical,
                                                           self.order[s.family]))
        evals = sorted(eval_queue)
        for spec in [s for s in trains if s.family == critical]:
            if self._room("train"):
                self.pending_train.remove(spec)
                self.launch_train(spec)
        for _, _, rid, step in evals:
            if self._room("eval"):
                self.launch_eval(rid, step)
        for spec in [s for s in trains if s.family != critical]:
            if self._room("train"):
                self.pending_train.remove(spec)
                self.launch_train(spec)
        self.write_status(len(evals))
        resolved = all(self.selection_path(r).exists() or (
            (m := self.manifest(r)) is not None and m["status"] == "failed") for r in self.runs)
        done = (not self.jobs and not self.pending_train and not evals
                and len(self.chosen) == len(self.plan.families) and resolved)
        return not done

    def _room(self, kind: str) -> bool:
        cap = self.a.max_train if kind == "train" else self.a.max_eval
        return len(self.running(kind)) < cap and len(self.jobs) < self.a.max_total

    def maybe_choose(self, family: str, complete: dict[str, dict[str, Any]]) -> None:
        fam = next(f for f in self.plan.families if f.name == family)
        specs = [self.plan.spec(family, t, self.plan.tuning_seed) for t in sorted(fam.trials)]
        manifests = {s.run_id: self.manifest(s.run_id) for s in specs}
        failed = {r for r, m in manifests.items() if m and m["status"] == "failed"}
        if not all(r in failed or (r in complete and self.selection_path(r).exists())
                   for r in manifests):
            return
        trials = [{"trial": s.trial, "overrides": list(s.overrides),
                   "manifest": manifests[s.run_id],
                   "selection": None if s.run_id in failed else json.loads(
                       self.selection_path(s.run_id).read_text())} for s in specs]
        rec = choose_configuration(family, self.plan.task, trials, sorted(fam.trials))
        rec["plan"] = {"path": repo_relative(Path(self.plan.path)),
                       "sha256": sha256_file(Path(self.plan.path))}
        rec["sources"] = [{"path": repo_relative(self.selection_path(s.run_id)),
                           "sha256": sha256_file(self.selection_path(s.run_id))}
                          for s in specs if s.run_id not in failed]
        out = self.reports / "tuning" / f"{family}_{self.plan.task}.json"
        if out.exists():
            def essence(r: dict[str, Any]) -> list[tuple[Any, ...]]:
                keys = ("trial", "train_run_id", "selected_sha256", "successes",
                        "first_success_step_mean")
                return [tuple(c.get(k) for k in keys) for c in (r["chosen"], *r["candidates"])]
            if essence(json.loads(out.read_text())) != essence(rec):
                raise SuiteError(f"{out} exists and differs from the recomputed choice")
        else:
            rec["git"], rec["generated_utc"] = git_revision(), _now()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n")
        trial = int(rec["chosen"]["trial"])
        self.chosen[family] = trial
        reps = self.plan.replicate_runs(family, trial)
        for spec in reps:
            self.runs[spec.run_id] = spec
            if self.manifest(spec.run_id) is None and spec.run_id not in {
                    j.key for j in self.jobs} and spec not in self.pending_train:
                self.pending_train.append(spec)
        self.event("chosen", family=family, trial=trial, record=repo_relative(out),
                   replicates=[s.run_id for s in reps])

    def write_status(self, n_eval_queue: int) -> None:
        st = {"utc": _now(), "running": [{"kind": j.kind, "key": j.key,
                                          "elapsed_s": round(time.time() - j.started)}
                                         for j in self.jobs],
              "pending_train": [s.run_id for s in self.pending_train],
              "eval_queue": n_eval_queue, "chosen": self.chosen, "failed": self.failed,
              "stopping": self.stopping}
        (self.logs / "status.json").write_text(json.dumps(st, indent=1) + "\n")

    def summarize(self) -> Path:
        """Write ``<reports>/suite/<plan stem>.json``: every run's selected checkpoint."""
        fams = []
        for fam in self.plan.families:
            tpath = self.reports / "tuning" / f"{fam.name}_{self.plan.task}.json"
            trial = self.chosen.get(fam.name)
            specs = [self.plan.spec(fam.name, t, self.plan.tuning_seed)
                     for t in sorted(fam.trials)]
            if trial is not None:
                specs += self.plan.replicate_runs(fam.name, trial)
            rows = []
            for s in specs:
                sp = self.selection_path(s.run_id)
                sel = json.loads(sp.read_text())["chosen"] if sp.exists() else None
                m = self.manifest(s.run_id)
                rows.append({"run_id": s.run_id, "trial": s.trial, "seed": s.seed,
                             "role": "tuning" if s.seed == self.plan.tuning_seed
                             else "replicate",
                             "status": m["status"] if m else None,
                             "selection": None if sel is None else {
                                 k: sel[k] for k in ("step", "sha256", "successes",
                                                     "n_episodes", "first_success_step_mean",
                                                     "eval_run_id")},
                             "selection_record": repo_relative(sp) if sel else None})
            fams.append({"family": fam.name, "chosen_trial": trial,
                         "tuning_record": repo_relative(tpath) if tpath.exists() else None,
                         "tuning_record_sha256": sha256_file(tpath) if tpath.exists() else None,
                         "runs": rows})
        out = self.reports / "suite" / f"{Path(self.plan.path).stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schema": "roboforge.suite_summary/1", "label": "VALIDATION ONLY (development "
            "numbers; selection used the same 50 validation starts)",
            "plan": repo_relative(Path(self.plan.path)),
            "plan_sha256": sha256_file(Path(self.plan.path)), "task": self.plan.task,
            "families": fams, "failed": self.failed, "git": git_revision(),
            "generated_utc": _now()}, indent=1, allow_nan=False) + "\n")
        return out

    def stop(self, signum: int, _frame: Any) -> None:
        """Stop launching; ask training runs to write a resume checkpoint."""
        self.stopping = True
        self.event("stop_requested", signal=signum)
        for j in self.running("train"):
            if not j.signalled:  # once: a second SIGINT after the handler is gone kills it
                j.signalled = True
                j.proc.send_signal(signal.SIGINT)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--reports-dir", type=Path, default=None,
                   help="default: reports/session$ROBOFORGE_EVIDENCE_SESSION")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="default: logs/session$N/suite/<plan stem>")
    p.add_argument("--max-train", type=int, default=8)
    p.add_argument("--max-eval", type=int, default=4)
    p.add_argument("--max-total", type=int, default=10)
    p.add_argument("--poll-s", type=float, default=15.0)
    p.add_argument("--dry-run", action="store_true", help="resolve and record the plan only")
    p.add_argument("--allow-dirty", action="store_true", help="rehearsals only")
    p.add_argument("--resume-interrupted", action="store_true")
    p.add_argument("--keep-going", action="store_true",
                   help="keep launching unrelated work after a failure")
    a = p.parse_args(argv)
    if a.max_train > 8 or a.max_eval > 4:
        p.error("Gate C (D-051) allows at most 8 training runs and 4 containers")
    plan = load_plan(a.plan)
    a.reports_dir = a.reports_dir or evidence_reports_dir()
    a.log_dir = a.log_dir or (REPO_ROOT / "logs" / f"session{evidence_session()}" / "suite"
                              / a.plan.stem)
    a.log_dir.mkdir(parents=True, exist_ok=True)
    table = plan_table(plan)
    rec = {"schema": "roboforge.suite_plan_resolved/1", "plan": repo_relative(a.plan),
           "plan_sha256": sha256_file(a.plan), "task": plan.task, "prefix": plan.prefix,
           "budget": list(plan.budget), "tuning_seed": plan.tuning_seed,
           "replicate_seeds": list(plan.replicate_seeds), "tuning_runs": table,
           "git": git_revision(), "generated_utc": _now()}
    (a.log_dir / "plan_resolved.json").write_text(json.dumps(rec, indent=1) + "\n")
    for r in table:
        print(f"{r['family']:15s} trial {r['trial']} {r['param_count']:>9,d} params "
              f"{' '.join(r['overrides']) or '(base config)':45s} -> {r['run_id']}")
    if a.dry_run:
        return 0
    suite = Suite(plan, a)
    signal.signal(signal.SIGINT, suite.stop)
    signal.signal(signal.SIGTERM, suite.stop)
    suite.event("start", plan=repo_relative(a.plan), plan_sha256=rec["plan_sha256"],
                git=rec["git"], limits={"train": a.max_train, "eval": a.max_eval,
                                        "total": a.max_total})
    code = 0
    try:
        while suite.tick():
            time.sleep(a.poll_s)
    except SuiteError as e:
        suite.event("suite_error", error=str(e))
        suite.stopping, code = True, 2
        while suite.jobs:  # never orphan running jobs
            time.sleep(a.poll_s)
            suite.reap()
    suite.write_status(0)
    if any(f["fatal"] for f in suite.failed):
        code = code or 1
    if suite.stopping and not code:
        code = EXIT_INTERRUPTED  # stopped by a signal before the plan was done
    summary = suite.summarize()
    suite.event("end", failed=suite.failed, chosen=suite.chosen, exit=code,
                summary=repo_relative(summary))
    return code


if __name__ == "__main__":
    sys.exit(main())
