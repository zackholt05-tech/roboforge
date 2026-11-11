"""Validation and §7 selection of the 12 X3 training runs (DECISIONS D-112; D-111).

**Separate replication; not a result.** The frozen ``scripts/run_suite.py`` trained the 12
X3 runs as preregistered (``make x3-train``), then stopped at its first tuning choice: the
frozen ``roboforge.eval.tuning.choose_configuration`` admits only seed-0 tuning trials, and
the X3 plan's one-trial "tuning" seed is 3 (KNOWN_ISSUES KI-062). Each family has one
configuration, fixed before training, so that choice decides nothing. What the suite left
undone is its evaluation work, and this module does that work only:

- every checkpoint of every completed X3 run is evaluated on the 50 Can validation starts by
  the frozen ``scripts/eval_policy.py`` in the canonical container, with the argv and
  environment ``run_suite.py`` uses (pinned by test against its logged launches);
- each run whose checkpoints are all evaluated is selected by the frozen
  ``scripts/select_checkpoint.py`` (RESEARCH_PROTOCOL §7). An existing selection record is
  checked against its sources and never rewritten;
- the outcome is summarised in ``reports/session<N>/suite/x3_validation.json``, which the X3
  gate (``x3_jobs``) requires committed.

Nothing here trains. A run without a final training manifest is waited for while its
``scripts/train.py`` process lives, and needs a manual decision otherwise. The eligibility
rule (eligible, not partial, 50 episodes, exactly one per checkpoint) and the refusal of an
evaluation directory without an eligible summary mirror ``run_suite.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import sha256_file
from roboforge.train.suite import load_plan

SCHEMA = "roboforge.x3_validation/1"
LABEL = "separate replication"
DECISION = "D-112"
PLAN = "configs/experiments/x3_can.yaml"
IN_SIM = "infra/containers/run_sim_cpu.sh"
SUMMARY_NAME = "x3_validation.json"
N_VALIDATION = 50
SELECTION_KEYS = ("step", "sha256", "successes", "n_episodes", "first_success_step_mean",
                  "eval_run_id")


class ValidationError(RuntimeError):
    """A condition that needs a manual decision; nothing more is launched."""


def run_specs(plan_path: Path | None = None) -> list[dict[str, Any]]:
    """The 12 X3 runs in plan family order, then seed: family, seed, run ID, train argv."""
    plan = load_plan(plan_path or REPO_ROOT / PLAN)
    out = []
    for f in plan.families:
        for seed in (plan.tuning_seed, *plan.replicate_seeds):
            spec = plan.spec(f.name, 1, seed)
            out.append({"family": f.name, "seed": int(seed), "run_id": spec.run_id,
                        "train_argv": spec.train_argv(plan.prefix, plan.budget)})
    return out


def eval_argv(run_id: str, step: int, reports_rel: str) -> list[str]:
    """The argv of one validation evaluation, exactly as ``run_suite.launch_eval`` builds it."""
    return [IN_SIM, "python", "scripts/eval_policy.py", "--train-run", f"runs/{run_id}",
            "--steps", str(step), "--reports-dir", reports_rel]


def select_argv(python: str, run_id: str, reports_rel: str) -> list[str]:
    """The argv of one §7 selection, exactly as ``run_suite.select`` builds it."""
    return [python, "scripts/select_checkpoint.py", "--train-run", f"runs/{run_id}",
            "--reports-dir", reports_rel]


def training_alive(train_argv: list[str], commands: Iterable[str]) -> bool:
    """Whether a ``scripts/train.py`` process of this run is among ``commands`` (``ps``)."""
    needle = " ".join(train_argv)
    return any(needle in c for c in commands)


def _read(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text())
    except ValueError:  # half-written (the writers are not atomic, KI-036)
        return None


def training_status(runs_dir: Path, run_id: str, alive: bool) -> tuple[str, dict | None]:
    """``complete`` / ``failed`` (final manifest), ``training`` (no final manifest, process
    alive) or ``missing`` (no final manifest and no process: a manual decision)."""
    p = runs_dir / run_id / "manifest.json"
    m = _read(p) if p.exists() else None
    if m is not None and m.get("status") in ("complete", "failed"):
        return m["status"], m
    if p.exists() and m is None and not alive:
        raise ValidationError(f"{repo_relative(p)} does not parse and no training writes it")
    return ("training" if alive else "missing"), m


def eligible(summary: dict[str, Any]) -> bool:
    """``run_suite.eval_summary``'s test: eligible, not partial, all 50 validation starts."""
    return (bool(summary["eligible_for_selection"]) and not summary["partial"]
            and summary["n_episodes"] == N_VALIDATION)


def eval_summary(reports: Path, run_id: str, step: int, *, busy: bool) -> dict | None:
    """The one eligible validation summary of a checkpoint, or ``None``. ``busy``: its
    evaluation is running (a summary that does not parse is then being written)."""
    hits = sorted((reports / "validation").glob(f"{run_id}__val_step{step:07d}__*.json"))
    good = []
    for h in hits:
        s = _read(h)
        if s is None:
            if busy:
                return None
            raise ValidationError(f"{repo_relative(h)} does not parse and no evaluation "
                                  "writes it")
        if eligible(s):
            good.append(s)
    if len(good) > 1:
        raise ValidationError(f"{run_id} step {step}: {len(good)} eligible evaluations")
    return good[0] if good else None


def partial_eval_dirs(runs_dir: Path, run_id: str, step: int) -> list[Path]:
    """Evaluation record directories of a checkpoint (``run_suite.launch_eval`` refuses to
    launch when one exists without an eligible summary)."""
    return sorted(runs_dir.glob(f"{run_id}__val_step{step:07d}__*"))


def selection_path(reports: Path, run_id: str) -> Path:
    """Where ``scripts/select_checkpoint.py`` writes a run's selection record."""
    return reports / "selection" / f"{run_id}.json"


def check_selection(reports: Path, run_id: str, manifest: dict[str, Any],
                    root: Path = REPO_ROOT) -> dict[str, Any]:
    """A run's existing selection record, checked: its run, a chosen checkpoint of the
    training manifest with that SHA-256, and every source file unchanged since it was read."""
    p = selection_path(reports, run_id)
    rec = json.loads(p.read_text())
    if rec["train_run_id"] != run_id:
        raise ValidationError(f"{repo_relative(p)}: record of {rec['train_run_id']}")
    ck = {int(c["step"]): c["sha256"] for c in manifest["checkpoints"]}
    ch = rec["chosen"]
    if ck.get(int(ch["step"])) != ch["sha256"]:
        raise ValidationError(f"{repo_relative(p)}: chosen checkpoint not in the manifest")
    if len(rec["sources"]) < len(ck):
        raise ValidationError(f"{repo_relative(p)}: {len(rec['sources'])} sources for "
                              f"{len(ck)} checkpoints")
    for s in rec["sources"]:
        if sha256_file(root / s["path"]) != s["sha256"]:
            raise ValidationError(f"{repo_relative(p)}: source {s['path']} changed")
    return rec


def survey(specs: list[dict[str, Any]], reports: Path, runs_dir: Path, busy: set[str],
           commands: list[str]) -> list[dict[str, Any]]:
    """Each run's state: training status, checkpoints still to evaluate, selection."""
    rows = []
    for sp in specs:
        rid = sp["run_id"]
        status, m = training_status(runs_dir, rid, training_alive(sp["train_argv"], commands))
        row = {"family": sp["family"], "seed": sp["seed"], "run_id": rid, "status": status,
               "steps": [], "missing": [], "selected": False, "manifest": m}
        if status == "complete":
            row["steps"] = [int(c["step"]) for c in m["checkpoints"]]
            row["missing"] = [s for s in row["steps"] if f"{rid}@{s}" in busy or eval_summary(
                reports, rid, s, busy=False) is None]
            row["selected"] = selection_path(reports, rid).exists()
        rows.append(row)
    return rows


def resolved(rows: list[dict[str, Any]]) -> bool:
    """Every run either failed in training or is selected."""
    return all(r["status"] == "failed" or (r["status"] == "complete" and r["selected"])
               for r in rows)


def summary_record(rows: list[dict[str, Any]], reports: Path,
                   plan_path: Path | None = None, root: Path = REPO_ROOT) -> dict[str, Any]:
    """The outcome of the 12 runs (each selection rechecked), for the X3 gate."""
    if not resolved(rows):
        raise ValidationError("not every X3 run is selected or failed")
    plan_path = plan_path or REPO_ROOT / PLAN
    runs = []
    for r in rows:
        out = {"family": r["family"], "seed": r["seed"], "run_id": r["run_id"],
               "status": r["status"], "selection": None, "selection_record": None,
               "selection_record_sha256": None}
        if r["status"] == "complete":
            rec = check_selection(reports, r["run_id"], r["manifest"], root)
            p = selection_path(reports, r["run_id"])
            out.update(selection={k: rec["chosen"][k] for k in SELECTION_KEYS},
                       selection_record=repo_relative(p),
                       selection_record_sha256=sha256_file(p), rule=rec["rule"])
        runs.append(out)
    return {"schema": SCHEMA, "label": LABEL, "decision": DECISION,
            "plan": PLAN, "plan_sha256": sha256_file(plan_path), "complete": True,
            "runs": runs, "failed": [r["run_id"] for r in runs if r["status"] == "failed"]}


__all__ = ["DECISION", "IN_SIM", "LABEL", "N_VALIDATION", "PLAN", "SCHEMA", "SUMMARY_NAME",
           "ValidationError", "check_selection", "eligible", "eval_argv", "eval_summary",
           "partial_eval_dirs", "resolved", "run_specs", "select_argv", "selection_path",
           "summary_record", "survey", "training_alive", "training_status"]
