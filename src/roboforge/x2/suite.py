"""X2-b: the tuning and replicate plan on Can MH, and selection from files (D-125 to D-127).

**Plan.** ``configs/experiments/x2_can_mh.yaml`` is a frozen-schema suite plan (Level 1's
four trials per family, seed 0, then seeds 1 and 2 of the chosen trial), loaded and resolved
by the frozen :func:`roboforge.train.suite.load_plan`. X2-b runs inside the X2 data binding
(D-121), so it has its own drivers (KI-064); the frozen ``run_suite.py`` never sees it.

**Pilot runs** (D-125). A planned seed-0 run whose resolved config SHA-256 equals an X2-a
pilot run's (D-124) *is* that pilot run: training is deterministic on CPU with one thread, and
the config, data, binding and trainer are the ones the pilot used. Its run directory is
``runs/x2/s15x2_*``; its 15 unevaluated checkpoints are evaluated like any trial's.

**Selection.** Every checkpoint of every run is evaluated once on all 50 X2 validation starts
(``scripts/x2b_eval.py``). Summaries come from the X2-a pilot's directory and X2-b's
(``summary_dirs`` of ``configs/extensions/x2b.yaml``); a step evaluated twice raises. The §7
rule is the frozen :func:`roboforge.eval.selection.select_checkpoint` per run; the §6.3 rule is
the frozen :func:`roboforge.eval.tuning.choose_configuration` per family (tuning seed 0,
KI-062). No X2 held-out start is ever read here.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT, configs_dir
from roboforge.train.suite import SuitePlan, load_plan
from roboforge.x2 import states as x2s
from roboforge.x2.pilot import FAMILIES, pilot_runs

SCHEMA = "roboforge.x2b/1"
CONFIG = "configs/extensions/x2b.yaml"
RUNS = REPO_ROOT / "runs" / "x2"
TRIALS = (1, 2, 3, 4)


def x2b_config() -> dict[str, Any]:
    """Load and validate ``configs/extensions/x2b.yaml``."""
    import yaml

    cfg = yaml.safe_load((configs_dir() / "extensions" / "x2b.yaml").read_text())
    keys = {"schema", "decisions", "label", "plan", "pilot_prefix", "summary_dirs", "out_dir",
            "suite", "n_states", "limits", "disk_estimate_gib"}
    if set(cfg) != keys or cfg["schema"] != SCHEMA:
        raise ValueError(f"{CONFIG}: keys {sorted(cfg)} or schema")
    if cfg["suite"] != x2s.VALIDATION_SUITE or int(cfg["n_states"]) != 50:
        raise ValueError(f"{CONFIG}: X2-b selects on the 50 {x2s.VALIDATION_SUITE} starts only")
    if sorted(cfg["disk_estimate_gib"]["run"]) != sorted(FAMILIES):
        raise ValueError(f"{CONFIG}: disk estimates must cover {FAMILIES}")
    return cfg


def x2b_plan() -> SuitePlan:
    """The X2-b plan, resolved by the frozen loader and checked against the preregistration."""
    plan = load_plan(REPO_ROOT / x2b_config()["plan"])
    if plan.task != "can" or plan.tuning_seed != 0 or plan.replicate_seeds != (1, 2):
        raise ValueError("X2-b: task can, tuning seed 0 (KI-062), replicate seeds 1 and 2")
    if tuple(f.name for f in plan.families) != FAMILIES:
        raise ValueError(f"X2-b families {[f.name for f in plan.families]} != {FAMILIES}")
    if any(tuple(sorted(f.trials)) != TRIALS for f in plan.families):
        raise ValueError("X2-b: every family has trials 1-4 (D-055, D-061)")
    if plan.replicates_with_tuning:
        raise ValueError("X2-b trains replicates only after the configuration choice")
    return plan


@dataclass(frozen=True)
class PlannedRun:
    """One X2-b training run: where it lives and whether X2-a already trained it."""

    family: str
    trial: int
    seed: int
    run_id: str
    config_sha256: str
    pilot: bool

    @property
    def run_dir(self) -> Path:
        """``runs/x2/<run_id>``."""
        return RUNS / self.run_id


@functools.lru_cache(maxsize=1)
def _pilot_by_config() -> tuple[tuple[str, str], ...]:
    return tuple((r.config.sha256(), r.run_id) for r in pilot_runs())


def pilot_by_config() -> dict[str, str]:
    """Config SHA-256 -> X2-a pilot run ID (the four D-124 runs)."""
    return dict(_pilot_by_config())


def planned_run(plan: SuitePlan, family: str, trial: int, seed: int) -> PlannedRun:
    """Resolve one run; a seed-0 run with a pilot run's config is that pilot run (D-125)."""
    spec = plan.spec(family, trial, seed)
    sha = spec.config.sha256()
    pilot = pilot_by_config().get(sha) if seed == plan.tuning_seed else None
    return PlannedRun(family, trial, seed, pilot or spec.run_id, sha, pilot is not None)


def tuning_runs(plan: SuitePlan) -> list[PlannedRun]:
    """Every trial of every family at seed 0, in plan order (diffusion first)."""
    return [planned_run(plan, f.name, t, plan.tuning_seed) for f in plan.families
            for t in sorted(f.trials)]


def replicate_runs(plan: SuitePlan, family: str, trial: int) -> list[PlannedRun]:
    """Seeds 1 and 2 of ``family``'s chosen trial (always new X2-b runs)."""
    out = [planned_run(plan, family, trial, s) for s in plan.replicate_seeds]
    if any(r.pilot for r in out):
        raise ValueError("a replicate run cannot be a pilot run")
    return out


def check_pilot_mapping(plan: SuitePlan) -> list[dict[str, Any]]:
    """Each pilot run maps to exactly one planned seed-0 trial, and its training manifest (if
    written) records that config, task ``can`` and seed 0. Returns the mapping rows."""
    rows = []
    for r in tuning_runs(plan):
        if not r.pilot:
            continue
        m_path = r.run_dir / "manifest.json"
        if m_path.exists():
            m = json.loads(m_path.read_text())
            if (m["config_sha256"], m["config"]["task"], int(m["config"]["seed"])) != (
                    r.config_sha256, "can", 0):
                raise ValueError(f"{r.run_id}: manifest config differs from the planned trial")
        rows.append({"family": r.family, "trial": r.trial, "run_id": r.run_id,
                     "config_sha256": r.config_sha256})
    if sorted(x["run_id"] for x in rows) != sorted(pilot_by_config().values()):
        raise ValueError("every pilot run must map to exactly one planned trial")
    return rows


def load_summaries(dirs: list[Path] | None = None,
                   cache: dict[Path, dict[str, Any]] | None = None
                   ) -> dict[tuple[str, int], dict[str, Any]]:
    """Every X2 validation summary in ``dirs`` keyed by (train run, step); duplicates raise.
    ``cache`` (path -> parsed summary) avoids re-reading files, which are written once."""
    cfg = x2b_config()
    dirs = dirs if dirs is not None else [REPO_ROOT / d for d in cfg["summary_dirs"]]
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*.json")):
            if cache is not None and p in cache:
                s = cache[p]
            else:
                s = json.loads(p.read_text())
                if cache is not None:
                    cache[p] = s
            if s.get("suite") != cfg["suite"]:
                raise ValueError(f"{p.name}: suite {s.get('suite')!r} is not {cfg['suite']!r}")
            key = (s["train_run_id"], int(s["checkpoint"]["step"]))
            if key in out:
                raise ValueError(f"{key[0]} step {key[1]} is evaluated twice")
            out[key] = s
    return out


def read_train_manifest(run: PlannedRun) -> dict[str, Any] | None:
    """A run's final training manifest, or None while it has none."""
    p = run.run_dir / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


def selection(run: PlannedRun,
              summaries: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any] | None:
    """The frozen §7 selection of a complete run once all its checkpoints are evaluated."""
    from roboforge.eval.selection import select_checkpoint

    m = read_train_manifest(run)
    if m is None or m["status"] != "complete":
        return None
    evals = [summaries[(run.run_id, int(c["step"]))] for c in m["checkpoints"]
             if (run.run_id, int(c["step"])) in summaries]
    if len(evals) != len(m["checkpoints"]):
        return None
    return select_checkpoint(m, evals)


def family_choice(plan: SuitePlan, family: str,
                  summaries: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any] | None:
    """The frozen §6.3 choice once every trial has a final manifest and (if complete) a
    selection; None before."""
    from roboforge.eval.tuning import choose_configuration

    fam = next(f for f in plan.families if f.name == family)
    trials = []
    for t in sorted(fam.trials):
        r = planned_run(plan, family, t, plan.tuning_seed)
        m = read_train_manifest(r)
        if m is None or m["status"] not in ("complete", "failed"):
            return None
        sel = selection(r, summaries) if m["status"] == "complete" else None
        if m["status"] == "complete" and sel is None:
            return None
        trials.append({"trial": t, "overrides": list(fam.trials[t]), "manifest": m,
                       "selection": sel})
    rec = choose_configuration(family, plan.task, trials, list(fam.trials))
    rec["x2b_runs"] = [{"trial": t["trial"], "run_id": t["manifest"]["run_id"],
                        "pilot": planned_run(plan, family, t["trial"], 0).pilot}
                       for t in trials]
    return rec


def x2_test_configuration(plan: SuitePlan, choices: dict[str, dict[str, Any]],
                       summaries: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any] | None:
    """The X2 test configuration (for X2-c): per family the chosen trial and, for seeds 0-2,
    the §7-selected checkpoint. None until every family is chosen and every run selected."""
    out: dict[str, Any] = {"schema": "roboforge.x2_test_configuration/1",
                           "decisions": x2b_config()["decisions"], "task": plan.task,
                           "suite": x2b_config()["suite"],
                           "label": "selected on X2 validation starts only; X2-c tests it once",
                           "families": {}}
    for fam in FAMILIES:
        ch = choices.get(fam)
        if ch is None:
            return None
        trial = int(ch["chosen"]["trial"])
        runs = [planned_run(plan, fam, trial, plan.tuning_seed), *replicate_runs(plan, fam, trial)]
        rows = []
        for r in runs:
            sel = selection(r, summaries)
            if sel is None:
                return None
            c = sel["chosen"]
            rows.append({"seed": r.seed, "train_run_id": r.run_id, "pilot": r.pilot,
                         "selected_step": c["step"], "checkpoint_sha256": c["sha256"],
                         "validation_successes": c["successes"], "n": c["n_episodes"],
                         "first_success_step_mean": c["first_success_step_mean"],
                         "eval_run_id": c["eval_run_id"]})
        out["families"][fam] = {"chosen_trial": trial, "overrides": ch["chosen"]["overrides"],
                                "config_sha256": ch["chosen"]["config_sha256"], "runs": rows}
    return out


def remaining_disk_gib(n_runs_by_family: dict[str, int], n_evals: int) -> float:
    """Estimated disk (GiB) for ``n_runs_by_family`` new runs and ``n_evals`` evaluations."""
    est = x2b_config()["disk_estimate_gib"]
    return (sum(float(est["run"][f]) * int(n) for f, n in n_runs_by_family.items())
            + float(est["evaluation"]) * int(n_evals))
