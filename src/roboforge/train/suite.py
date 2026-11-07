"""Main-suite plan: families, predeclared tuning trials, seeds and the shared budget.

A suite plan (``configs/experiments/<name>.yaml``, schema ``roboforge.suite_plan/1``) fixes,
before any run of the suite, everything RESEARCH_PROTOCOL §6.3 requires to be declared:

* the task, the run-ID prefix and the training budget, given once as ``--set`` overrides and
  applied identically to every run of every family;
* per family: the base config and its tuning trials. Trial 1 is the base config unchanged
  (the pilot configuration at the main budget). Trials 2..n change only allow-listed
  optimizer keys, so architecture, horizons, parameter count, batch and budget stay fixed;
* the tuning seed (0) and the replicate seeds that train the chosen configuration.

:func:`load_plan` validates the plan and resolves every run's full config and run ID exactly
as ``scripts/train.py`` does, so the trial -> run-ID mapping is known (and recorded) before
anything trains.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from roboforge.models.registry import build_model, count_parameters
from roboforge.paths import REPO_ROOT
from roboforge.train.config import RunConfig, load_config, run_id

SCHEMA = "roboforge.suite_plan/1"
MAX_TRIALS = 4  # RESEARCH_PROTOCOL §6.3: at most 4 configurations per family per task
# Keys a tuning trial may change. Everything else (model, horizons, train budget, loss,
# seed, task) is identical to trial 1, so a trial never changes capacity or the budget.
TRIAL_KEYS = ("name", "optim.lr", "optim.schedule", "optim.weight_decay")
BUDGET_KEYS = ("train.max_steps", "train.ckpt_every")
OBS_DIM, ACTION_DIM = 19, 7  # RESEARCH_PROTOCOL §3 (both tasks)


@dataclass(frozen=True)
class RunSpec:
    """One planned training run: family, trial, seed, overrides and the resolved config."""

    family: str
    trial: int
    seed: int
    config_path: str
    overrides: tuple[str, ...]
    config: RunConfig
    run_id: str

    def train_argv(self, prefix: str, budget: tuple[str, ...]) -> list[str]:
        """Arguments of ``scripts/train.py`` that reproduce exactly this run."""
        argv = ["scripts/train.py", "--config", self.config_path, "--task", self.config.task,
                "--seed", str(self.seed), "--prefix", prefix]
        for o in (*budget, *self.overrides):
            argv += ["--set", o]
        return argv


@dataclass(frozen=True)
class Family:
    """A family's base config and its predeclared trials (trial number -> overrides)."""

    name: str
    config_path: str
    trials: dict[int, tuple[str, ...]]


@dataclass(frozen=True)
class SuitePlan:
    """A validated suite plan."""

    path: str
    task: str
    prefix: str
    budget: tuple[str, ...]
    tuning_seed: int
    replicate_seeds: tuple[int, ...]
    families: tuple[Family, ...]

    def spec(self, family: str, trial: int, seed: int) -> RunSpec:
        """Resolve the run of ``family``'s ``trial`` with ``seed``."""
        fam = next(f for f in self.families if f.name == family)
        ov = fam.trials[trial]
        cfg = load_config(REPO_ROOT / fam.config_path,
                          [*self.budget, *ov, f"task={self.task}", f"seed={seed}"])
        return RunSpec(family, trial, seed, fam.config_path, ov, cfg, run_id(cfg, self.prefix))

    def tuning_runs(self) -> list[RunSpec]:
        """Every trial of every family with the tuning seed, in plan (launch) order."""
        return [self.spec(f.name, t, self.tuning_seed) for f in self.families
                for t in sorted(f.trials)]

    def replicate_runs(self, family: str, trial: int) -> list[RunSpec]:
        """The replicate-seed runs of ``family``'s chosen ``trial``."""
        return [self.spec(family, trial, s) for s in self.replicate_seeds]


def _check_override(o: str, allowed: tuple[str, ...], where: str) -> None:
    if "=" not in o or o.split("=", 1)[0] not in allowed:
        raise ValueError(f"{where}: override {o!r} not allowed (allowed keys: {allowed})")


def _without_name_and_seed(cfg: RunConfig) -> dict[str, Any]:
    d = cfg.to_dict()
    d.pop("name")
    d.pop("seed")
    return d


def load_plan(path: Path) -> SuitePlan:
    """Read and validate a suite plan; every run's config is resolved and checked."""
    raw = yaml.safe_load(Path(path).read_text())
    keys = {"schema", "task", "prefix", "budget", "tuning_seed", "replicate_seeds", "families"}
    if set(raw) != keys:
        raise ValueError(f"suite plan keys {sorted(raw)} != {sorted(keys)}")
    if raw["schema"] != SCHEMA:
        raise ValueError(f"schema {raw['schema']!r} != {SCHEMA!r}")
    budget = tuple(raw["budget"])
    for o in budget:
        _check_override(o, BUDGET_KEYS, "budget")
    if sorted(o.split("=", 1)[0] for o in budget) != sorted(BUDGET_KEYS):
        raise ValueError(f"budget must set exactly {BUDGET_KEYS}")
    seeds = tuple(int(s) for s in raw["replicate_seeds"])
    if int(raw["tuning_seed"]) in seeds or len(set(seeds)) != len(seeds):
        raise ValueError("replicate seeds must be distinct and differ from the tuning seed")
    fams = []
    for f in raw["families"]:
        if set(f) != {"family", "config", "trials"}:
            raise ValueError(f"family entry keys {sorted(f)}")
        trials = {int(t["trial"]): tuple(t["overrides"]) for t in f["trials"]}
        if sorted(trials) != list(range(1, len(trials) + 1)) or len(trials) > MAX_TRIALS:
            raise ValueError(f"{f['family']}: trials must be 1..n with n <= {MAX_TRIALS}")
        if trials[1]:
            raise ValueError(f"{f['family']}: trial 1 is the base (pilot) config, unchanged")
        for n, ov in trials.items():
            for o in ov:
                _check_override(o, TRIAL_KEYS, f"{f['family']} trial {n}")
        fams.append(Family(str(f["family"]), str(f["config"]), trials))
    if len({f.name for f in fams}) != len(fams):
        raise ValueError("duplicate family")
    plan = SuitePlan(str(path), str(raw["task"]), str(raw["prefix"]), budget,
                     int(raw["tuning_seed"]), seeds, tuple(fams))
    _validate_resolved(plan)
    return plan


def _validate_resolved(plan: SuitePlan) -> None:
    """Identical budget everywhere; per family identical architecture; distinct trials."""
    budgets, ids = set(), set()
    for fam in plan.families:
        seen, archs = set(), set()
        for n in sorted(fam.trials):
            cfg = plan.spec(fam.name, n, plan.tuning_seed).config
            t = cfg.train
            budgets.add((t.max_steps, t.batch_size, t.ckpt_every, t.val_every))
            archs.add(repr((cfg.family, cfg.model, vars(cfg.horizons), cfg.loss)))
            key = repr(_without_name_and_seed(cfg))
            if key in seen:
                raise ValueError(f"{fam.name} trial {n} duplicates an earlier trial")
            seen.add(key)
        if len(archs) != 1:
            raise ValueError(f"{fam.name}: trials differ in architecture or horizons")
        for spec in plan.tuning_runs():
            ids.add(spec.run_id)
    if len(budgets) != 1:
        raise ValueError(f"budget differs between runs: {sorted(budgets)}")
    if len(ids) != len(plan.tuning_runs()):
        raise ValueError("two planned runs share a run ID")


def plan_table(plan: SuitePlan) -> list[dict[str, Any]]:
    """Trial -> overrides, config hash, parameter count and run ID of every tuning run."""
    rows = []
    for spec in plan.tuning_runs():
        c = spec.config
        net = build_model(c.family, c.model, OBS_DIM, ACTION_DIM, c.horizons.obs,
                          c.horizons.pred)
        rows.append({"family": spec.family, "trial": spec.trial, "seed": spec.seed,
                     "overrides": list(spec.overrides), "config_sha256": c.sha256(),
                     "run_id": spec.run_id, "param_count": count_parameters(net),
                     "optim": {"lr": c.optim.lr, "schedule": c.optim.schedule,
                               "weight_decay": c.optim.weight_decay},
                     "budget": {"max_steps": c.train.max_steps,
                                "batch_size": c.train.batch_size,
                                "ckpt_every": c.train.ckpt_every}})
    return rows


__all__ = ["BUDGET_KEYS", "MAX_TRIALS", "SCHEMA", "TRIAL_KEYS", "Family", "RunSpec",
           "SuitePlan", "load_plan", "plan_table"]
