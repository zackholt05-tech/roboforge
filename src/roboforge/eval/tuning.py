"""The configuration-choice rule of RESEARCH_PROTOCOL §6.3, identical for every family.

A family's tuning trials are trained with seed 0 at the main budget. Each trial's
checkpoint is first selected by the §7 rule (:mod:`roboforge.eval.selection`). The family's
configuration is then the trial whose **selected** checkpoint has:

1. the highest validation success count;
2. on a tie, the lower mean first-success step over its successful episodes (none sorts
   last);
3. on a remaining tie, the lower trial number.

A trial whose training run failed (manifest ``status: failed``, e.g. a non-finite loss) has
no selected checkpoint. It still counts against the family's 4-trial budget, is never
retried, and ranks below every completed trial.

The chosen configuration is then trained with the replicate seeds. Offline losses never
enter. Every trial of the family must be present and must use the same task, seed and budget;
otherwise the rule raises instead of choosing from an incomplete or mismatched set.
"""

from __future__ import annotations

import math
from typing import Any

RULE = ("among a family's seed-0 tuning trials at the main budget, the trial whose "
        "§7-selected checkpoint has the highest validation success count; ties -> lower mean "
        "first-success step; remaining ties -> lower trial number (RESEARCH_PROTOCOL §6.3)")
SCHEMA = "roboforge.tuning_choice/1"


def choice_key(c: dict[str, Any]) -> tuple[float, float, int]:
    """Sort key of one trial (smaller is better); a failed trial sorts last."""
    if c["status"] != "complete":
        return (math.inf, math.inf, int(c["trial"]))
    mean_fs = c["first_success_step_mean"]
    return (-int(c["successes"]), math.inf if mean_fs is None else float(mean_fs),
            int(c["trial"]))


def choose_configuration(family: str, task: str, trials: list[dict[str, Any]],
                         expected_trials: list[int]) -> dict[str, Any]:
    """Apply the §6.3 rule to a family's trials and return the choice record.

    Each entry of ``trials`` holds ``trial`` (int), ``overrides`` (list of str), the training
    manifest ``manifest`` and the §7 selection record ``selection`` of that run (``None``
    for a failed run).
    """
    by_trial: dict[int, dict[str, Any]] = {}
    budgets = set()
    for t in trials:
        n = int(t["trial"])
        if n in by_trial:
            raise ValueError(f"{family}: trial {n} listed twice")
        m, sel = t["manifest"], t["selection"]
        cfg = m["config"]
        if m.get("status") not in ("complete", "failed"):
            raise ValueError(f"{family} trial {n}: run {m['run_id']} has no final status")
        if cfg["task"] != task or int(cfg["seed"]) != 0:
            raise ValueError(f"{family} trial {n}: task/seed {cfg['task']}/{cfg['seed']} "
                             f"!= {task}/0")
        tr = cfg["train"]
        budgets.add((tr["max_steps"], tr["batch_size"], tr["ckpt_every"]))
        if m["status"] == "failed":
            if sel is not None:
                raise ValueError(f"{family} trial {n}: a failed run cannot have a selection")
            by_trial[n] = {"trial": n, "overrides": list(t["overrides"]), "status": "failed",
                           "train_run_id": m["run_id"], "config_sha256": m["config_sha256"],
                           "successes": None, "first_success_step_mean": None}
            continue
        if sel is None or sel["train_run_id"] != m["run_id"]:
            raise ValueError(f"{family} trial {n}: selection belongs to another run")
        chosen = sel["chosen"]
        by_trial[n] = {"trial": n, "overrides": list(t["overrides"]), "status": "complete",
                       "train_run_id": m["run_id"], "config_sha256": m["config_sha256"],
                       "selected_step": int(chosen["step"]),
                       "selected_sha256": chosen["sha256"],
                       "successes": int(chosen["successes"]),
                       "n_episodes": int(chosen["n_episodes"]),
                       "first_success_step_mean": chosen["first_success_step_mean"],
                       "selection_eval_run_id": chosen["eval_run_id"]}
    if sorted(by_trial) != sorted(expected_trials):
        raise ValueError(f"{family}: trials {sorted(by_trial)} != planned "
                         f"{sorted(expected_trials)}")
    if len(budgets) != 1:
        raise ValueError(f"{family}: trials differ in budget {sorted(budgets)}")
    ordered = [by_trial[n] for n in sorted(by_trial)]
    best = min(ordered, key=choice_key)
    if best["status"] != "complete":
        raise ValueError(f"{family}: every trial failed; no configuration can be chosen")
    return {"schema": SCHEMA, "family": family, "task": task, "rule": RULE,
            "budget": dict(zip(("max_steps", "batch_size", "ckpt_every"), budgets.pop(),
                               strict=True)),
            "candidates": ordered, "chosen": best,
            "n_failed": sum(c["status"] == "failed" for c in ordered),
            "n_tied_on_success": sum(c["successes"] == best["successes"] for c in ordered)}


__all__ = ["RULE", "SCHEMA", "choice_key", "choose_configuration"]
