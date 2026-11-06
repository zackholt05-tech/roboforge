"""The checkpoint-selection rule of RESEARCH_PROTOCOL §7, identical for every family.

Inputs are the canonical-platform validation summaries (:mod:`roboforge.eval.validation`) of
**every** checkpoint of one training run, each on all 50 validation starts. The rule:

1. highest validation success count;
2. on a tie, lower mean first-success step over the successful episodes (a checkpoint with
   no success has none, which sorts last);
3. on a remaining tie, the earlier checkpoint.

Offline losses never enter. Missing, partial, duplicate or non-canonical evaluations raise
instead of being skipped, because selecting from an incomplete set would silently change the
rule.
"""

from __future__ import annotations

import math
from typing import Any

RULE = ("highest validation success count over all 50 validation starts (canonical platform); "
        "ties -> lower mean first-success step over successful episodes; remaining ties -> "
        "earlier checkpoint (RESEARCH_PROTOCOL §7)")


def selection_key(r: dict[str, Any]) -> tuple[int, float, int]:
    """Sort key: smaller is better."""
    mean_fs = r["first_success_step_mean"]
    return (-int(r["successes"]), math.inf if mean_fs is None else float(mean_fs), int(r["step"]))


def select_checkpoint(train_manifest: dict[str, Any],
                      evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the rule to a training run's evaluated checkpoints; return the selection record."""
    expected = {c["step"]: c["sha256"] for c in train_manifest["checkpoints"]}
    rows: dict[int, dict[str, Any]] = {}
    for e in evaluations:
        if e["train_run_id"] != train_manifest["run_id"]:
            raise ValueError(f"{e['eval_run_id']} belongs to another training run")
        if not e["eligible_for_selection"] or e["n_episodes"] != 50 or e["partial"]:
            raise ValueError(f"{e['eval_run_id']} is not a complete canonical evaluation")
        step = int(e["checkpoint"]["step"])
        if e["checkpoint"]["sha256"] != expected.get(step):
            raise ValueError(f"{e['eval_run_id']}: checkpoint hash differs from the manifest")
        if step in rows:
            raise ValueError(f"two evaluations of step {step}; resolve before selecting")
        rows[step] = {"step": step, "sha256": expected[step], "eval_run_id": e["eval_run_id"],
                      "successes": int(e["successes"]), "n_episodes": int(e["n_episodes"]),
                      "first_success_step_mean": e["first_success_step"]["mean"],
                      "wilson95": e["wilson95"], "platform_id": e["platform"]["platform_id"]}
    missing = sorted(set(expected) - set(rows))
    if missing:
        raise ValueError(f"checkpoints without a canonical validation evaluation: {missing}")
    if len({r["platform_id"] for r in rows.values()}) != 1:
        raise ValueError("evaluations come from more than one platform")
    ordered = [rows[s] for s in sorted(rows)]
    chosen = min(ordered, key=selection_key)
    return {"schema": "roboforge.checkpoint_selection/1", "train_run_id": train_manifest["run_id"],
            "rule": RULE, "candidates": ordered, "chosen": chosen,
            "n_tied_on_success": sum(r["successes"] == chosen["successes"] for r in ordered)}
