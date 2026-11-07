"""``scripts/plot_validation.py`` table/selection helpers and the montage rule of
``scripts/render_episodes.py``, on a synthetic suite summary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from roboforge.paths import REPO_ROOT


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pv = _load("plot_validation")
re_ = _load("render_episodes")


def _run(trial: int, seed: int, successes: int | None, step: int = 5000) -> dict:
    rid = f"s6_x_t{trial}_lift_seed{seed}_0000000{trial}"
    sel = None if successes is None else {
        "step": step, "sha256": f"{trial}{seed}".ljust(64, "0"), "successes": successes,
        "n_episodes": 50, "first_success_step_mean": 50.0 + trial,
        "eval_run_id": f"{rid}__val_step{step:07d}__plat"}
    return {"run_id": rid, "trial": trial, "seed": seed,
            "role": "tuning" if seed == 0 else "replicate",
            "status": "complete" if sel else "failed", "selection": sel,
            "evals": {} if sel is None else {step: {"wilson95": [0.9, 1.0]}}}


def _summary() -> dict:
    runs = [_run(1, 0, 48), _run(2, 0, 50), _run(3, 0, None), _run(4, 0, 49),
            _run(2, 2, 47), _run(2, 1, 50)]
    return {"task": "lift", "families": [{"family": "bc_mlp", "chosen_trial": 2,
                                          "runs": runs}]}


def test_chosen_runs_are_the_chosen_trial_by_seed() -> None:
    fam = _summary()["families"][0]
    assert [(r["trial"], r["seed"]) for r in pv.chosen_runs(fam)] == [(2, 0), (2, 1), (2, 2)]


def test_table_lists_every_run_including_failures_and_the_seed_mean() -> None:
    text = pv.table(_summary(), {"bc_mlp": {"motion_clip_fraction": 0.01,
                                            "gripper_clip_fraction": 0.4}},
                    {"bc_mlp": {"median_ms": 0.9}}, "x.json")
    assert "VALIDATION ONLY" in text and "run failed" in text
    assert "| bc_mlp | 2 | 49.00 | 50, 50, 47 | 1.00 % | 40.0 % | 0.90 |" in text
    assert text.count("(chosen)") == 3


def test_montage_rule_takes_the_first_success_and_first_failure_by_state(tmp_path: Path) -> None:
    summ = _summary()
    (tmp_path / "validation").mkdir()
    eps = {0: [("lift_validation_0003", True), ("lift_validation_0001", True)],
           1: [("lift_validation_0002", True), ("lift_validation_0004", False),
               ("lift_validation_0000", False)],
           2: [("lift_validation_0001", True)]}
    for r in pv.chosen_runs(summ["families"][0]):
        ep = [{"state_id": s, "success": ok, "steps": 40 if ok else 400,
               "stop_reason": "x", "physics_sha256": s} for s, ok in eps[r["seed"]]]
        (tmp_path / "validation" / f"{r['selection']['eval_run_id']}.json").write_text(
            json.dumps({"episodes": ep}))
    picks = re_.pick(summ["families"][0], tmp_path)
    assert [(p["kind"], p["seed"], p["state_id"]) for p in picks] == [
        ("success", 0, "lift_validation_0001"), ("failure", 1, "lift_validation_0000")]
