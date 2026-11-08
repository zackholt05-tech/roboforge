"""Suite plans (``roboforge.train.suite``) and the §6.3 configuration choice (``eval.tuning``)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from roboforge.eval.tuning import choose_configuration
from roboforge.paths import REPO_ROOT
from roboforge.train.config import load_config, run_id
from roboforge.train.suite import load_plan, plan_table

PLAN = REPO_ROOT / "configs" / "experiments" / "level1_core_lift.yaml"
PARAMS = {"bc_mlp": 1077255, "bc_rnn": 1080703, "bc_mlp_chunk": 1204336,
          "diffusion_unet": 1092551}  # RESEARCH_PROTOCOL §6.1


def test_committed_lift_plan_resolves_to_the_protocol_budget_and_capacities() -> None:
    plan = load_plan(PLAN)
    rows = plan_table(plan)
    assert len(rows) == 16 and {r["family"] for r in rows} == set(PARAMS)
    for r in rows:
        assert r["budget"] == {"max_steps": 100000, "batch_size": 100, "ckpt_every": 5000}
        assert r["param_count"] == PARAMS[r["family"]]
        assert r["seed"] == 0 and r["run_id"].startswith("s6_")
    assert plan.replicate_seeds == (1, 2) and plan.task == "lift"
    # Trial 1 is the pilot configuration: only the budget differs from the base config.
    for fam in plan.families:
        base = load_config(REPO_ROOT / fam.config_path).to_dict()
        t1 = plan.spec(fam.name, 1, 0).config.to_dict()
        base["train"].update(max_steps=100000, ckpt_every=5000)
        assert t1 == base


def test_planned_run_ids_equal_what_train_py_computes() -> None:
    plan = load_plan(PLAN)
    spec = plan.spec("bc_rnn", 3, 2)
    argv = spec.train_argv(plan.prefix, plan.budget)
    sets = [argv[i + 1] for i, a in enumerate(argv) if a == "--set"]
    cfg = load_config(REPO_ROOT / "configs/models/bc_rnn.yaml", [*sets, "task=lift", "seed=2"])
    assert run_id(cfg, "s6") == spec.run_id and cfg.optim.lr == 3e-4 and cfg.seed == 2


def _write(tmp_path: Path, **changes: object) -> Path:
    raw = yaml.safe_load(PLAN.read_text())
    raw.update(changes)
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


@pytest.mark.parametrize("override", ["train.max_steps=50000", "model.dropout=0.1",
                                      "horizons.obs=1", "seed=3", "train.batch_size=64"])
def test_trials_may_not_change_budget_architecture_or_seed(tmp_path: Path,
                                                           override: str) -> None:
    raw = yaml.safe_load(PLAN.read_text())
    raw["families"][1]["trials"][1]["overrides"] = [override]
    with pytest.raises(ValueError, match="not allowed"):
        load_plan(_write(tmp_path, families=raw["families"]))


def test_plan_rejects_bad_trial_sets_and_budgets(tmp_path: Path) -> None:
    raw = yaml.safe_load(PLAN.read_text())
    fams = raw["families"]
    five = [*fams[1]["trials"], {"trial": 5, "overrides": ["optim.lr=1.0e-3"]}]
    with pytest.raises(ValueError, match="n <= 4"):
        load_plan(_write(tmp_path, families=[{**fams[1], "trials": five}]))
    changed_t1 = [{"trial": 1, "overrides": ["optim.lr=1.0e-3"]}, *fams[1]["trials"][1:]]
    with pytest.raises(ValueError, match="trial 1"):
        load_plan(_write(tmp_path, families=[{**fams[1], "trials": changed_t1}]))
    dup = [*fams[1]["trials"][:2], {"trial": 3, "overrides": ["name=other",
                                                               "optim.schedule=cosine"]}]
    with pytest.raises(ValueError, match="duplicates"):
        load_plan(_write(tmp_path, families=[{**fams[1], "trials": dup}]))
    with pytest.raises(ValueError, match="exactly"):
        load_plan(_write(tmp_path, budget=["train.max_steps=100000"]))
    with pytest.raises(ValueError, match="replicate"):
        load_plan(_write(tmp_path, replicate_seeds=[0, 1]))


def _trial(n: int, successes: int, mean_fs: float | None, *, seed: int = 0,
           max_steps: int = 100000) -> dict:
    rid = f"s6_x{n}_lift_seed{seed}_0000000{n}"
    cfg = {"task": "lift", "seed": seed,
           "train": {"max_steps": max_steps, "batch_size": 100, "ckpt_every": 5000}}
    chosen = {"step": 5000 * n, "sha256": f"{n:064x}", "successes": successes,
              "n_episodes": 50, "first_success_step_mean": mean_fs,
              "eval_run_id": f"{rid}__val"}
    return {"trial": n, "overrides": [] if n == 1 else [f"optim.lr={n}e-4"],
            "manifest": {"run_id": rid, "status": "complete", "config": cfg,
                         "config_sha256": f"{n:064x}"},
            "selection": {"train_run_id": rid, "chosen": chosen}}


def test_choice_orders_by_success_then_first_success_step_then_trial() -> None:
    trials = [_trial(1, 49, 40.0), _trial(2, 50, 52.0), _trial(3, 50, 47.5),
              _trial(4, 50, 47.5)]
    rec = choose_configuration("bc_mlp", "lift", trials, [1, 2, 3, 4])
    assert rec["chosen"]["trial"] == 3 and rec["n_tied_on_success"] == 3
    assert [c["trial"] for c in rec["candidates"]] == [1, 2, 3, 4]
    # No success at all sorts last on the first-success tiebreak.
    rec = choose_configuration("bc_mlp", "lift", [_trial(1, 0, None), _trial(2, 0, None)],
                               [1, 2])
    assert rec["chosen"]["trial"] == 1


def test_choice_refuses_incomplete_or_mismatched_trials() -> None:
    with pytest.raises(ValueError, match="planned"):
        choose_configuration("bc_mlp", "lift", [_trial(1, 50, 40.0)], [1, 2])
    with pytest.raises(ValueError, match="seed"):
        choose_configuration("bc_mlp", "lift", [_trial(1, 50, 40.0), _trial(2, 50, 41.0,
                                                                            seed=1)], [1, 2])
    with pytest.raises(ValueError, match="budget"):
        choose_configuration("bc_mlp", "lift", [_trial(1, 50, 40.0),
                                                _trial(2, 50, 41.0, max_steps=30000)], [1, 2])
    bad = _trial(2, 50, 41.0)
    bad["selection"]["train_run_id"] = "other"
    with pytest.raises(ValueError, match="selection belongs"):
        choose_configuration("bc_mlp", "lift", [_trial(1, 50, 40.0), bad], [1, 2])


def test_failed_trial_counts_ranks_last_and_cannot_be_chosen() -> None:
    failed = _trial(1, 50, 30.0)
    failed["manifest"]["status"], failed["selection"] = "failed", None
    rec = choose_configuration("bc_mlp", "lift", [failed, _trial(2, 3, 90.0)], [1, 2])
    assert rec["chosen"]["trial"] == 2 and rec["n_failed"] == 1
    assert rec["candidates"][0]["status"] == "failed"
    other = _trial(2, 0, None)
    other["manifest"]["status"], other["selection"] = "failed", None
    with pytest.raises(ValueError, match="every trial failed"):
        choose_configuration("bc_mlp", "lift", [failed, other], [1, 2])


def test_replicates_with_tuning_needs_a_single_trial(tmp_path: Path) -> None:
    raw = yaml.safe_load(PLAN.read_text())
    raw["replicates_with_tuning"] = True
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="exactly one trial"):
        load_plan(p)
    raw["families"] = [{**raw["families"][0], "trials": [{"trial": 1, "overrides": []}]}]
    p.write_text(yaml.safe_dump(raw))
    plan = load_plan(p)
    assert plan.replicates_with_tuning
    assert [s.seed for s in plan.replicate_runs(plan.families[0].name, 1)] == [1, 2]
    raw["replicates_with_tuning"] = "yes"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="true or false"):
        load_plan(p)
    assert not load_plan(PLAN).replicates_with_tuning
