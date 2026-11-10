"""X3 trains the frozen chosen Can configurations with new seeds only (DECISIONS D-111).

The X3 plan's configs, resolved with the frozen seeds 0-2, must give exactly the frozen
configuration hashes and run IDs of ``docs/FINAL_TEST_FREEZE.md``. The new runs (seeds 3-5)
must then differ from them in the seed alone. No ``runs/`` directory is needed.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from roboforge.eval.freeze import FAMILY_ORDER, load_freeze
from roboforge.paths import REPO_ROOT
from roboforge.train.config import run_id
from roboforge.train.suite import load_plan

PLAN = REPO_ROOT / "configs" / "experiments" / "x3_can.yaml"
NEW_SEEDS = (3, 4, 5)
# The 12 X3 training runs, resolved by scripts/run_suite.py --dry-run before any of them
# trained and listed in docs/extensions/X3_PREREGISTRATION.md.
RUN_IDS = {
    "diffusion_unet": ("s14_diffusion_unet_cos_can_seed3_a6116493",
                       "s14_diffusion_unet_cos_can_seed4_de04fdae",
                       "s14_diffusion_unet_cos_can_seed5_9119b2ec"),
    "bc_mlp": ("s14_bc_mlp_cos_can_seed3_04223e1b", "s14_bc_mlp_cos_can_seed4_7c8e5c7b",
               "s14_bc_mlp_cos_can_seed5_c001f43c"),
    "bc_mlp_chunk": ("s14_bc_mlp_chunk_wd1e1_can_seed3_2fec9f65",
                     "s14_bc_mlp_chunk_wd1e1_can_seed4_26a20b25",
                     "s14_bc_mlp_chunk_wd1e1_can_seed5_4b41dda8"),
    "bc_rnn": ("s14_bc_rnn_cos_can_seed3_24bd8478", "s14_bc_rnn_cos_can_seed4_bc40d865",
               "s14_bc_rnn_cos_can_seed5_ad65ce36"),
}


@pytest.fixture(scope="module")
def plan():
    return load_plan(PLAN)


@pytest.fixture(scope="module")
def frozen():
    return load_freeze(require_committed=False)["tasks"]["can"]["families"]


def test_plan_trains_new_seeds_only(plan) -> None:
    assert plan.task == "can" and plan.prefix == "s14"
    assert (plan.tuning_seed, *plan.replicate_seeds) == NEW_SEEDS
    assert plan.replicates_with_tuning
    assert sorted(f.name for f in plan.families) == sorted(FAMILY_ORDER)
    assert all(f.trials == {1: ()} for f in plan.families)
    assert plan.budget == ("train.max_steps=100000", "train.ckpt_every=5000")


@pytest.mark.parametrize("family", FAMILY_ORDER)
def test_frozen_seeds_resolve_to_the_frozen_configurations(plan, frozen, family: str) -> None:
    for run in frozen[family]["runs"]:
        cfg = plan.spec(family, 1, run["seed"]).config
        assert cfg.sha256() == run["config_sha256"]
        assert run_id(cfg, "s7") == run["train_run_id"]


@pytest.mark.parametrize("family", FAMILY_ORDER)
def test_new_runs_differ_from_the_frozen_ones_in_the_seed_only(plan, frozen, family: str) -> None:
    seed0 = next(r for r in frozen[family]["runs"] if r["seed"] == 0)
    for seed, rid in zip(NEW_SEEDS, RUN_IDS[family], strict=True):
        spec = plan.spec(family, 1, seed)
        assert spec.seed == seed and spec.config.seed == seed
        assert replace(spec.config, seed=0).sha256() == seed0["config_sha256"]
        assert spec.run_id == rid


def test_new_run_ids_are_distinct_and_new(plan) -> None:
    ids = [s.run_id for s in plan.tuning_runs()]
    ids += [s.run_id for f in plan.families for s in plan.replicate_runs(f.name, 1)]
    assert len(ids) == len(set(ids)) == 12
    assert all(i.startswith("s14_") for i in ids)
