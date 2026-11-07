"""Session 7 robustness starts (RESEARCH_PROTOCOL §14.1-14.2, D-063): perturbation rules,
determinism, the reserved seed block and an end-to-end generation of one state per task."""

from __future__ import annotations

import numpy as np
import pytest

from roboforge.sim import eval_states as es
from roboforge.sim import robustness as rb
from roboforge.sim.factory import load_sim_config, make_env


@pytest.fixture(scope="module")
def lift_env():
    env = make_env(load_sim_config("lift"))
    yield env
    env.close()


def _qadr(env, name: str) -> int:
    return 1 + int(env.sim.model._model.jnt_qposadr[rb._jnt(env.sim.model._model, name)])


def test_config_and_sets() -> None:
    cfg = rb.robustness_config()
    assert cfg["base"]["seed_base"] == 300000 and cfg["base"]["n_states"] == 50
    assert rb.sets_for("lift") == ["robustness_base", "ood_cube_shift", "ood_arm_offset"]
    assert rb.sets_for("can") == ["robustness_base", "ood_arm_offset"]
    lo, hi = cfg["base"]["reserved_seeds"]
    for other in es.suites_config()["suites"].values():  # disjoint from the evaluation blocks
        assert hi <= other["reserved_seeds"][0] or lo >= other["reserved_seeds"][1]


@pytest.mark.parametrize(("seed", "partition"), [(300000, "eval_robustness"),
                                                 (399999, "eval_robustness"),
                                                 (400000, "eval_adhoc")])
def test_reserved_seed_block(seed: int, partition: str) -> None:
    assert es.check_seed_allowed("lift", seed) == partition


def test_draws_are_deterministic_and_distinct() -> None:
    a, b = rb._draw(2, 7, 0).random(4), rb._draw(2, 7, 0).random(4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, rb._draw(2, 7, 1).random(4))
    assert not np.array_equal(a, rb._draw(1, 7, 0).random(4))


def test_arm_offset_changes_only_the_arm_joints(lift_env) -> None:
    spec = rb.robustness_config()["ood"]["ood_arm_offset"]
    base = np.array(lift_env.sim.get_state().flatten())
    out, params = rb.perturb_arm_offset(lift_env.sim.model._model, base, rb._draw(2, 0, 0), spec)
    idx = [_qadr(lift_env, j) for j in spec["joints"]]
    assert np.allclose(np.abs(out[idx] - base[idx]), 0.10, atol=1e-15)
    assert np.array_equal(np.sign(out[idx] - base[idx]).astype(int), params["signs"])
    rest = np.setdiff1d(np.arange(len(base)), idx)
    assert np.array_equal(out[rest], base[rest])


def test_cube_shift_leaves_the_training_square(lift_env) -> None:
    spec = rb.robustness_config()["ood"]["ood_cube_shift"]
    model = lift_env.sim.model._model
    base = np.array(lift_env.sim.get_state().flatten())
    adr = _qadr(lift_env, spec["object_joint"])
    base[adr:adr + 2] = (0.02, -0.025)
    for i in range(50):
        out, p = rb.perturb_cube_shift(model, base, rb._draw(1, i, 0), spec)
        r = float(np.hypot(*out[adr:adr + 2]))
        assert 0.05 <= r <= 0.07 and np.isclose(r, p["radius_m"])
        assert np.abs(out[adr:adr + 2]).max() > 0.03  # outside centre +- 0.03 m
        assert np.array_equal(out[adr + 2:adr + 7], base[adr + 2:adr + 7])  # z and quaternion
    bad = base.copy()
    bad[adr] = 0.05
    with pytest.raises(AssertionError, match="outside the training placement range"):
        rb.perturb_cube_shift(model, bad, rb._draw(1, 0, 0), spec)


@pytest.mark.parametrize("task", ["lift", "can"])
def test_one_state_per_set_generates_and_validates(task: str) -> None:
    env = make_env(load_sim_config(task))
    try:
        cfg = rb.robustness_config()
        b = cfg["base"]
        from roboforge.sim.snapshot import generate
        base = [generate(env, task, int(b["seed_base"]), f"{task}_robustness_base_0000",
                         cfg["partition"], {"suite": b["name"], "index": 0})]
        for name in rb.sets_for(task)[1:]:
            art = rb.generate_ood(task, name, base, env, "0" * 64)
            (entry,) = art.manifest["states"]
            assert entry["base_state_id"] == base[0].state_id
            assert entry["model_xml_sha256"] == base[0].model_xml_sha256
            assert entry["state_sha256"] != base[0].state_sha256
            assert art.states[0].partition == "eval_robustness"
            ref = rb.base_reference(env, task, base[0])
            assert rb.check_validity(env, task, base[0], ref, art.states[0])["passed"]
    finally:
        env.close()
