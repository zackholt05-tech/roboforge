"""X1 action clamp (DECISIONS D-107; KI-052): arithmetic, wrapper, IDs, and the analysis.

These tests need no simulator, no dataset and no ``runs/``. They read only committed files: the
normalizers, the task manifests, the test configuration records and the Session 7 robustness
summaries and analysis.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from roboforge.data.normalization import ActionCodec
from roboforge.eval.policies import PolicyInfo
from roboforge.eval.statistics import hierarchical_paired_bootstrap
from roboforge.eval.validation import eval_run_id
from roboforge.exploratory import clamp as xc
from roboforge.paths import REPO_ROOT
from roboforge.sim import robustness as rb

TASKS = ("lift", "can")
PLATFORM = "linux-aarch64-container-1f685638f28c"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _codec(task: str) -> ActionCodec:
    norm, _ = xc.committed_action_normalizer(task)
    return ActionCodec(norm, -np.ones(norm.dim), np.ones(norm.dim))


def _actions(seed: int = 0, n: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = rng.uniform(-3.0, 3.0, size=(n, 7))
    y[:50] = rng.uniform(-1.0, 1.0, size=(50, 7))
    y[50], y[51] = 1.0, -1.0  # the exact bound
    return y


@pytest.mark.parametrize("task", TASKS)
def test_bounds_are_the_training_range_and_bind_where_expected(task: str) -> None:
    b = xc.clamp_bounds(task)
    assert b["mode"] == "minmax" and b["normalized_bound"] == [-1.0, 1.0]
    assert b["max_abs_low_minus_train_min"] < 1e-15
    assert b["max_abs_high_minus_train_max"] < 1e-15
    expect = {"lift": ([1, 3, 4, 5], [1, 3, 4, 5]), "can": ([3], [3, 4, 5])}[task]
    assert (b["binds_low"], b["binds_high"]) == expect
    assert 6 not in b["binds_low"] + b["binds_high"]  # the gripper clamp is a no-op


@pytest.mark.parametrize("task", TASKS)
def test_disabled_codec_is_exactly_the_frozen_codec(task: str) -> None:
    base = _codec(task)
    off = xc.ClampedCodec(base, enabled=False)
    for y in _actions():
        e0, m0 = base.decode(y)
        e1, m1 = off.decode(y)
        assert np.array_equal(e0, e1) and np.array_equal(m0, m1)
    assert off.total["n_actions_clamped"] == 0


@pytest.mark.parametrize("task", TASKS)
def test_clamped_decode_lies_in_the_bound_and_changes_only_binding_entries(task: str) -> None:
    base = _codec(task)
    on = xc.ClampedCodec(base, enabled=True)
    b = xc.clamp_bounds(task)
    lo, hi = np.array(b["low"]), np.array(b["high"])
    free = [i for i in range(7) if i not in set(b["binds_low"]) | set(b["binds_high"])]
    n_out = 0
    for y in _actions(1):
        e0, m0 = base.decode(y)
        e1, m1 = on.decode(y)
        assert np.all(e1 >= lo) and np.all(e1 <= hi)
        outside = np.abs(y) > 1
        n_out += int(outside.any())
        assert np.array_equal(m1, m0 | outside)
        assert np.array_equal(e1[free], e0[free])  # identical where the clamp cannot bind
        if not outside.any():
            assert np.array_equal(e1, e0)  # bitwise unchanged inside the bound
    assert on.total["n_actions_clamped"] == n_out
    assert on.total["n_actions_decoded"] == 4000


@pytest.mark.parametrize("task", TASKS)
def test_diffusion_outputs_pass_the_clamp_unchanged(task: str) -> None:
    """The sampler returns float32 values already in [-1, 1]; cast to float64 they are fixed
    points of the clamp, so a diffusion action never changes (D-107 check 2, offline part)."""
    base = _codec(task)
    on = xc.ClampedCodec(base, enabled=True)
    rng = np.random.default_rng(2)
    y32 = np.clip(rng.normal(0, 0.8, size=(2000, 7)), -1, 1).astype(np.float32)
    y32[0], y32[1], y32[2] = np.float32(1.0), np.float32(-1.0), np.float32(0.0)
    for y in y32.astype(np.float64):
        e0, m0 = base.decode(y)
        e1, m1 = on.decode(y)
        assert np.array_equal(e0, e1) and np.array_equal(m0, m1)
    assert on.total["n_actions_clamped"] == 0
    assert on.total["entries_clamped_per_dim"] == [0] * 7


class _Inner:
    def __init__(self, task: str, stochastic: bool) -> None:
        self.task, self.stochastic = task, stochastic
        self.info = PolicyInfo(name="x", kind="learned", learned=True, autonomous=True,
                               action_space="normalized", label="inner", params={"a": 1})
        self.calls: list[str] = []

    def codec(self, spec):
        return _codec(self.task)

    def reset_episode(self, ep) -> None:
        self.calls.append("reset")

    def observe(self, obs) -> None:
        self.calls.append("observe")

    def act(self, obs):
        self.calls.append("act")
        return np.array([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    def episode_info(self):
        return {"noise_seed": 7} if self.stochastic else None


@pytest.mark.parametrize("stochastic", [False, True])
def test_wrapper_forwards_and_counts_per_episode(stochastic: bool) -> None:
    inner = _Inner("lift", stochastic)
    w = xc.ClampedPolicy(inner, clamp=True)
    assert w.stochastic is stochastic and w.info.action_space == "normalized"
    assert w.info.params["a"] == 1 and w.info.params["x1_clamp"]["enabled"] is True
    assert xc.LABEL in w.info.label and inner.info.params == {"a": 1}  # inner untouched
    codec = w.codec(None)
    for _ in range(2):
        w.reset_episode(SimpleNamespace())
        w.observe(None)
        for a in w.act(None):
            codec.decode(a)
        info = w.episode_info()
        assert info["x1_clamp"]["counts"]["n_actions_clamped"] == 1  # reset per episode
        assert info["x1_clamp"]["counts"]["entries_clamped_per_dim"][0] == 1
        assert info["x1_clamp"]["counts"]["max_excess_per_dim"][0] == 1.0
        assert ("noise_seed" in info) is stochastic  # run_states reads it for diffusion
    assert codec.total["n_actions_clamped"] == 2
    assert inner.calls == ["reset", "observe", "act"] * 2


def test_ids_are_new_and_directories_are_never_reused(tmp_path: Path) -> None:
    rid = xc.x1_eval_run_id("s6_bc_mlp_x", 5000, "ood_arm_offset", PLATFORM, clamp=True)
    frozen = eval_run_id("s6_bc_mlp_x", 5000, PLATFORM, stochastic=False,
                         suite="ood_arm_offset")
    assert rid == frozen + "__x1-clamp" and rid != frozen
    assert xc.x1_eval_run_id("r", 1, "robustness_base", PLATFORM, clamp=False).endswith(
        "__x1-noclamp")
    out = xc.prepare_out_dir(tmp_path, rid, clamp=True)
    assert out == tmp_path / "x1_clamp" / rid and out.is_dir()
    with pytest.raises(FileExistsError):
        xc.prepare_out_dir(tmp_path, rid, clamp=True)
    (tmp_path / "top").mkdir()
    with pytest.raises(FileExistsError):
        xc.prepare_out_dir(tmp_path, "top", clamp=True)


def test_the_orchestrator_plans_exactly_the_45_bc_evaluations() -> None:
    rx = _script("run_x1_clamp")
    jobs = rx.jobs()
    assert len(jobs) == 45
    assert {j["family"] for j in jobs} == {"bc_mlp", "bc_rnn", "bc_mlp_chunk"}
    for task, n_sets in (("lift", 3), ("can", 2)):
        tj = [j for j in jobs if j["task"] == task]
        assert len(tj) == 9 * n_sets
        assert {j["suite"] for j in tj} == set(rb.sets_for(task))
    rec = json.loads((REPO_ROOT / rx.CONFIGS["lift"][0]).read_text())
    sha = {s["train_run_id"]: s["checkpoint_sha256"] for f in rec["families"].values()
           for s in f["seeds"]}
    assert all(j["checkpoint_sha256"] == sha[j["train_run_id"]] for j in jobs
               if j["task"] == "lift")
    assert all(j["eval_run_id"].endswith("__x1-clamp") for j in jobs)


def test_the_analysis_refuses_wrong_hashes_and_a_second_run(tmp_path: Path) -> None:
    ax = _script("analyze_x1_clamp")
    with pytest.raises(SystemExit):
        ax.pinned(ax.SESSION7_ANALYSIS[0], "0" * 64)
    ax.refuse_existing(tmp_path)
    (tmp_path / "results.json").write_text("{}")
    with pytest.raises(SystemExit):
        ax.refuse_existing(tmp_path)


def test_grouped_extra_keys_leave_the_frozen_draws_unchanged() -> None:
    rng = np.random.default_rng(5)
    fams = ["f1", "f2", "f3"]
    base = {f"{f}:{s}": (rng.random((3, 50)) < 0.7).astype(float) for f in fams
            for s in ("a", "b")}
    comps = [("f1:a", "f1:b"), ("f3:b", "f2:b")]
    con = [("c", {"f1:a": 1, "f1:b": -1, "f2:a": -1, "f2:b": 1})]
    g = {k: k.split(":")[0] for k in base}
    ref = hierarchical_paired_bootstrap(base, comps, groups=g, contrasts=con, b=2000)
    ext = dict(base)
    ext.update({f"f1+clamp:{s}": (rng.random((3, 50)) < 0.5).astype(float) for s in ("a", "b")})
    g2 = {**g, "f1+clamp:a": "f1", "f1+clamp:b": "f1"}
    got = hierarchical_paired_bootstrap(ext, [*comps, ("f1+clamp:a", "f1:a")], groups=g2,
                                        contrasts=con, b=2000)
    for k in base:
        assert got["families"][k] == ref["families"][k]
    assert got["differences"][:2] == ref["differences"]
    assert got["contrasts"] == ref["contrasts"]


@pytest.mark.parametrize("task", TASKS)
def test_the_frozen_session7_bootstrap_reproduces_exactly_with_clamped_arms(task: str) -> None:
    """The analysis's core assumption, checked before any X1 data exist: the committed Session 7
    summaries plus three stand-in clamped arms reproduce every frozen number of analysis.json."""
    ax = _script("analyze_x1_clamp")
    a7 = json.loads(ax.pinned(*ax.SESSION7_ANALYSIS).read_text())
    fz = ax.frozen_outcomes(task, a7, 50)
    outcomes, groups = dict(fz["outcomes"]), dict(fz["groups"])
    rng = np.random.default_rng(13)
    for f in ax.BC:
        for s in fz["sets"]:
            outcomes[f"{f}{ax.CL}:{s}"] = (rng.random((3, 50)) < 0.6).astype(float)
            groups[f"{f}{ax.CL}:{s}"] = f
    comps, contrasts, n_c, n_k = ax.comparisons(fz["sets"])
    boot = hierarchical_paired_bootstrap(outcomes, comps, groups=groups, contrasts=contrasts)
    assert ax.frozen_mismatches(boot, fz["h7"]["bootstrap"], n_c, n_k) == []
    names = [c["name"] for c in boot["contrasts"][n_k:]]
    if task == "lift":
        assert ax.PRIMARY["name"] in names and ax.SUPPORT in names
        assert ax.PRIMARY["frozen_name"] in [c["name"] for c in boot["contrasts"][:n_k]]
    assert ax.BASE_LEG in [(d["a"], d["b"]) for d in boot["differences"][n_c:]]


def test_every_reading_covers_every_verdict() -> None:
    ax = _script("analyze_x1_clamp")
    for r in ax.READINGS.values():
        assert set(r) == {"separated (-)", "not separated", "separated (+)"}
