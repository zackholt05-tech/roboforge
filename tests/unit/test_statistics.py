"""Hierarchical paired bootstrap on synthetic controls (RESEARCH_PROTOCOL §9, D-066)."""

from __future__ import annotations

import numpy as np
import pytest

from roboforge.eval.statistics import hierarchical_paired_bootstrap, separation, wilson_interval


def _boot(outcomes, comps, **kw):
    return hierarchical_paired_bootstrap(outcomes, comps, b=kw.pop("b", 2000), **kw)


def test_identical_families_are_not_separated() -> None:
    # Seeds are resampled independently per family (§9), so identical families with
    # seed-level variation get an interval that straddles 0, not a point.
    rng = np.random.default_rng(0)
    x = (rng.random((3, 50)) < 0.6).astype(float)
    r = _boot({"a": x, "b": x.copy()}, [("a", "b")])
    d = r["differences"][0]
    assert d["difference"] == 0.0 and d["ci95"][0] < 0 < d["ci95"][1]
    assert d["separation"] == "not separated"


def test_certain_difference_is_exact() -> None:
    r = _boot({"a": np.ones((3, 50)), "b": np.zeros((3, 50))}, [("a", "b"), ("b", "a")])
    assert r["differences"][0]["ci95"] == [1.0, 1.0]
    assert r["differences"][0]["separation"] == "separated (+)"
    assert r["differences"][1]["separation"] == "separated (-)"


def test_one_state_draw_is_shared_by_every_family() -> None:
    # Strong between-state variation, identical in both families and across seeds: each
    # family's interval is wide, the paired difference is exactly 0 on every replicate.
    rng = np.random.default_rng(1)
    a = np.repeat((rng.random((1, 50)) < 0.5).astype(float), 3, axis=0)
    r = _boot({"a": a, "b": a.copy()}, [("a", "b")])
    assert r["differences"][0]["ci95"] == [0.0, 0.0]
    for f in ("a", "b"):
        lo, hi = r["families"][f]["ci95"]
        assert hi - lo > 0.2


def test_seeds_are_resampled_per_family() -> None:
    # Family a's seeds differ completely (0, 0.5, 1); b's seeds are identical.
    a = np.stack([np.zeros(50), np.r_[np.ones(25), np.zeros(25)], np.ones(50)])
    b = np.full((3, 50), 0.5)
    r = _boot({"a": a, "b": b}, [])
    assert r["families"]["a"]["per_seed"] == [0.0, 0.5, 1.0]
    wa = np.diff(r["families"]["a"]["ci95"])[0]
    assert wa > 0.3  # seed-level variation dominates
    assert r["families"]["b"]["ci95"] == [0.5, 0.5]


def test_reproducible_with_the_fixed_seed_and_order_sensitive_draws() -> None:
    rng = np.random.default_rng(2)
    o = {"a": (rng.random((3, 50)) < 0.7).astype(float),
         "b": (rng.random((3, 50)) < 0.4).astype(float)}
    r1, r2 = _boot(o, [("a", "b")]), _boot(o, [("a", "b")])
    assert r1 == r2 and r1["seed"] == 20261106 and r1["draw_order"] == ["states", "a", "b"]
    r3 = _boot(o, [("a", "b")], seed=7)
    assert r3["differences"][0]["ci95"] != r1["differences"][0]["ci95"]


def test_single_seed_binomial_interval_is_plausible() -> None:
    x = np.r_[np.ones(25), np.zeros(25)][None]
    lo, hi = _boot({"a": x}, [], b=10_000)["families"]["a"]["ci95"]
    assert 0.34 <= lo <= 0.38 and 0.62 <= hi <= 0.66  # sqrt(.25/50) = 0.0707 -> +-0.139


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="same n_states"):
        _boot({"a": np.zeros((3, 50)), "b": np.zeros((3, 49))}, [])
    with pytest.raises(ValueError, match="finite"):
        _boot({"a": np.full((1, 5), np.nan)}, [])
    with pytest.raises(ValueError, match="every family"):
        _boot({"a": np.zeros((1, 5))}, [], order=["b"])


def test_separation_and_wilson() -> None:
    assert separation([-0.1, 0.2]) == "not separated"
    assert separation([0.0, 0.2]) == "not separated"
    lo, hi = wilson_interval(50, 50)
    assert round(lo, 3) == 0.929 and hi == 1.0


def test_default_groups_reproduce_the_ungrouped_draws() -> None:
    rng = np.random.default_rng(3)
    o = {"a": (rng.random((3, 50)) < 0.7).astype(float),
         "b": (rng.random((3, 50)) < 0.4).astype(float)}
    r1 = _boot(o, [("a", "b")])
    r2 = _boot(o, [("a", "b")], groups={"a": "a", "b": "b"})
    assert r1["differences"] == r2["differences"] and r1["families"] == r2["families"]
    assert "seed_groups" not in r1 and r2["seed_groups"] == {"a": "a", "b": "b"}


def test_a_shared_seed_group_keeps_the_drop_paired_by_seed() -> None:
    # One family on two sets with identical outcomes and strong seed variation: with a shared
    # seed draw the drop is exactly 0 on every replicate; with independent draws it is not.
    a = np.stack([np.zeros(50), np.r_[np.ones(25), np.zeros(25)], np.ones(50)])
    o = {"f:base": a, "f:ood": a.copy()}
    shared = _boot(o, [("f:base", "f:ood")], groups={"f:base": "f", "f:ood": "f"})
    indep = _boot(o, [("f:base", "f:ood")])
    assert shared["differences"][0]["ci95"] == [0.0, 0.0]
    assert indep["differences"][0]["ci95"][0] < 0 < indep["differences"][0]["ci95"][1]
    assert shared["draw_order"] == ["states", "f"]


def test_contrasts_match_differences_and_combine_drops() -> None:
    rng = np.random.default_rng(4)
    o = {k: (rng.random((3, 50)) < p).astype(float)
         for k, p in (("x:base", 0.9), ("x:ood", 0.5), ("y:base", 0.9), ("y:ood", 0.8))}
    g = {k: k.split(":")[0] for k in o}
    r = _boot(o, [("x:base", "x:ood")], groups=g,
              contrasts=[("drop x", {"x:base": 1, "x:ood": -1}),
                         ("drop x - drop y", {"x:base": 1, "x:ood": -1,
                                              "y:base": -1, "y:ood": 1})])
    assert r["contrasts"][0]["ci95"] == r["differences"][0]["ci95"]
    want = (o["x:base"].mean() - o["x:ood"].mean()) - (o["y:base"].mean() - o["y:ood"].mean())
    assert r["contrasts"][1]["estimate"] == pytest.approx(want)
    with pytest.raises(ValueError, match="same n_seeds"):
        _boot({"p": np.zeros((3, 5)), "q": np.zeros((2, 5))}, [], groups={"p": "g", "q": "g"})
