"""``scripts/smoke_sim.py`` never builds a held-out start (KNOWN_ISSUES KI-040).

Since the Level 1 freeze was committed the real final-test lock is open, and the smoke test's
seed-lock check called ``env.reset(900000)``, which then generated Lift held-out start 0. The
check now points the lock at a missing freeze and resets only after seeing the lock shut.
"""

from __future__ import annotations

import importlib.util

import pytest

from roboforge.paths import REPO_ROOT
from roboforge.sim import env_adapter
from roboforge.sim import eval_states as es

pytestmark = pytest.mark.sim

_spec = importlib.util.spec_from_file_location("smoke_sim", REPO_ROOT / "scripts" / "smoke_sim.py")
smoke_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke_sim)


def test_adapter_smoke_never_generates_a_held_out_start(monkeypatch) -> None:
    generated: list[int] = []
    real_generate = env_adapter.generate

    def guarded(env, task, seed, state_id, partition):  # noqa: ANN001, ANN202
        generated.append(int(seed))
        lo, hi = es.suites_config()["suites"][es.LOCKED_SUITE]["reserved_seeds"]
        if lo <= int(seed) < hi:
            raise AssertionError(f"held-out start {seed} was generated")
        return real_generate(env, task, seed, state_id, partition)

    monkeypatch.setattr(env_adapter, "generate", guarded)
    report, checks = smoke_sim.adapter_smoke("lift")
    assert checks["adapter_reserved_test_seed_locked"] is True
    assert generated == []  # the refusal comes before any generation
    assert report["seed_lock_checked_against"] == smoke_sim.MISSING_FREEZE
    assert isinstance(report["real_heldout_lock_open"], bool)


def test_missing_freeze_context_shuts_the_lock_and_restores_the_config() -> None:
    original = es.suites_config
    with smoke_sim.lock_pointed_at_missing_freeze() as why:
        assert why is not None and "does not exist" in why
        assert es.lock_status("can")[0] is False
    assert es.suites_config is original
