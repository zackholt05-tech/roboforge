"""The held-out lock re-gated on the RGB freeze (RGB freeze §9, D-081).

The Level 1 lock has been open since the Level 1 freeze. These tests prove three things:
* the RGB gate opens only when its preconditions hold:
  * the RGB freeze is committed and unchanged, and so is its §10 code;
  * the 6 pinned selection artifacts are present and tracked;
  * D-081 is in HEAD's DECISIONS;
  * the Level 1 lock is open;
* an RGB held-out evaluation is refused before any held-out state is read;
* ``heldout.evaluate_entry`` still refuses every Level 1 configuration.

An autouse guard makes any read of the held-out suite or reset to its seed block fail the
test, so no test here touches held-out states, even with the real lock open (KI-040).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboforge.eval import freeze as fz
from roboforge.eval import heldout
from roboforge.eval import rgb_heldout as rh
from roboforge.paths import REPO_ROOT
from roboforge.sim import eval_states as es

CANONICAL = {"platform_id": "linux-aarch64-container-1f685638f28c"}


class TouchedHeldout(AssertionError):
    """A test reached the held-out states."""


@pytest.fixture(autouse=True)
def heldout_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    real_load, real_seed = es.load_suite, es.check_seed_allowed

    def load(task: str, suite: str):  # noqa: ANN202
        if suite == es.LOCKED_SUITE:
            raise TouchedHeldout(f"{task}/{suite} was read")
        return real_load(task, suite)

    def seed_ok(task: str, seed: int) -> str:
        if 900000 <= seed < 1_000_000:
            raise TouchedHeldout(f"reset to held-out seed {seed}")
        return real_seed(task, seed)

    monkeypatch.setattr(es, "load_suite", load)
    monkeypatch.setattr(es, "check_seed_allowed", seed_ok)


def test_the_gate_is_open_with_every_precondition_committed() -> None:
    assert rh.gate_status() == (True, [])
    assert len(rh.freeze_code_table()) == 12
    assert {rh.load_selection(r)["family"] for r in rh.SELECTIONS} == set(rh.FAMILY_ORDER)


def test_the_gate_is_shut_without_the_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rh, "DECISION_ID", "D-999")
    ok, problems = rh.gate_status()
    assert not ok and any("D-999 is not in the committed" in p for p in problems)


def test_the_gate_is_shut_when_a_selection_is_missing_or_changed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    run = next(iter(rh.SELECTIONS))
    with monkeypatch.context() as mp:
        mp.setattr(rh, "SELECTIONS", {**rh.SELECTIONS, run: "0" * 64})
        ok, problems = rh.gate_status()
        assert not ok and any("differs from the pinned value" in p for p in problems)
    with monkeypatch.context() as mp:
        mp.setattr(rh, "SELECTIONS", {**rh.SELECTIONS, "s8_rgb_missing_run": "0" * 64})
        ok, problems = rh.gate_status()
        assert not ok and any("does not exist" in p for p in problems)
    with monkeypatch.context() as mp:
        mp.setattr(rh, "SELECTION_DIR_REL", "reports/session8/no_such_dir")
        assert not rh.gate_status()[0]


def test_the_gate_is_shut_when_the_freeze_or_its_code_changed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as mp:
        mp.setattr(rh, "RGB_FREEZE_SHA256", "0" * 64)
        ok, problems = rh.gate_status()
        assert not ok and any("differs from the freeze of" in p for p in problems)
    with monkeypatch.context() as mp:
        table = {**rh.freeze_code_table(), "src/roboforge/eval/rgb.py": "0" * 64}
        mp.setattr(rh, "freeze_code_table", lambda: table)
        ok, problems = rh.gate_status()
        assert not ok and problems == ["src/roboforge/eval/rgb.py: differs from the RGB "
                                       "freeze §10"]
    with monkeypatch.context() as mp:
        mp.setattr(rh, "RGB_FREEZE_REL", "docs/NO_SUCH_FREEZE.md")
        assert rh.gate_status() == (False, ["docs/NO_SUCH_FREEZE.md does not exist"])


def test_the_gate_is_shut_when_the_level1_lock_is(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(es, "lock_status", lambda task: (False, "shut for the test"))
    ok, problems = rh.gate_status()
    assert not ok and problems == ["Level 1 lock: shut for the test"]


def test_inside_the_container_the_gate_needs_a_clean_host_tree(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rh, "_has_git", lambda: False)
    monkeypatch.setenv("ROBOFORGE_GIT_SHA", "x" * 40)
    monkeypatch.setenv("ROBOFORGE_GIT_DIRTY", "1")
    ok, problems = rh.gate_status()
    assert not ok and any("must be clean" in p for p in problems)
    monkeypatch.setenv("ROBOFORGE_GIT_DIRTY", "0")
    assert rh.gate_status() == (True, [])
    assert not rh._committed("reports/session8/rgb_selection/x.json")[0]


def test_an_rgb_heldout_run_is_refused_before_any_state_is_read(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run = next(iter(rh.SELECTIONS))
    with monkeypatch.context() as mp:  # gate shut
        mp.setattr(rh, "gate_status", lambda: (False, ["shut for the test"]))
        with pytest.raises(rh.RGBHeldoutError, match="gate is shut"):
            rh.evaluate_selected(run, out_root=tmp_path)
    with pytest.raises(rh.RGBHeldoutError, match="canonical platform"):  # host platform
        rh.evaluate_selected(run, out_root=tmp_path)
    with pytest.raises(rh.RGBHeldoutError, match="not one of the 6"):
        rh.evaluate_selected("s7_bc_mlp_cos_can_seed0_ae8152e3", out_root=tmp_path)
    sel = rh.load_selection(run)
    eid = rh.eval_id(run, sel["family"], sel["selected"]["step"], CANONICAL["platform_id"])
    (tmp_path / eid).mkdir()
    import roboforge.provenance as prov
    monkeypatch.setattr(prov, "evaluation_platform", lambda: CANONICAL)
    with pytest.raises(rh.RGBHeldoutError, match="--engineering-rerun"):
        rh.evaluate_selected(run, out_root=tmp_path)
    (tmp_path / eid / "manifest.json").write_text("{}")
    with pytest.raises(rh.RGBHeldoutError, match="runs once"):
        rh.evaluate_selected(run, out_root=tmp_path, engineering_rerun="test")


def test_one_engineering_rerun_keeps_the_incomplete_attempt(tmp_path: Path) -> None:
    d = tmp_path / "eid"
    d.mkdir()
    incident = rh._prepare_out_dir(d, "container killed")
    assert incident["reason"] == "container killed" and not d.exists()
    assert (tmp_path / "eid.incomplete1").is_dir()
    d.mkdir()
    with pytest.raises(rh.RGBHeldoutError, match="already happened once"):
        rh._prepare_out_dir(d, "again")


def test_the_rgb_validation_path_cannot_reach_the_heldout_suite() -> None:
    from roboforge.eval.validation import check_suite_allowed

    with pytest.raises(ValueError, match="not evaluable here"):
        check_suite_allowed(rh.TASK, es.LOCKED_SUITE)


@pytest.mark.parametrize("task", ["lift", "can"])
def test_level1_evaluate_entry_still_refuses_every_level1_configuration(
        task: str, monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = fz.load_freeze()
    entries = fz.entries(frozen, task)
    assert len(entries) == 13
    for entry in entries:  # host: never the canonical platform
        with pytest.raises(heldout.HeldoutError, match="canonical platform"):
            heldout.evaluate_entry(entry)
    runs = REPO_ROOT / "runs"
    if not runs.is_dir():
        pytest.skip("local runs/ not present")
    monkeypatch.setattr(heldout, "evaluation_platform", lambda: CANONICAL)
    monkeypatch.setattr(es, "load_suite", lambda task, suite: [])
    monkeypatch.setattr(heldout, "_policy", lambda entry, frozen: (None, None))
    for entry in entries:  # canonical: every frozen entry already ran once
        eid = heldout.eval_id(entry, CANONICAL["platform_id"], rehearsal=False)
        assert json.loads((runs / eid / "manifest.json").read_text())["eval_run_id"] == eid
        with pytest.raises(heldout.HeldoutError, match="already ran"):
            heldout.evaluate_entry(entry)
    rgb = {"task": "lift", "family": "rgb_diffusion", "train_run_id": next(iter(rh.SELECTIONS)),
           "step": 55000, "checkpoint_sha256": "0" * 64}
    with pytest.raises(heldout.HeldoutError, match="not an entry of the freeze"):
        heldout.evaluate_entry(rgb)
