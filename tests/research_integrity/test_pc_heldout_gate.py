"""The held-out lock re-gated on the Level 2 freeze (Level 2 freeze §7.3, D-086).

The D-081 pattern for the point-cloud arm. These tests prove four things:
* the Level 2 gate opens only when its preconditions hold:
  * the Level 2 freeze is committed and unchanged, and so is its §9 code;
  * the 3 pinned point-cloud selection artifacts are present and tracked;
  * D-086 is in HEAD's DECISIONS;
  * the RGB study's gate (D-081) is still open and its 3 reused held-out reports unchanged;
  * the Level 1 lock is open;
* a Level 2 held-out evaluation is refused before any held-out state is read;
* the point-cloud validation and placement path cannot reach the held-out suite;
* the RGB gate and ``heldout.evaluate_entry`` still refuse what they refused (the RGB and
  Level 1 tests stay in place and must still pass).

An autouse guard makes any read of the held-out suite or reset to its seed block fail the
test, so no test here touches held-out states, even with the real lock open (KI-040).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from roboforge.eval import pc_heldout as ph
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
    assert ph.gate_status() == (True, [])
    assert len(ph.freeze_code_table()) == ph.N_CODE_FILES
    assert {ph.load_selection(r)["family"] for r in ph.SELECTIONS} == {"pc_diffusion"}


def test_the_gate_is_shut_without_the_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ph, "DECISION_ID", "D-999")
    ok, problems = ph.gate_status()
    assert not ok and any("D-999 is not in the committed" in p for p in problems)


def test_the_gate_is_shut_when_a_selection_is_missing_or_changed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    run = next(iter(ph.SELECTIONS))
    with monkeypatch.context() as mp:
        mp.setattr(ph, "SELECTIONS", {**ph.SELECTIONS, run: "0" * 64})
        ok, problems = ph.gate_status()
        assert not ok and any("differs from the pinned value" in p for p in problems)
    with monkeypatch.context() as mp:
        mp.setattr(ph, "SELECTIONS", {**ph.SELECTIONS, "s9_pc_missing_run": "0" * 64})
        ok, problems = ph.gate_status()
        assert not ok and any("does not exist" in p for p in problems)
    with monkeypatch.context() as mp:
        mp.setattr(ph, "SELECTION_DIR_REL", "reports/session9/no_such_dir")
        assert not ph.gate_status()[0]


def test_the_gate_is_shut_when_the_freeze_or_its_code_changed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as mp:
        mp.setattr(ph, "LEVEL2_FREEZE_SHA256", "0" * 64)
        ok, problems = ph.gate_status()
        assert not ok and any("differs from the freeze of" in p for p in problems)
    with monkeypatch.context() as mp:
        table = {**ph.freeze_code_table(), "src/roboforge/eval/pointcloud.py": "0" * 64}
        mp.setattr(ph, "freeze_code_table", lambda: table)
        ok, problems = ph.gate_status()
        assert not ok and problems == ["src/roboforge/eval/pointcloud.py: differs from the "
                                       "Level 2 freeze §9"]
    with monkeypatch.context() as mp:
        mp.setattr(ph, "LEVEL2_FREEZE_REL", "docs/NO_SUCH_FREEZE.md")
        assert ph.gate_status() == (False, ["docs/NO_SUCH_FREEZE.md does not exist"])


def test_the_gate_is_shut_when_the_rgb_study_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as mp:
        mp.setattr(rh, "gate_status", lambda: (False, ["shut for the test"]))
        ok, problems = ph.gate_status()
        assert not ok and problems == ["RGB study gate (D-081): shut for the test"]
    with monkeypatch.context() as mp:
        run = next(iter(ph.RGB_HELDOUT_REPORTS))
        mp.setattr(ph, "RGB_HELDOUT_REPORTS", {**ph.RGB_HELDOUT_REPORTS, run: "0" * 64})
        ok, problems = ph.gate_status()
        assert not ok and any("not the frozen RGB held-out report" in p for p in problems)


def test_the_gate_is_shut_when_the_level1_lock_is(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(es, "lock_status", lambda task: (False, "shut for the test"))
    ok, problems = ph.gate_status()
    assert not ok and "Level 1 lock: shut for the test" in problems


def test_inside_the_container_the_gate_needs_a_clean_host_tree(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ph, "_has_git", lambda: False)
    monkeypatch.setattr(rh, "_has_git", lambda: False)
    monkeypatch.setenv("ROBOFORGE_GIT_SHA", "x" * 40)
    monkeypatch.setenv("ROBOFORGE_GIT_DIRTY", "1")
    ok, problems = ph.gate_status()
    assert not ok and any("must be clean" in p for p in problems)
    monkeypatch.setenv("ROBOFORGE_GIT_DIRTY", "0")
    assert ph.gate_status() == (True, [])
    assert not ph._committed("reports/session9/pc_selection/x.json")[0]


def test_a_level2_heldout_run_is_refused_before_any_state_is_read(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run = next(iter(ph.SELECTIONS))
    with monkeypatch.context() as mp:  # gate shut
        mp.setattr(ph, "gate_status", lambda: (False, ["shut for the test"]))
        with pytest.raises(ph.PCHeldoutError, match="gate is shut"):
            ph.evaluate_selected(run, out_root=tmp_path)
    if not (REPO_ROOT / "runs" / run / "manifest.json").is_file():
        # Fresh clone or CI: the gitignored training record is absent, and the frozen code
        # reads it before the platform check (Session 12 fresh-clone finding).
        pytest.skip(f"frozen run record runs/{run} not present (gitignored)")
    with pytest.raises(ph.PCHeldoutError, match="canonical platform"):  # host platform
        ph.evaluate_selected(run, out_root=tmp_path)
    with pytest.raises(ph.PCHeldoutError, match="not one of the 3"):
        ph.evaluate_selected(next(iter(rh.SELECTIONS)), out_root=tmp_path)
    sel = ph.load_selection(run)
    eid = ph.eval_id(run, sel["selected"]["step"], CANONICAL["platform_id"])
    (tmp_path / eid).mkdir()
    import roboforge.provenance as prov
    monkeypatch.setattr(prov, "evaluation_platform", lambda: CANONICAL)
    with pytest.raises(ph.PCHeldoutError, match="--engineering-rerun"):
        ph.evaluate_selected(run, out_root=tmp_path)
    (tmp_path / eid / "manifest.json").write_text("{}")
    with pytest.raises(ph.PCHeldoutError, match="runs once"):
        ph.evaluate_selected(run, out_root=tmp_path, engineering_rerun="test")


def test_one_engineering_rerun_keeps_the_incomplete_attempt(tmp_path: Path) -> None:
    d = tmp_path / "eid"
    d.mkdir()
    incident = ph._prepare_out_dir(d, "container killed")
    assert incident["reason"] == "container killed" and not d.exists()
    assert (tmp_path / "eid.incomplete1").is_dir()
    d.mkdir()
    with pytest.raises(ph.PCHeldoutError, match="already happened once"):
        ph._prepare_out_dir(d, "again")


@pytest.mark.parametrize("suite", ["validation", "robustness_base", "ood_cube_shift"])
def test_the_level2_evaluation_path_admits_only_its_suites(suite: str) -> None:
    from roboforge.eval.validation import check_suite_allowed

    check_suite_allowed(ph.TASK, suite)
    with pytest.raises(ValueError, match="not evaluable here"):
        check_suite_allowed(ph.TASK, es.LOCKED_SUITE)


def test_the_rgb_gate_is_unaffected() -> None:
    assert rh.gate_status() == (True, [])
    assert rh.DECISION_ID == "D-081"


def test_the_session8_checks_run_on_the_host_and_make_runs_them_first(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The container mounts only reports/session9, so the RGB gate and the reused RGB reports
    are checked on the host, and `make pc-heldout` runs the host gate before the container."""
    called = []
    monkeypatch.setattr(rh, "gate_status", lambda: called.append(1) or (False, ["x"]))
    assert not ph.gate_status()[0] and called  # host: checked (and shut here)
    called.clear()
    monkeypatch.setattr(ph, "_has_git", lambda: False)
    monkeypatch.setenv("ROBOFORGE_GIT_SHA", "x" * 40)
    monkeypatch.setenv("ROBOFORGE_GIT_DIRTY", "0")
    assert ph.gate_status() == (True, []) and not called  # container: not reachable there
    lines = (REPO_ROOT / "Makefile").read_text().split("pc-heldout:", 1)[1].splitlines()[1:4]
    host = next(i for i, x in enumerate(lines) if "run_pc_heldout.py gate" in x)
    run = next(i for i, x in enumerate(lines) if "run_pc_heldout.py run" in x)
    assert host < run and "IN_SIM" not in lines[host]

