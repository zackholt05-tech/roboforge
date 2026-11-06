"""The checkpoint-selection rule (RESEARCH_PROTOCOL §7) and its refusals."""

from __future__ import annotations

import pytest

from roboforge.eval.selection import select_checkpoint


def _manifest(steps):
    return {"run_id": "r", "checkpoints": [{"step": s, "sha256": f"h{s}"} for s in steps]}


def _eval(step, k, fs, **kw):
    e = {"train_run_id": "r", "eval_run_id": f"e{step}", "eligible_for_selection": True,
         "partial": False, "n_episodes": 50, "successes": k, "first_success_step": {"mean": fs},
         "checkpoint": {"step": step, "sha256": f"h{step}"}, "wilson95": [0, 1],
         "platform": {"platform_id": "p"}}
    e.update(kw)
    return e


def test_highest_success_wins() -> None:
    r = select_checkpoint(_manifest([1, 2, 3]), [_eval(1, 40, 50.0), _eval(2, 48, 70.0),
                                                 _eval(3, 45, 40.0)])
    assert r["chosen"]["step"] == 2


def test_tie_goes_to_lower_mean_first_success_then_earlier() -> None:
    m = _manifest([1, 2, 3, 4])
    r = select_checkpoint(m, [_eval(1, 50, 52.0), _eval(2, 50, 49.5), _eval(3, 50, 49.5),
                              _eval(4, 49, 30.0)])
    assert r["chosen"]["step"] == 2 and r["n_tied_on_success"] == 3


def test_all_zero_success_picks_the_earliest() -> None:
    r = select_checkpoint(_manifest([5, 10]), [_eval(10, 0, None), _eval(5, 0, None)])
    assert r["chosen"]["step"] == 5


@pytest.mark.parametrize("bad", [
    {"eligible_for_selection": False}, {"partial": True}, {"n_episodes": 49},
    {"checkpoint": {"step": 1, "sha256": "other"}}, {"train_run_id": "x"}])
def test_incomplete_or_foreign_evaluations_are_refused(bad) -> None:
    with pytest.raises(ValueError):
        select_checkpoint(_manifest([1]), [_eval(1, 1, 1.0, **bad)])


def test_missing_duplicate_or_mixed_platform_evaluations_are_refused() -> None:
    with pytest.raises(ValueError):
        select_checkpoint(_manifest([1, 2]), [_eval(1, 1, 1.0)])
    with pytest.raises(ValueError):
        select_checkpoint(_manifest([1]), [_eval(1, 1, 1.0), _eval(1, 2, 1.0)])
    with pytest.raises(ValueError):
        select_checkpoint(_manifest([1, 2]), [_eval(1, 1, 1.0),
                                              _eval(2, 1, 1.0, platform={"platform_id": "q"})])
