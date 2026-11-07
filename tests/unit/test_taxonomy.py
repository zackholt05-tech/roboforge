"""Failure taxonomy (RESEARCH_PROTOCOL §14.5, D-065) on synthetic recorded episodes: every rule,
the rule order, and the Can goal-bin footprint."""

from __future__ import annotations

import numpy as np
import pytest

from roboforge.eval import taxonomy as tx

LIFT_BODIES = ["robot0_base", "cube_main"]
CAN_BODIES = ["bin1", "bin2", "robot0_base", "Can_main"]


def _episode(task: str, obj_z: np.ndarray, grip: np.ndarray, *, eef_offset: float = 0.0,
             success: bool = False, stop: str = "truncated_horizon",
             final_xy: tuple[float, float] = (0.0, 0.0)) -> tuple[dict, dict]:
    n = len(obj_z)
    bodies = LIFT_BODIES if task == "lift" else CAN_BODIES
    pos = np.zeros((n, len(bodies), 3))
    oi = bodies.index("cube_main" if task == "lift" else "Can_main")
    pos[:, oi, 2] = obj_z
    pos[-1, oi, :2] = final_xy
    if task == "can":
        pos[:, bodies.index("bin2")] = (0.1, 0.28, 0.8)
    obs = np.zeros((n, 19))
    obs[:, 0:3] = pos[:, oi] + (eef_offset, 0.0, 0.05)  # robot0_eef_pos is dims 0-2
    ex = np.zeros((n - 1, 7))
    ex[:, -1] = np.where(grip, 1.0, -1.0)
    meta = {"task": task, "bodies": {"names": bodies},
            "outcome": {"success": success, "stop_reason": stop, "steps": n - 1}}
    return meta, {"body_pos": pos, "obs": obs, "executed_action": ex,
                  "clip_mask": np.zeros((n - 1, 7), dtype=bool)}


Z0 = 0.83  # dropped 1 cm onto the table at t = 0, rests at 0.82
FLAT = np.r_[Z0, np.full(59, 0.82)]
LIFT_THEN_DROP = np.r_[Z0, np.full(19, 0.82), np.full(20, 0.85), np.full(20, 0.82)]
LIFT_HELD = np.r_[Z0, np.full(19, 0.82), np.full(40, 0.85)]
CLOSE_LATE = np.r_[np.zeros(15, dtype=bool), np.ones(44, dtype=bool)]


@pytest.mark.parametrize(("args", "kw", "want"), [
    ((FLAT, CLOSE_LATE), {"success": True, "stop": "terminated_success"}, "success"),
    ((FLAT, CLOSE_LATE), {"stop": "error"}, "engineering_error"),
    ((FLAT, np.zeros(59, dtype=bool)), {}, "gripper_never_closed"),
    ((FLAT, CLOSE_LATE), {"eef_offset": 0.5}, "no_grasp_drift"),
    ((FLAT, CLOSE_LATE), {}, "no_grasp_near"),
    ((LIFT_HELD, CLOSE_LATE), {}, "timeout_holding"),
    ((LIFT_THEN_DROP, CLOSE_LATE), {}, "dropped"),
])
def test_lift_rules(args, kw, want) -> None:
    assert tx.classify(*_episode("lift", *args, **kw))["category"] == want


def test_rise_counts_from_the_rest_height_not_frame_zero() -> None:
    # 0.83 -> 0.82 settle, then +1.5 cm with the gripper closed: lifted relative to 0.82,
    # although it never exceeds the frame-0 height by 1 cm.
    z = np.r_[Z0, np.full(19, 0.82), np.full(20, 0.835), np.full(20, 0.82)]
    lab = tx.classify(*_episode("lift", z, CLOSE_LATE))
    assert lab["category"] == "dropped" and lab["max_rise_m"] == pytest.approx(0.015)


def test_rise_with_an_open_gripper_is_not_a_grasp() -> None:
    open_then_close = np.r_[np.zeros(45, dtype=bool), np.ones(14, dtype=bool)]
    z = np.r_[Z0, np.full(19, 0.82), np.full(20, 0.85), np.full(20, 0.82)]  # pushed up, open
    assert tx.classify(*_episode("lift", z, open_then_close))["category"] == "no_grasp_near"


def test_error_outranks_every_task_rule() -> None:
    assert tx.classify(*_episode("lift", LIFT_HELD, CLOSE_LATE, stop="error"))["category"] \
        == "engineering_error"


@pytest.mark.parametrize(("final_xy", "want"), [((0.2, 0.4), "in_goal_bin_not_success"),
                                                ((0.1, -0.25), "dropped"),
                                                ((0.1, 0.28 + 0.25), "dropped")])
def test_can_goal_bin_footprint(final_xy, want) -> None:
    lab = tx.classify(*_episode("can", LIFT_THEN_DROP, CLOSE_LATE, final_xy=final_xy))
    assert lab["category"] == want
    assert lab["final_object_in_goal_bin_footprint"] is (want == "in_goal_bin_not_success")


def test_can_holding_at_the_horizon_outranks_the_bin() -> None:
    lab = tx.classify(*_episode("can", LIFT_HELD, CLOSE_LATE, final_xy=(0.2, 0.4)))
    assert lab["category"] == "timeout_holding"


def test_counts_cover_every_category_and_reject_unknown_labels() -> None:
    c = tx.counts([{"category": "dropped"}, {"category": "success"}])
    assert list(c) == tx.taxonomy_config()["categories"] and sum(c.values()) == 2
    with pytest.raises(ValueError, match="outside the taxonomy"):
        tx.counts([{"category": "other"}])

