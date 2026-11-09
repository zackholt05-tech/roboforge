"""RGB frames of recorded demo states are aligned and render consistently (Session 8).

Frames are rendered from the raw file's states (no images exist in the dataset), so these tests
check the four things a visual policy depends on: the raw states are the committed dataset's
states, the camera model maps the task object onto its visible pixels (and a wrong mapping
does not), rendering has no hidden or privileged visual state, and the evaluation-start model
renders like the demonstration model. MASTER_SPEC §7.2 items 1-2.
"""

from __future__ import annotations

import numpy as np
import pytest

from roboforge.data.processed import load_processed, processed_path
from roboforge.data.rgb import (
    DEFAULT_CAMERAS,
    TASK_OBJECT_POS_FEATURE,
    DemoFrameRenderer,
    iter_raw_demos,
    raw_file,
)
from roboforge.data.rgb_checks import (
    compare_raw_lowdim,
    eval_model_parity,
    features_match_processed,
    projection_check,
    render_invariants,
)
from roboforge.sim.eval_states import load_suite

pytestmark = [pytest.mark.sim, pytest.mark.data, pytest.mark.render]


def _demo(task: str, demo: str = "demo_0"):
    for kind in ("raw", "low_dim"):
        try:
            raw_file(task, kind)
        except FileNotFoundError as e:
            pytest.skip(str(e))
    return next(iter_raw_demos(task, [demo]))


@pytest.mark.parametrize("task", ["lift", "can"])
def test_raw_file_holds_the_committed_states_actions_and_models(task: str) -> None:
    _demo(task)
    res = compare_raw_lowdim(task)
    assert res["pass"] and res["demos"] == 200


@pytest.mark.parametrize("task", ["lift", "can"])
def test_frame_state_is_the_processed_observation_state(task: str) -> None:
    d = _demo(task)
    proc = load_processed(processed_path(task, "lowdim_v1"), [d.demo])[d.demo]
    r = DemoFrameRenderer(task)
    try:
        r.load_model(d.model_xml, d.states[0])
        assert features_match_processed(r, d.states, proc, d.actions)["pass"]
    finally:
        r.close()


@pytest.mark.parametrize("task", ["lift", "can"])
def test_projected_object_lands_on_its_visible_pixels(task: str) -> None:
    from roboforge.data.pipeline import load_specs

    d = _demo(task)
    spec, _ = load_specs(task)
    lo, hi = spec.slices()[TASK_OBJECT_POS_FEATURE[task]]
    pos = load_processed(processed_path(task, "lowdim_v1"), [d.demo])[d.demo]["obs"][0, lo:hi]
    r = DemoFrameRenderer(task)
    try:
        r.load_model(d.model_xml, d.states[0])
        p = projection_check(r, d.states[0], "agentview", 84, object_pos=pos.astype(np.float64),
                             other_camera="robot0_eye_in_hand")
        assert p["area_px"] > 0 and p["on_object"] and p["dist_to_centroid_px"] < 1.5
        # Negative control: a point 0.2 m above the object must not land on it.
        wrong = projection_check(r, d.states[0], "agentview", 84,
                                 object_pos=pos.astype(np.float64) + np.array([0.0, 0.0, 0.2]))
        assert not wrong["on_object"]
    finally:
        r.close()


@pytest.mark.parametrize("camera", DEFAULT_CAMERAS)
def test_rendering_has_no_hidden_or_privileged_visual_state(camera: str) -> None:
    d = _demo("lift")
    res = render_invariants("lift", d.model_xml, d.states, d.actions, camera, 84, n_steps=10)
    assert res["rerender_bitwise_equal"]
    assert res["sites_hidden_bitwise_equal"]  # no visualization site is drawn
    assert res["live_vs_fresh_bitwise_equal"]  # stepping leaves no visual trace


@pytest.mark.parametrize("task", ["lift", "can"])
def test_evaluation_model_renders_like_the_demonstration_model(task: str) -> None:
    d = _demo(task)
    res = eval_model_parity(task, d.model_xml, d.states[0], load_suite(task, "validation")[0],
                            "agentview", 84)
    assert res["comparable"] and res["pass"]
    if task == "can":  # one model for every reset; only the KI-014 markers differ
        assert res["bitwise_equal"]
