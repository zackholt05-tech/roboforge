"""Alignment and rendering checks for RGB frames of recorded states (Session 8 feasibility).

Each function returns a JSON-able record; :mod:`scripts.rgb_feasibility` runs them and writes
the evidence, and ``tests/simulator/test_rgb_alignment.py`` asserts on them. The checks answer
the MASTER_SPEC §7.2 questions before any RGB dataset is generated:

* :func:`compare_raw_lowdim`: the raw file (the RGB source) holds the very states, actions and
  model XML from which the committed low-dim dataset was built, bitwise, for every demo;
* :func:`features_match_processed`: the canonical features of the raw states equal the committed
  processed observations bitwise, so frame ``t`` and ``obs[t]`` describe one state;
* :func:`projection_check`: the task object's world position, projected with the recorded camera
  intrinsics and extrinsics, lands on the object's visible pixels in the segmentation, and two
  negative controls (vertically flipped mask, the other camera's mask) do not;
* :func:`render_invariants`: re-rendering is bitwise deterministic; no visualization site is
  visible; and a state reached by stepping the live simulator renders exactly like the same
  state restored into a freshly compiled model (no hidden visual state such as runtime site
  colours, which robosuite derives from privileged distances in Lift);
* :func:`eval_model_parity`: the evaluation-start model (robosuite's default model, with its two
  mocap markers, KI-014) renders a demonstration state like the demonstration's own model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from roboforge.data.rgb import (
    TASK_OBJECT_BODY,
    DemoFrameRenderer,
    body_geom_ids,
    frame_sha256,
    mask_stats,
    object_mask,
    project_world_point,
)

# Evaluation-model parity tolerates differences within this many pixels of the task object
# (Lift samples the cube size per reset, which moves its edge and shadow border by ~1 px).
PARITY_EDGE_PX = 2
# A projection counts as "on the object" if it lies on a mask pixel or within this many
# pixels of a mask pixel centre (half a pixel diagonal plus rounding).
ON_MASK_TOL_PX = 1.0


def compare_raw_lowdim(task: str) -> dict[str, Any]:
    """Bitwise comparison of the raw and low-dim files: model XML, states and actions per demo."""
    import h5py

    from roboforge.data.inspect import demo_keys
    from roboforge.data.rgb import raw_file

    raw_path, raw_sha = raw_file(task, "raw")
    low_path, low_sha = raw_file(task, "low_dim")
    out: dict[str, Any] = {"raw_sha256": raw_sha, "low_dim_sha256": low_sha, "demos": 0,
                           "frames": 0, "xml_mismatch": [], "state_mismatch": [],
                           "action_mismatch": []}
    with h5py.File(raw_path, "r") as fr, h5py.File(low_path, "r") as fl:
        names = demo_keys(fr)
        if names != demo_keys(fl):
            raise AssertionError(f"{task}: raw and low-dim demo lists differ")
        for d in names:
            gr, gl = fr["data"][d], fl["data"][d]
            out["demos"] += 1
            out["frames"] += int(gr["states"].shape[0])
            if str(gr.attrs["model_file"]) != str(gl.attrs["model_file"]):
                out["xml_mismatch"].append(d)
            if not np.array_equal(gr["states"][()], gl["states"][()]):
                out["state_mismatch"].append(d)
            if not np.array_equal(gr["actions"][()], gl["actions"][()]):
                out["action_mismatch"].append(d)
    out["pass"] = not (out["xml_mismatch"] or out["state_mismatch"] or out["action_mismatch"])
    return out


def features_match_processed(renderer: DemoFrameRenderer, states: np.ndarray,
                             processed: dict[str, np.ndarray],
                             actions: np.ndarray) -> dict[str, Any]:
    """Canonical features of ``states`` (float32) vs the processed ``obs``; actions likewise."""
    from roboforge.data.features import FeatureExtractor
    from roboforge.data.pipeline import load_specs

    spec, _ = load_specs(renderer.task)
    feats = FeatureExtractor(spec).bind(renderer.model).from_states(states).astype(np.float32)
    obs_equal = feats.shape == processed["obs"].shape and np.array_equal(feats, processed["obs"])
    act_equal = np.array_equal(actions.astype(np.float32), processed["actions"])
    diff = (float(np.abs(feats - processed["obs"]).max())
            if feats.shape == processed["obs"].shape else float("inf"))
    return {"steps": int(len(states)), "obs_bitwise_equal": bool(obs_equal),
            "obs_max_abs_diff": diff, "actions_bitwise_equal": bool(act_equal),
            "pass": bool(obs_equal and act_equal)}


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation of ``mask`` by a ``(2 radius + 1)`` square (pure numpy)."""
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            src = mask[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
            out[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)] |= src
    return out


def _on_object(stats: dict[str, Any]) -> bool:
    d = stats["dist_to_mask_px"]
    return bool(stats["projection_on_mask"] or (d is not None and np.isfinite(d)
                                               and d <= ON_MASK_TOL_PX))


def projection_check(renderer: DemoFrameRenderer, state: np.ndarray, camera: str, size: int,
                     object_pos: np.ndarray | None = None,
                     other_camera: str | None = None) -> dict[str, Any]:
    """Project the task object onto ``camera``'s image and compare with its visible pixels.

    ``object_pos`` defaults to the body position from scratch kinematics; callers pass the
    committed processed ``*_pos`` observation to tie the image to the dataset value. Controls:
    the same projection against the vertically flipped mask, and (``other_camera``) against
    the other camera's mask; both are expected to fail whenever the object is visible.
    """
    pos = renderer.object_position(state) if object_pos is None else np.asarray(object_pos)
    cam = renderer.camera(state, camera, size)
    u, v, depth = project_world_point(cam, pos)
    mask = object_mask(renderer.segmentation(state, camera, size), renderer.object_geoms)
    stats = mask_stats(mask, u, v)
    out = {"camera": camera, "size": size, "object_pos": [float(x) for x in pos],
           "depth_m": depth, **stats, "on_object": _on_object(stats)}
    # A control is informative only if it moves the projection by more than the object's
    # pixel radius (plus the on-object tolerance); otherwise it lands on the object anyway.
    radius = float(np.sqrt(mask.sum() / np.pi))
    flip_shift = abs(2.0 * v - size) if np.isfinite(v) else float("nan")
    out["control_flipped_applicable"] = bool(mask.any() and np.isfinite(flip_shift)
                                             and flip_shift > radius + ON_MASK_TOL_PX)
    out["control_flipped_on_object"] = (_on_object(mask_stats(mask[::-1], u, v))
                                        if mask.any() else None)
    if other_camera is not None:
        other = object_mask(renderer.segmentation(state, other_camera, size),
                            renderer.object_geoms)
        uo, vo, _ = project_world_point(renderer.camera(state, other_camera, size), pos)
        shift = float(np.hypot(u - uo, v - vo))
        r_other = float(np.sqrt(other.sum() / np.pi))
        out["control_other_camera_applicable"] = bool(other.any() and np.isfinite(shift)
                                                      and shift > r_other + ON_MASK_TOL_PX)
        out["control_other_camera_on_object"] = (_on_object(mask_stats(other, u, v))
                                                  if other.any() else None)
    return out


def _render_with_sites_hidden(model: Any, state: np.ndarray, camera: str,
                              size: int) -> np.ndarray:
    """Render like ``ShadowRenderer`` but with every site group switched off."""
    import mujoco

    from roboforge.sim.shadow_render import ROBOSUITE_GEOMGROUP

    d = mujoco.MjData(model)
    d.time = state[0]
    d.qpos[:] = state[1:1 + model.nq]
    d.qvel[:] = state[1 + model.nq:]
    mujoco.mj_forward(model, d)  # scratch data only
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = ROBOSUITE_GEOMGROUP
    opt.sitegroup[:] = 0
    r = mujoco.Renderer(model, height=size, width=size)
    try:
        r.update_scene(d, camera=camera, scene_option=opt)
        r.render()  # discard the first render of the fresh context (WARM_UP_RENDERS)
        return np.ascontiguousarray(r.render())
    finally:
        r.close()


def render_invariants(task: str, model_xml: str, states: np.ndarray, actions: np.ndarray,
                      camera: str, size: int, n_steps: int = 20) -> dict[str, Any]:
    """Determinism, hidden sites and live-vs-restored equality for one demo and camera."""
    a = DemoFrameRenderer(task)
    try:
        a.load_model(model_xml, states[0])
        f1 = a.rgb(states[0], camera, size)
        f2 = a.rgb(states[0], camera, size)
        hidden = _render_with_sites_hidden(a.model, states[0], camera, size)
        env = a.env  # step the live simulator with the recorded actions
        for act in actions[:n_steps]:
            env.step(np.asarray(act, dtype=np.float64))
        live_state = np.array(env.sim.get_state().flatten())
        live_frame = a.rgb(live_state, camera, size)
    finally:
        a.close()
    b = DemoFrameRenderer(task)
    try:
        b.load_model(model_xml, states[0])
        fresh_frame = b.rgb(live_state, camera, size)
    finally:
        b.close()
    out = {"camera": camera, "size": size, "steps_stepped": int(min(n_steps, len(actions))),
           "rerender_bitwise_equal": bool(np.array_equal(f1, f2)),
           "sites_hidden_bitwise_equal": bool(np.array_equal(f1, hidden)),
           "sites_hidden_pixels_differing": int(np.any(f1 != hidden, axis=-1).sum()),
           "live_vs_fresh_bitwise_equal": bool(np.array_equal(live_frame, fresh_frame)),
           "live_vs_fresh_pixels_differing": int(np.any(live_frame != fresh_frame,
                                                        axis=-1).sum()),
           "frame_sha256_t0": frame_sha256(f1)}
    out["pass"] = (out["rerender_bitwise_equal"] and out["sites_hidden_bitwise_equal"]
                   and out["live_vs_fresh_bitwise_equal"])
    return out


def eval_model_parity(task: str, model_xml: str, state0: np.ndarray, eval_init: Any,
                      camera: str, size: int) -> dict[str, Any]:
    """Render a demo state with the demo's model and with an evaluation start's model.

    The evaluation model is the snapshot XML of ``eval_init`` (a validation start). For Can it
    differs from the dataset model only by the two KI-014 mocap markers and site colours; for
    Lift the cube size is sampled per reset, so pixels on the cube and on its anti-aliased
    edge may differ. It passes when no pixel differs farther than :data:`PARITY_EDGE_PX` from
    the task object's mask in either render.
    """
    from roboforge.sim.factory import load_sim_config, make_env
    from roboforge.sim.shadow_render import ShadowRenderer
    from roboforge.sim.snapshot import restore

    demo = DemoFrameRenderer(task)
    try:
        demo.load_model(model_xml, state0)
        f_demo = demo.rgb(state0, camera, size)
        m_demo = object_mask(demo.segmentation(state0, camera, size), demo.object_geoms)
    finally:
        demo.close()
    env = make_env(load_sim_config(task))
    try:
        restore(env, eval_init)
        model = env.sim.model._model
        if 1 + model.nq + model.nv != state0.shape[0]:
            return {"camera": camera, "size": size, "comparable": False}
        sr = ShadowRenderer(model)
        try:
            sr.rgb(state0, camera, size, size)  # discard the cold first render (WARM_UP_RENDERS)
            f_eval = sr.rgb(state0, camera, size, size)
            m_eval = object_mask(sr.segmentation(state0, camera, size, size),
                                 body_geom_ids(model, TASK_OBJECT_BODY[task]))
        finally:
            sr.close()
    finally:
        env.close()
    differ = np.any(f_demo != f_eval, axis=-1)
    outside = differ & ~dilate(m_demo | m_eval, PARITY_EDGE_PX)
    return {"camera": camera, "size": size, "comparable": True,
            "eval_state_id": eval_init.state_id,
            "pixels_differing": int(differ.sum()), "pixels_total": int(differ.size),
            "edge_tolerance_px": PARITY_EDGE_PX,
            "pixels_differing_outside_object": int(outside.sum()),
            "max_abs_diff": int(np.abs(f_demo.astype(int) - f_eval.astype(int)).max()),
            "bitwise_equal": bool(not differ.any()),
            "pass": bool(not outside.any())}
