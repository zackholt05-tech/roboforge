"""Simulator-contract checks for Session 2 (KI-014, KI-016, D-024, D-025).

Per task, writes ``reports/session2/sim_contract_<task>.json``:

A. **Markers on demos (KI-014):** each sampled demo (``replay_compare.select_demos``) replayed
   on its dataset XML and on the same XML with robosuite's two mocap markers inserted.
   Gate: bitwise-identical states.
B. **Markers on eval starts:** 5 validation starts, 400 random actions, eval XML vs eval XML
   without markers. Gate: bitwise identical.
C. **Marker renders:** every camera, with vs without markers. Gate: identical pixels.
D. **Dataset model vs eval model:** name-aligned parameter diff (reported; expected
   differences only in markers, index fields, site colours and Lift's per-start cube).
E. **Live vs saved-XML model (KI-016):** what ``mj_saveLastXML`` changes (reported).
F. **Purity (D-024):** RoboForgeEnv with recording, rendering and extra ``get_success`` calls
   vs a bare robosuite loop on the same actions. Gate: bitwise. Also reported: the
   perturbation caused by ``mj_forward`` on live data after every step (it rewrites the
   solver warm start), which is why the adapter never calls it.
G. **Episode isolation:** a start replayed after other episodes and a demo replay in the same
   instance, and in a fresh instance. Gate: bitwise-identical records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from roboforge.data.fetch import load_manifest  # noqa: E402
from roboforge.data.pipeline import load_task_artifacts  # noqa: E402
from roboforge.eval.policies import DemoReplayPolicy, RandomPolicy  # noqa: E402
from roboforge.eval.rollout import RenderSpec, run_episode  # noqa: E402
from roboforge.paths import REPO_ROOT  # noqa: E402
from roboforge.provenance import environment_snapshot, evaluation_platform  # noqa: E402
from roboforge.sim import model_compare as mc  # noqa: E402
from roboforge.sim.env_adapter import RoboForgeEnv  # noqa: E402
from roboforge.sim.eval_states import load_suite  # noqa: E402
from roboforge.sim.replay_compare import bare_replay, select_demos  # noqa: E402
from roboforge.sim.shadow_render import ShadowRenderer  # noqa: E402
from roboforge.sim.snapshot import from_demo, generate  # noqa: E402
from roboforge.sim.state_replay import reset_to_state  # noqa: E402

OUT = REPO_ROOT / "reports" / "session2"
N_STEPS = 400


def _random_actions(i: int) -> np.ndarray:
    a = np.random.default_rng([0, i]).uniform(-1, 1, size=(N_STEPS, 7))
    a[:, -1] = np.where((np.arange(N_STEPS) // 40) % 2 == 0, -1.0, 1.0)
    return a


def _compiled(raw_env: Any, xml: str) -> Any:
    return mujoco.MjModel.from_xml_string(raw_env.edit_model_xml(xml))


def run_task(task: str) -> dict[str, Any]:
    """All checks for one task."""
    env = RoboForgeEnv(task)
    raw = env.raw_env
    val = load_suite(task, "validation")
    snippets = mc.marker_snippets(val[0].model_xml)
    _, files = load_manifest()
    entry = next(f for f in files if f.task == task and f.hdf5_type == "low_dim")
    split = load_task_artifacts(task)["split"]["splits"]
    rep: dict[str, Any] = {"task": task, "environment": environment_snapshot(),
                           "platform": evaluation_platform()}
    with h5py.File(entry.destination(), "r") as f:
        lengths = {d: int(f["data"][d].attrs["num_samples"]) for d in f["data"]}
        demos = [p["demo"] for p in select_demos(lengths, split["valid"], task)]
        a_res = {}
        for d in demos:
            g = f["data"][d]
            xml, states, actions = g.attrs["model_file"], g["states"][()], g["actions"][()]
            t1 = bare_replay(raw, xml, states, actions)
            t2 = bare_replay(raw, mc.insert_markers(xml, snippets), states, actions)
            a_res[d] = float(np.abs(t1 - t2).max())
        g1 = f["data"]["demo_1"]
        demo1 = from_demo(task, "demo_1", g1.attrs["model_file"], g1["states"][()],
                          "demo_train", "-")
    rep["A_markers_on_demos_max_abs_state_diff"] = a_res
    b_res = {}
    for i, init in enumerate(val[:5]):
        acts = _random_actions(i)
        t1 = bare_replay(raw, init.model_xml, init.state[None], acts)
        t2 = bare_replay(raw, mc.strip_markers(init.model_xml), init.state[None], acts)
        b_res[init.state_id] = float(np.abs(t1 - t2).max())
    rep["B_markers_on_eval_starts_max_abs_state_diff"] = b_res
    c_res = {}
    m_with, m_without = _compiled(raw, val[0].model_xml), _compiled(
        raw, mc.strip_markers(val[0].model_xml))
    r1, r2 = ShadowRenderer(m_with), ShadowRenderer(m_without)
    cams = [mujoco.mj_id2name(m_with, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in range(m_with.ncam)]
    for cam in cams:
        x, y = r1.rgb(val[0].state, cam, 256, 256), r2.rgb(val[0].state, cam, 256, 256)
        c_res[cam] = int(np.abs(x.astype(int) - y.astype(int)).max())
    r1.close()
    r2.close()
    rep["C_marker_render_max_pixel_diff"] = c_res
    # Lift samples the cube size at every reset, so the cube itself is excluded there.
    sampled = ("cube_main",) if task == "lift" else ()
    rep["D_dataset_vs_eval_model_param_diff"] = mc.param_diff(
        _compiled(raw, demo1.model_xml), m_with, exclude_bodies=sampled)
    np.random.seed(100000)
    raw.reset()
    live = raw.sim.model._model
    rep["E_live_vs_saved_xml_param_diff"] = mc.param_diff(
        live, _compiled(raw, generate(raw, task, 100000, "probe", "eval_adhoc").model_xml))
    f_res = {}
    for init in val[:3]:
        rec = run_episode(env, RandomPolicy(env.action_spec, 0), init,
                          render=RenderSpec(("agentview", "robot0_eye_in_hand"), 64, 64, 20))
        env.get_success()
        bare = bare_replay(raw, init.model_xml, init.state[None], rec.arrays["executed_action"])
        f_res[init.state_id] = float(np.abs(bare - rec.arrays["state"]).max())
    reset_to_state(raw, val[0].model_xml, val[0].state)
    acts = rec.arrays["executed_action"]
    base, pert = [], []
    for a in acts[:150]:
        raw.step(a)
        base.append(np.array(raw.sim.get_state().flatten()))
    reset_to_state(raw, val[0].model_xml, val[0].state)
    for a in acts[:150]:
        raw.step(a)
        mujoco.mj_forward(raw.sim.model._model, raw.sim.data._data)
        pert.append(np.array(raw.sim.get_state().flatten()))
    rep["F_purity_adapter_vs_bare_max_abs_state_diff"] = f_res
    rep["F_reference_mj_forward_after_every_step_max_abs_state_diff_150_steps"] = float(
        np.abs(np.asarray(base) - np.asarray(pert)).max())
    pol = RandomPolicy(env.action_spec, 3)
    first = run_episode(env, pol, val[0])
    run_episode(env, pol, val[1])
    with h5py.File(entry.destination(), "r") as f:
        g = f["data"]["demo_1"]
        run_episode(env, DemoReplayPolicy(g["actions"][()], "demo_1", "-"),
                    from_demo(task, "demo_1", g.attrs["model_file"], g["states"][()],
                              "demo_train", "-"))
    again = run_episode(env, pol, val[0])
    env.close()
    fresh_env = RoboForgeEnv(task)
    fresh = run_episode(fresh_env, pol, val[0])
    fresh_env.close()
    keys = ("state", "obs", "executed_action", "success", "body_pos", "body_quat_wxyz")

    def maxdiff(x: Any, y: Any) -> float:
        return max(float(np.abs(x.arrays[k].astype(np.float64)
                                - y.arrays[k].astype(np.float64)).max()) for k in keys)

    rep["G_isolation_max_abs_diff"] = {
        "same_instance_after_other_episodes": maxdiff(first, again),
        "fresh_instance": maxdiff(first, fresh)}
    d_fields = rep["D_dataset_vs_eval_model_param_diff"]["fields"]
    # world subtree mass includes the markers' mass (no DOFs); site colours are render-only.
    allowed = ("body_subtreemass", "site_rgba")
    rep["checks"] = {
        "A_markers_no_effect_on_demo_dynamics": all(v == 0.0 for v in a_res.values()),
        "B_markers_no_effect_on_closed_loop_dynamics": all(v == 0.0 for v in b_res.values()),
        "C_markers_invisible_in_renders": all(v == 0 for v in c_res.values()),
        "D_only_expected_model_differences": all(
            v["index_field"] or k in allowed for k, v in d_fields.items()),
        "F_recording_and_rendering_do_not_perturb_physics":
            all(v == 0.0 for v in f_res.values()),
        "G_episode_isolation": all(v == 0.0 for v in rep["G_isolation_max_abs_diff"].values())}
    return rep


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tasks", nargs="+", default=["lift", "can"])
    p.add_argument("--out-dir", type=Path, default=OUT)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for task in args.tasks:
        rep = run_task(task)
        (args.out_dir / f"sim_contract_{task}.json").write_text(json.dumps(rep, indent=1) + "\n")
        for name, passed in rep["checks"].items():
            print(f"[{task}] [{'PASS' if passed else 'FAIL'}] {name}")
        print(f"[{task}] mj_forward on live data after every step changes the 150-step rollout by "
              f"{rep['F_reference_mj_forward_after_every_step_max_abs_state_diff_150_steps']:.3g}")
        ok &= all(rep["checks"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
