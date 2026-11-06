"""Replay sampled demonstrations through RoboForgeEnv + recorder; divergence and frame review.

Sampling rule and gates: ``roboforge.sim.replay_compare`` (DECISIONS D-029). Outputs in
``--out-dir`` (default ``reports/session2/replay``):

* ``replay_<task>.json``: selection, per-demo divergence, gates, provenance;
* ``<task>_montage.png``: dataset-recorded vs replayed frames at 4 timesteps per demo
  (agentview); the top-left label of every tile says which it is;
* ``<task>_divergence.png``: task-object position error vs step for every sampled demo;
* ``<task>_<demo>_recorded_vs_replayed.mp4`` (``--video``): side-by-side clip.

All replays are **non-autonomous** open-loop playback of human actions (engineering check).
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
import numpy as np  # noqa: E402

from roboforge.data.fetch import load_manifest, verify_file  # noqa: E402
from roboforge.data.pipeline import load_task_artifacts  # noqa: E402
from roboforge.paths import REPO_ROOT, repo_relative  # noqa: E402
from roboforge.provenance import environment_snapshot, evaluation_platform  # noqa: E402
from roboforge.sim import replay_compare as rc  # noqa: E402
from roboforge.sim.env_adapter import RoboForgeEnv  # noqa: E402
from roboforge.sim.factory import load_sim_config  # noqa: E402
from roboforge.sim.shadow_render import ShadowRenderer  # noqa: E402
from roboforge.sim.snapshot import restore  # noqa: E402

OUT = REPO_ROOT / "reports" / "session2" / "replay"
CAM, SIZE = "agentview", 128


def _label(img: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 6 * len(text) + 4, 11], fill=(0, 0, 0))
    d.text((2, 0), text, fill=(255, 255, 255))
    return np.asarray(im)


def frames_for(env: RoboForgeEnv, init: Any, rec: Any, states: np.ndarray,
               ts: list[int], size: int) -> list[np.ndarray]:
    """Recorded (dataset) and replayed tiles at ``ts`` on the demo's own compiled model."""
    restore(env.raw_env, init)
    sh = ShadowRenderer(env.model)
    tiles = []
    for t in ts:
        tiles.append(_label(sh.rgb(states[t], CAM, size, size), f"rec t={t}"))
        tiles.append(_label(sh.rgb(rec.arrays["state"][t], CAM, size, size), f"replay t={t}"))
    sh.close()
    return tiles


def write_video(env: RoboForgeEnv, init: Any, rec: Any, states: np.ndarray,
                path: Path) -> None:
    """Side-by-side MP4 (recorded | replayed), 20 fps = control rate, 192 px per view."""
    import imageio.v2 as iio2

    restore(env.raw_env, init)
    sh = ShadowRenderer(env.model)
    with iio2.get_writer(path, fps=20, codec="libx264", quality=8) as w:
        for t in range(len(states)):
            a = _label(sh.rgb(states[t], CAM, 192, 192), "recorded")
            b = _label(sh.rgb(rec.arrays["state"][t], CAM, 192, 192), "replayed")
            w.append_data(np.concatenate([a, b], axis=1))
    sh.close()


def run_task(task: str, out_dir: Path, video: bool) -> dict[str, Any]:
    """Replay the sample for one task and write its report and figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, files = load_manifest()
    entry = next(f for f in files if f.task == task and f.hdf5_type == "low_dim")
    sha = verify_file(entry, entry.destination())
    split = load_task_artifacts(task)["split"]["splits"]
    s0 = {r["demo"]: r for r in json.loads((REPO_ROOT / "reports" / "feasibility" /
                                            f"data_report_{task}.json").read_text())
          ["replay"]["per_demo"]}
    env = RoboForgeEnv(task, hold_steps=10_000)  # replay continues after first success
    task_objects = load_sim_config(task)["task_object_joints"]
    results, tiles, curves = [], [], {}
    with h5py.File(entry.destination(), "r") as f:
        lengths = {d: int(f["data"][d].attrs["num_samples"]) for d in f["data"]}
        sample = rc.select_demos(lengths, split["valid"], task)
        for pick in sample:
            demo = pick["demo"]
            grp = f["data"][demo]
            part = "demo_valid" if demo in split["valid"] else "demo_train"
            summ, rec, init = rc.replay_one(env, task, demo, grp, part, sha, task_objects)
            legacy = s0[demo]["max_errors"]
            summ["reason"] = pick["reason"]
            summ["matches_session0_legacy_metrics"] = (
                summ["object_pos_err_m"]["max"] == legacy["object_pos_max_abs_m"]
                and summ["robot_qpos_err_rad"]["max"] == legacy["robot_qpos_max_abs_rad"])
            summ["session0_replay_success"] = s0[demo]["replay_success_at_end"]
            curves[demo] = summ.pop("object_pos_err_curve_m")
            results.append(summ)
            states = grp["states"][()]
            n = len(states)
            tiles.append(frames_for(env, init, rec, states,
                                    sorted({0, n // 3, 2 * n // 3, n - 1}), SIZE))
            if video and pick["reason"] in ("length bin 0",
                                            "lowest-index Session 0 replay failure"):
                vp = out_dir / f"{task}_{demo}_recorded_vs_replayed.mp4"
                write_video(env, init, rec, states, vp)
                summ["video"] = repo_relative(vp)
    env.close()
    rows = [np.concatenate(r + [np.zeros_like(r[0])] * (8 - len(r)), axis=1) for r in tiles]
    import imageio.v3 as iio

    iio.imwrite(out_dir / f"{task}_montage.png", np.concatenate(rows, axis=0))
    fig, ax = plt.subplots(figsize=(7, 4))
    styles = ["-", "--", ":"]
    for i, (demo, c) in enumerate(curves.items()):  # colour + line style: unique per demo
        ax.plot(np.arange(1, len(c) + 1), np.asarray(c) * 1000, lw=1.2,
                color=plt.cm.tab10(i % 10), linestyle=styles[(i // 10) % 3], label=demo)
    ax.set_xlabel("control step (20 Hz)")
    ax.set_ylabel("task-object position error (mm)")
    ax.set_title(f"{task}: open-loop demo replay vs recorded states (non-autonomous)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_divergence.png", dpi=120)
    plt.close(fig)

    def agg(key: str, sub: str) -> dict[str, float]:
        v = np.array([r[key][sub] for r in results])
        return {"median": float(np.median(v)), "p95": float(np.quantile(v, 0.95)),
                "max": float(v.max())}

    gates = {"all_bitwise_equal_to_bare_robosuite_loop":
             all(r["bitwise_equal_to_bare_robosuite_loop"] for r in results),
             "zero_clip_events": all(r["clip_events"] == 0 for r in results),
             "portable_xml_compiles_identically":
             all(r["portable_xml_compiles_identically"] for r in results),
             "all_recorded_final_states_successful":
             all(r["recorded_final_state_success"] for r in results),
             "at_least_10_demos": len(results) >= 10,
             "at_least_2_valid_demos": sum(r["partition"] == "demo_valid" for r in results) >= 2}
    return {"task": task, "environment": environment_snapshot(),
            "platform": evaluation_platform(), "rule": rc.__doc__.split("**Sampling rule")[1],
            "label": "NON-AUTONOMOUS open-loop playback of recorded human actions",
            "n_demos": len(results), "gates": gates,
            "session0_legacy_metrics_match_all": all(r["matches_session0_legacy_metrics"]
                                                     for r in results),
            "replay_success_count": sum(r["replay_success_final_frame"] for r in results),
            "divergence": {"object_pos_err_max_m": agg("object_pos_err_m", "max"),
                           "object_pos_err_final_m": agg("object_pos_err_m", "final"),
                           "robot_qpos_err_max_rad": agg("robot_qpos_err_rad", "max"),
                           "eef_pos_err_max_m": agg("eef_pos_err_m", "max")},
            "figures": {"montage": repo_relative(out_dir / f"{task}_montage.png"),
                        "divergence": repo_relative(out_dir / f"{task}_divergence.png")},
            "per_demo": results, "object_pos_err_curves_m": curves}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tasks", nargs="+", default=["lift", "can"])
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--video", action="store_true")
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for task in args.tasks:
        rep = run_task(task, args.out_dir, args.video)
        (args.out_dir / f"replay_{task}.json").write_text(json.dumps(rep, indent=1) + "\n")
        print(f"[{task}] {rep['n_demos']} demos; replay success "
              f"{rep['replay_success_count']}/{rep['n_demos']}; legacy match "
              f"{rep['session0_legacy_metrics_match_all']}; object err max p95 "
              f"{rep['divergence']['object_pos_err_max_m']['p95']:.4f} m")
        for g, v in rep["gates"].items():
            print(f"  [{'PASS' if v else 'FAIL'}] {g}")
        ok &= all(rep["gates"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
