"""Export one real simulation clip for a small web diagnostic, with coordinate verification.

The clip is a **non-autonomous** open-loop replay of validation demo ``demo_0`` (Lift), run
through RoboForgeEnv and the recorder. Its motion is a real grasp and lift. Writes to
``reports/session2/web_diagnostic/``:

* ``clip_<task>_<demo>.json.gz``: link hierarchy, joint values, per-frame body poses in the
  MuJoCo and three.js frames, fixed cameras, and verification points with their projected pixels;
* ``ref_t<k>_<camera>.png`` / ``overlay_t<k>_<camera>.png``: MuJoCo reference renders at 3
  timestamps, and the same renders with the projected link skeleton (green) and the projected
  object box (yellow) drawn over them;
* ``verification.json``: forward-kinematics, conversion and render checks with declared
  tolerances;
* ``clip.js`` + ``index.html``: a dependency-free canvas viewer of the exported clip.

``tests/web_contract/check_clip.mjs`` then re-checks the clip with the real three.js library.
"""

from __future__ import annotations

import argparse
import gzip
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
from roboforge.eval.policies import DemoReplayPolicy  # noqa: E402
from roboforge.eval.rollout import run_episode  # noqa: E402
from roboforge.paths import REPO_ROOT, repo_relative  # noqa: E402
from roboforge.provenance import environment_snapshot  # noqa: E402
from roboforge.sim.env_adapter import RoboForgeEnv  # noqa: E402
from roboforge.sim.shadow_render import ShadowRenderer  # noqa: E402
from roboforge.sim.snapshot import from_demo  # noqa: E402
from roboforge.viz import clip_export as ce  # noqa: E402
from roboforge.viz import coords  # noqa: E402

OUT = REPO_ROOT / "reports" / "session2" / "web_diagnostic"
CAMERAS, SIZE = ("frontview", "agentview"), 256
TOL = {"fk_pos_m": 1e-12, "fk_angle_rad": 1e-9, "three_vs_mujoco_projection_px": 1e-9,
       "containment_min": 0.999, "object_t0_bbox_edge_px": 1.0}
TASK_OBJECT = {"lift": "cube_main", "can": "Can_main"}


def _angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Rotation angle between unit quaternions, accurate near 0 (atan2 of q1* q2)."""
    rel = coords.quat_mul(q1 * np.array([1, -1, -1, -1]), q2)
    return float(2 * np.arctan2(np.linalg.norm(rel[1:]), abs(rel[0])))


def _draw(img: np.ndarray, segs: list[tuple[np.ndarray, np.ndarray]],
          color: tuple[int, int, int]) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for a, b in segs:
        d.line([tuple(a), tuple(b)], fill=color, width=1)
    return np.asarray(im)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 - linear export pipeline
    """CLI entrypoint."""
    import imageio.v3 as iio

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="lift", choices=["lift", "can"])
    p.add_argument("--demo", default="demo_0")
    p.add_argument("--out-dir", type=Path, default=OUT)
    args = p.parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    _, files = load_manifest()
    e = next(f for f in files if f.task == args.task and f.hdf5_type == "low_dim")
    with h5py.File(e.destination(), "r") as f:
        g = f["data"][args.demo]
        init = from_demo(args.task, args.demo, g.attrs["model_file"], g["states"][()],
                         "demo_valid", e.sha256)
        actions = g["actions"][()]
    env = RoboForgeEnv(args.task, hold_steps=10_000)
    rec = run_episode(env, DemoReplayPolicy(actions, args.demo, e.sha256), init,
                      run_id=f"s2_clip_{args.task}_{args.demo}")
    ce.require_provenance(rec.meta)
    rec.save(REPO_ROOT / "runs" / rec.meta["run_id"])
    env.reset(init)
    model, kin = env.model, env.kinematics
    hier = [dict(b.__dict__) for b in kin.hierarchy()]
    names = kin.body_names
    states = rec.arrays["state"]
    adr = {j["name"]: (int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                                  j["name"])]),
                       7 if j["type"] == "free" else 1) for b in hier for j in b["joints"]}
    fk_pos = fk_ang = 0.0
    joints_per_frame = []
    for k, s in enumerate(states):
        qmap = {n: s[1 + a:1 + a + w].tolist() for n, (a, w) in adr.items()}
        joints_per_frame.append(qmap)
        fk = coords.forward_kinematics(hier, qmap)
        for i, n in enumerate(names):
            fk_pos = max(fk_pos, float(np.abs(fk[n][0] - rec.arrays["body_pos"][k, i]).max()))
            fk_ang = max(fk_ang, _angle(fk[n][1], rec.arrays["body_quat_wxyz"][k, i]))
    T = len(states) - 1
    vframes = sorted({0, T // 2, T})
    shadow = ShadowRenderer(model)
    data = mujoco.MjData(model)
    obj_geoms = ce.tracked_geoms(model, [TASK_OBJECT[args.task]])
    robot_bodies = [n for n, c in zip(names, kin.body_categories, strict=True) if c == "robot"]
    geoms = obj_geoms + ce.tracked_geoms(model, robot_bodies)
    render_checks, points, proj_err = [], [], 0.0
    parents = {b["name"]: b["parent"] for b in hier}
    for k in vframes:
        s = states[k]
        data.qpos[:] = s[1:1 + model.nq]
        mujoco.mj_kinematics(model, data)
        for cam_name in CAMERAS:
            cam = kin.camera(s, cam_name, SIZE, SIZE)
            rgb = shadow.rgb(s, cam_name, SIZE, SIZE)
            seg = shadow.segmentation(s, cam_name, SIZE, SIZE)
            iio.imwrite(out / f"ref_t{k:03d}_{cam_name}.png", rgb)
            checks = ce.check_geoms_against_render(model, data, seg, cam, geoms)
            render_checks.append({"t": k, "camera": cam_name, "geoms": checks})
            body_pts = np.array([data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)]
                                 for n in robot_bodies])
            obj_pts = ce.geom_vertices(model, data, obj_geoms[0])
            pts = np.concatenate([body_pts, obj_pts])
            px = coords.project_mujoco(pts, cam["pos"], cam["rot_world_from_cam"],
                                       cam["fovy_deg"], SIZE, SIZE)
            cpos = coords.mj_to_three_pos(cam["pos"])
            cq = coords.mj_to_three_quat_xyzw(coords.mat_to_quat(cam["rot_world_from_cam"]))
            px3 = coords.project_three(coords.mj_to_three_pos(pts), cpos, cq, cam["fovy_deg"],
                                       SIZE, SIZE)
            proj_err = max(proj_err, float(np.abs(px - px3).max()))
            points.append({"t": k, "camera": cam_name, "labels": robot_bodies +
                           [f"object_vertex_{i}" for i in range(len(obj_pts))],
                           "mujoco": pts.tolist(), "three": coords.mj_to_three_pos(pts).tolist(),
                           "pixels_mujoco_projection": px.tolist(),
                           "camera_three": {"position": cpos.tolist(), "quaternion_xyzw":
                                            cq.tolist(), "fov_deg": cam["fovy_deg"],
                                            "width": SIZE, "height": SIZE}})
            idx = {n: i for i, n in enumerate(robot_bodies)}
            skel = [(px[idx[n]], px[idx[parents[n]]]) for n in robot_bodies
                    if parents[n] in idx]
            ob = px[len(robot_bodies):]
            box = [(ob[i], ob[j]) for i in range(8) for j in range(i + 1, 8)
                   if bin(i ^ j).count("1") == 1]
            iio.imwrite(out / f"overlay_t{k:03d}_{cam_name}.png",
                        _draw(_draw(rgb, skel, (0, 255, 0)), box, (255, 255, 0)))
    shadow.close()
    env.close()
    obj_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, obj_geoms[0])
    contain = min(c["containment"] for r in render_checks for c in r["geoms"]
                  if "containment" in c)
    t0_edge = max(c["bbox_edge_err_px"] for r in render_checks if r["t"] == 0
                  for c in r["geoms"] if c["geom"] == obj_name)
    ver = {"tolerances": TOL, "fk_max_pos_err_m": fk_pos, "fk_max_angle_err_rad": fk_ang,
           "three_vs_mujoco_projection_max_px": proj_err, "min_containment": contain,
           "object_t0_bbox_edge_err_px": t0_edge, "verification_frames": vframes,
           "cameras": list(CAMERAS), "n_geoms_tracked": len(geoms),
           "render_checks": render_checks}
    ver["checks"] = {"fk_reproduces_mujoco": fk_pos <= TOL["fk_pos_m"]
                     and fk_ang <= TOL["fk_angle_rad"],
                     "three_projection_equals_mujoco":
                         proj_err <= TOL["three_vs_mujoco_projection_px"],
                     "render_masks_inside_projection": contain >= TOL["containment_min"],
                     "object_t0_bbox_matches_render":
                         t0_edge <= TOL["object_t0_bbox_edge_px"],
                     "at_least_3_timestamps": len(vframes) >= 3}
    clip = {"schema": ce.CLIP_SCHEMA, "project": "RoboForge", "task": args.task,
            "label": rec.meta["policy"]["label"], "policy": rec.meta["policy"],
            "run_id": rec.meta["run_id"], "physics_sha256": rec.meta["physics_sha256"],
            "initial_state": rec.meta["episode"]["initial_state"],
            "platform": rec.meta["platform"], "git": rec.meta["git"],
            "simulator": rec.meta["simulator"], "outcome": rec.outcome,
            "conversion": {"position": "three = (x, z, -y) of MuJoCo world (x, y, z)",
                           "basis_C": coords.C.tolist(), "q_C_wxyz": coords.Q_C_WXYZ.tolist(),
                           "quaternion": "three (x,y,z,w) = reorder(q_C * q_mujoco_wxyz)",
                           "root_group_equivalent": "rotation.x = -pi/2, MuJoCo values inside"},
            "frame_convention": rec.meta["frame_convention"],
            "bodies": {"names": names, "categories": kin.body_categories, "hierarchy": hier},
            "frames": {"time_s": rec.arrays["time_s"].tolist(),
                       "body_pos_mujoco": rec.arrays["body_pos"].tolist(),
                       "body_quat_wxyz_mujoco": rec.arrays["body_quat_wxyz"].tolist(),
                       "body_pos_three": coords.mj_to_three_pos(rec.arrays["body_pos"]).tolist(),
                       "body_quat_xyzw_three":
                           coords.mj_to_three_quat_xyzw(rec.arrays["body_quat_wxyz"]).tolist(),
                       "success": rec.arrays["success"].tolist(),
                       "joint_qpos": joints_per_frame},
            "verification_points": points,
            "reference_images": {f"t{k}_{c}": f"ref_t{k:03d}_{c}.png" for k in vframes
                                 for c in CAMERAS}}
    path = out / f"clip_{args.task}_{args.demo}.json.gz"
    with path.open("wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
        gz.write(json.dumps(clip, sort_keys=True).encode())
    ver.update(clip=repo_relative(path), environment=environment_snapshot())
    (out / "verification.json").write_text(json.dumps(ver, indent=1) + "\n")
    write_viewer(out, clip, vframes)
    for name, ok in ver["checks"].items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"fk {fk_pos:.2e} m / {fk_ang:.2e} rad; three-vs-mujoco {proj_err:.2e} px; "
          f"containment {contain:.4f}; object t0 edge {t0_edge:.3f} px -> {repo_relative(path)}")
    return 0 if all(ver["checks"].values()) else 1


def write_viewer(out: Path, clip: dict[str, Any], vframes: list[int]) -> None:
    """Dependency-free canvas viewer: projected skeleton per frame + reference renders."""
    cam = next(p["camera_three"] for p in clip["verification_points"]
               if p["camera"] == "frontview")
    small = {"label": clip["label"], "run_id": clip["run_id"], "camera": cam,
             "names": clip["bodies"]["names"], "categories": clip["bodies"]["categories"],
             "parents": [b["parent"] for b in clip["bodies"]["hierarchy"]],
             "pos": np.round(np.asarray(clip["frames"]["body_pos_three"]), 6).tolist(),
             "success": clip["frames"]["success"], "refs": vframes,
             "physics_sha256": clip["physics_sha256"]}
    (out / "clip.js").write_text("window.ROBOFORGE_CLIP = " + json.dumps(small) + ";\n")
    (out / "index.html").write_text(VIEWER_HTML)


VIEWER_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>RoboForge web diagnostic: exported clip</title>
<style>body{font:14px system-ui;background:#1d1d1f;color:#f5f0e6;margin:16px}
canvas{background:#2b2b2e}img{width:256px;height:256px}label{margin-right:12px}</style></head>
<body><h1>Exported clip (diagnostic)</h1>
<p id="label"></p><p>Projected body origins and parent links of the robot (green) and of
the object (yellow). The points use the three.js-frame data and the frontview camera, with
three.js projection math re-implemented in JS. The MuJoCo reference renders below are the
ground truth.</p>
<canvas id="c" width="256" height="256" aria-label="projected skeleton of the exported clip">
</canvas><br><label>frame <input id="s" type="range" min="0" value="0"></label>
<span id="info"></span><div id="refs"></div>
<script src="clip.js"></script><script>
const C = window.ROBOFORGE_CLIP, cv = document.getElementById("c"), g = cv.getContext("2d");
document.getElementById("label").textContent = C.label + " | run " + C.run_id +
  " | physics_sha256 " + C.physics_sha256.slice(0, 12);
const cam = C.camera, [qx, qy, qz, qw] = cam.quaternion_xyzw;
const R = [[1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
           [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
           [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]];
const f = 1 / Math.tan(cam.fov_deg * Math.PI / 360);
function proj(p) { const d = [0, 1, 2].map(i => p[i] - cam.position[i]);
  const c = [0, 1, 2].map(j => d[0]*R[0][j] + d[1]*R[1][j] + d[2]*R[2][j]);
  return [(f * cam.height / cam.width * c[0] / -c[2] + 1) / 2 * cam.width,
          (1 - f * c[1] / -c[2]) / 2 * cam.height]; }
function draw(k) { g.clearRect(0, 0, 256, 256); const P = C.pos[k].map(proj);
  C.names.forEach((n, i) => { const pi = C.parents[i] ? C.names.indexOf(C.parents[i]) : -1;
    g.strokeStyle = C.categories[i] === "robot" ? "#3ddc84" : "#ffd60a";
    g.fillStyle = g.strokeStyle; g.fillRect(P[i][0] - 1.5, P[i][1] - 1.5, 3, 3);
    if (pi >= 0) { g.beginPath(); g.moveTo(...P[i]); g.lineTo(...P[pi]); g.stroke(); } });
  document.getElementById("info").textContent = "frame " + k + " / " + (C.pos.length - 1) +
    (C.success[k] ? " | success predicate true" : ""); }
const s = document.getElementById("s"); s.max = C.pos.length - 1;
s.oninput = () => draw(+s.value); draw(0);
document.getElementById("refs").innerHTML = C.refs.map(k =>
  `<figure style="display:inline-block"><img src="ref_t${String(k).padStart(3, "0")}` +
  `_frontview.png" alt="MuJoCo frontview render at frame ${k}"><figcaption>MuJoCo ` +
  `render, frame ${k}</figcaption></figure>`).join("");
</script></body></html>
"""


if __name__ == "__main__":
    sys.exit(main())
