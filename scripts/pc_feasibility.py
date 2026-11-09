"""Session 9 Level 2 feasibility: depth, calibration and clouds of recorded states (no training).

MASTER_SPEC §7.3 and the Session 9 prompt, step 2. Under
``reports/session<N>/pc_feasibility/<platform_id>/`` it writes:
* ``lift_calibration.json``:
  * the depth convention (near/far, buffer);
  * rendered z-depth against exact ``mj_ray`` casts through every pixel centre;
  * the known-point test (the cube's top-face centre → pixel → back-projected point);
  * the cube-surface and table-plane residuals of back-projected labelled pixels.
* ``lift_invariants.json``:
  * re-render determinism;
  * live-versus-restored equality of clouds;
  * evaluation-start model parity inside the workspace box.
* ``lift_coverage.json``:
  * points inside the box per frame, and sampled points on the cube;
  * over every frame of every demo (``--coverage-stride``), the 50 validation starts, and the
    ``robustness_base`` and ``ood_cube_shift`` starts (t = 0, their own models).
  The OOD starts are used only for geometry: no policy runs. Segmentation and the cube pose
  are diagnostics only.
* ``lift_cost.json``: time per cloud, farthest-point sampling alone, and the dataset size
  (raw bytes, plus gzip-4 measured on a sample).
* ``lift_overlay_<demo>_t<k>.png``: the sampled cloud projected onto frontview and agentview
  RGB at 256², plus a 3-D scatter with the cube's true box as a diagnostic wireframe.
* ``lift_cloud_hashes.json``: cloud SHA-256 of the calibration sample, for a two-process
  comparison.

**Sample rule** (declared before any output was seen):
* calibration uses demos at positions 0, 20, …, 180 of the numeric demo order, frames 0, T // 2
  and T − 1;
* overlays use the lowest-index validation-split demo, frames 0, T // 2 and T − 1.

No held-out state is read: only ``validation``, ``robustness_base`` and ``ood_cube_shift``
(KI-039, KI-040).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

BACKEND = configure_mujoco_gl()

import numpy as np  # noqa: E402

from roboforge.data import pc_checks  # noqa: E402
from roboforge.data.pointcloud import (  # noqa: E402
    CloudMaker,
    cloud_sha256,
    depth_to_world,
    farthest_point_sample,
    in_box,
)
from roboforge.data.rgb import DemoFrameRenderer, iter_raw_demos, project_world_point  # noqa: E402
from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative  # noqa: E402
from roboforge.provenance import environment_snapshot, evaluation_platform, write_json  # noqa: E402

TASK = "lift"
SOURCE_COLOURS = {0: (255, 140, 0), 1: (0, 200, 255)}  # agentview orange, wrist cyan


def spec_from_config() -> Any:
    """The point-cloud spec of the committed Level 2 config."""
    from roboforge.train.pointcloud import PCConfig, config_path

    return PCConfig.load(config_path()).cloud_spec


def demo_names() -> list[str]:
    """All demos of the raw file in numeric order."""
    import h5py

    from roboforge.data.inspect import demo_keys
    from roboforge.data.rgb import raw_file

    path, _ = raw_file(TASK, "raw")
    with h5py.File(path, "r") as f:
        return demo_keys(f)


def valid_demos() -> list[str]:
    """Validation-split demos, numeric order."""
    split = json.loads((REPO_ROOT / "data" / "splits" / f"{TASK}_ph.json").read_text())
    names = split["splits"]["valid"]
    return sorted(names, key=lambda d: int(d.split("_")[1]))


def _agg(values: list[float | None]) -> float | None:
    v = [x for x in values if x is not None]
    return float(max(v)) if v else None


def calibration(spec: Any, demos: list[str]) -> tuple[dict[str, Any], dict[str, str]]:
    """Depth vs rays, known point, cube surface and table plane on the calibration sample."""
    r = DemoFrameRenderer(TASK)
    rays, known, cube, table, hashes = [], [], [], [], {}
    convention = None
    for demo in iter_raw_demos(TASK, demos):
        r.load_model(demo.model_xml, demo.states[0])
        mk = CloudMaker(r.model, spec)
        convention = convention or mk.depth.convention
        n = len(demo.states)
        for t in sorted({0, n // 2, n - 1}):
            s = demo.states[t]
            for cam in spec.cameras:
                rays.append({"demo": demo.demo, "t": t, **pc_checks.depth_vs_ray(mk, s, cam)})
                for size in (spec.size, 256):
                    known.append({"demo": demo.demo, "t": t,
                                  **pc_checks.known_point_check(mk, s, cam, size)})
            cube.append({"demo": demo.demo, "t": t,
                         **pc_checks.cube_surface_residual(mk, s, spec.size)})
            table.append({"demo": demo.demo, "t": t,
                          **pc_checks.table_plane_residual(mk, s, spec.size)})
            hashes[f"{demo.demo}/{t}"] = cloud_sha256(mk.cloud(s)[0])
        mk.close()
    r.close()
    app = [k for k in known if k["applicable"]]
    summary = {
        "frames": len(cube), "ray_comparisons": len(rays),
        "interior_abs_err_max_m": _agg([x["interior_abs_err_max_m"] for x in rays]),
        "interior_abs_err_p99_max_m": _agg([x["interior_abs_err_p99_m"] for x in rays]),
        "interior_pixels_over_1mm": int(sum(x["interior_pixels_over_1mm"] for x in rays)),
        "interior_pixels": int(sum(x["interior_pixels"] for x in rays)),
        "boundary_pixels": int(sum(x["boundary_pixels"] for x in rays)),
        "boundary_pixels_over_1mm": int(sum(x["boundary_pixels_over_1mm"] for x in rays)),
        "known_point_applicable": len(app), "known_point_total": len(known),
        "known_point_plane_residual_max_m": _agg([k["plane_residual_m"] for k in app]),
        "known_point_reproj_err_max_px": _agg([k["reproj_err_px"] for k in app]),
        "known_point_all_within_footprint": all(k["within_footprint"] for k in app),
        "known_point_dist_max_m_by_size": {
            str(sz): _agg([k["dist_to_known_m"] for k in app if k["size"] == sz])
            for sz in (spec.size, 256)},
        "cube_surface_abs_max_m": _agg([c[cam]["abs_max_m"] for c in cube
                                        for cam in spec.cameras]),
        "table_plane_abs_max_m": _agg([x[cam]["abs_max_m"] for x in table
                                       for cam in spec.cameras])}
    summary["pass"] = bool(summary["interior_pixels_over_1mm"] == 0 and app
                           and summary["known_point_all_within_footprint"]
                           and summary["known_point_plane_residual_max_m"] < 1e-4
                           and summary["cube_surface_abs_max_m"] < 1e-4
                           and summary["table_plane_abs_max_m"] < 1e-4)
    return {"convention": convention, "summary": summary, "depth_vs_ray": rays,
            "known_point": known, "cube_surface": cube, "table_plane": table}, hashes


def _restored_maker(init: Any, spec: Any) -> tuple[Any, CloudMaker]:
    from roboforge.sim.factory import load_sim_config, make_env
    from roboforge.sim.snapshot import restore

    env = make_env(load_sim_config(TASK))
    restore(env, init)
    return env, CloudMaker(env.sim.model._model, spec)


def invariants(spec: Any, demo_name: str) -> dict[str, Any]:
    """Re-render, live vs restored, and evaluation-model parity inside the box."""
    from roboforge.sim.eval_states import load_suite

    demo = next(iter_raw_demos(TASK, [demo_name]))
    a = DemoFrameRenderer(TASK)
    a.load_model(demo.model_xml, demo.states[0])
    mk = CloudMaker(a.model, spec)
    c1, _ = mk.cloud(demo.states[0])
    c2, _ = mk.cloud(demo.states[0])
    for act in demo.actions[:20]:
        a.env.step(np.asarray(act, dtype=np.float64))
    live_state = np.array(a.env.sim.get_state().flatten())
    live_cloud, _ = mk.cloud(live_state)
    mk.close()
    a.close()
    b = DemoFrameRenderer(TASK)
    b.load_model(demo.model_xml, demo.states[0])
    mk_b = CloudMaker(b.model, spec)
    fresh_cloud, _ = mk_b.cloud(live_state)
    demo_depths = mk_b.depths(demo.states[0])
    demo_cams = mk_b.cameras(demo.states[0])
    cube_ids = pc_checks.body_geom_ids(b.model, "cube_main")
    demo_cube = {c: pc_checks.object_mask(pc_checks.maker_segmentation(
        mk_b, demo.states[0], c, spec.size), cube_ids) for c in spec.cameras}
    mk_b.close()
    b.close()
    # Evaluation-start model (validation start 0): robosuite's default model with the KI-014
    # markers and its own cube size. Compare the demo state's in-box depth pixels.
    init = load_suite(TASK, "validation")[0]
    env, mk_e = _restored_maker(init, spec)
    parity: dict[str, Any] = {"eval_state_id": init.state_id}
    ok = True
    try:
        for cam in spec.cameras:
            de = mk_e.depth.depth(demo.states[0], cam, spec.size)
            ids = pc_checks.body_geom_ids(mk_e.model, "cube_main")
            m_eval = pc_checks.object_mask(pc_checks.maker_segmentation(
                mk_e, demo.states[0], cam, spec.size), ids)
            w, valid = depth_to_world(demo_cams[cam], demo_depths[cam],
                                      mk_e.depth.convention["far_m"])
            box = (valid & in_box(w, spec)).reshape(spec.size, spec.size)
            differ = np.abs(de - demo_depths[cam]) > 1e-6
            near_cube = pc_checks.dilate_mask(demo_cube[cam] | m_eval, 2)
            outside = differ & box & ~near_cube
            parity[cam] = {"in_box_pixels": int(box.sum()),
                           "in_box_differing": int((differ & box).sum()),
                           "in_box_differing_outside_cube_2px": int(outside.sum())}
            ok = ok and not outside.any()
    finally:
        mk_e.close()
        env.close()
    a2 = DemoFrameRenderer(TASK)
    a2.load_model(demo.model_xml, demo.states[0])
    mk_s = CloudMaker(a2.model, spec)
    sites = [pc_checks.sites_in_depth(mk_s, demo.states[0], c, spec.size)
             for c in spec.cameras]
    mk_s.close()
    a2.close()
    out = {"demo": demo_name, "rerender_bitwise_equal": bool(np.array_equal(c1, c2)),
           "sites_left_on_would_change": sites,
           "live_vs_restored_bitwise_equal": bool(np.array_equal(live_cloud, fresh_cloud)),
           "eval_model_parity": parity, "eval_model_parity_pass": ok}
    out["pass"] = bool(out["rerender_bitwise_equal"] and out["live_vs_restored_bitwise_equal"]
                       and ok)
    return out


def coverage(spec: Any, stride: int) -> dict[str, Any]:
    """Box and cube coverage on demo frames and on validation / robustness starts at t = 0."""
    from roboforge.sim.eval_states import load_suite

    r = DemoFrameRenderer(TASK)
    on, nc, cz = [], [], []
    for demo in iter_raw_demos(TASK):
        r.load_model(demo.model_xml, demo.states[0])
        mk = CloudMaker(r.model, spec)
        for s in demo.states[::stride]:
            c = pc_checks.cube_coverage(mk, s)
            on.append(c["on_cube_sampled"])
            nc.append(c["n_crop"])
            cz.append(c["cube_z_m"])
        mk.close()
    r.close()
    on_a, nc_a, cz_a = np.array(on), np.array(nc), np.array(cz)
    lifted = cz_a > 0.84
    out: dict[str, Any] = {"demo_frames": {
        "stride": stride, "frames": int(len(on_a)), "n_crop_min": int(nc_a.min()),
        "n_crop_median": float(np.median(nc_a)), "frames_below_n_points":
            int(np.sum(nc_a < spec.n_points)), "on_cube_min": int(on_a.min()),
        "on_cube_p5": float(np.percentile(on_a, 5)), "on_cube_median": float(np.median(on_a)),
        "on_cube_zero_frames": int(np.sum(on_a == 0)), "lifted_frames": int(lifted.sum()),
        "lifted_on_cube_min": int(on_a[lifted].min()) if lifted.any() else None}}
    for suite in ("validation", "robustness_base", "ood_cube_shift"):
        rows = []
        for init in load_suite(TASK, suite):
            env, mk = _restored_maker(init, spec)
            try:
                c = pc_checks.cube_coverage(mk, init.state)
                box = pc_checks.box_pose(mk, init.state, pc_checks.CUBE_VISUAL_GEOM)
                corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                                    for sz in (-1, 1)]) * box["half"] @ box["rot"].T + box["pos"]
                lo, hi = np.asarray(spec.box_lo), np.asarray(spec.box_hi)
                # The cube is inside the box above the table if every corner above the
                # crop's z floor lies inside the box.
                above = corners[:, 2] >= lo[2]
                inside = bool(np.all((corners[above] >= lo) & (corners[above] <= hi)))
                rows.append({"state_id": init.state_id, "on_cube_sampled": c["on_cube_sampled"],
                             "n_crop": c["n_crop"], "cube_xy": [float(x) for x in box["pos"][:2]],
                             "cube_inside_box": inside})
            finally:
                mk.close()
                env.close()
        o = np.array([x["on_cube_sampled"] for x in rows])
        out[suite] = {"n": len(rows), "on_cube_min": int(o.min()),
                      "on_cube_median": float(np.median(o)),
                      "n_crop_min": int(min(x["n_crop"] for x in rows)),
                      "all_cubes_inside_box": all(x["cube_inside_box"] for x in rows),
                      "max_cube_radius_m": float(max(np.hypot(*x["cube_xy"]) for x in rows)),
                      "states": rows}
    out["pass"] = bool(out["demo_frames"]["on_cube_zero_frames"] == 0
                       and out["demo_frames"]["frames_below_n_points"] == 0
                       and all(out[s]["on_cube_min"] > 0 and out[s]["all_cubes_inside_box"]
                               for s in ("validation", "robustness_base", "ood_cube_shift")))
    return out


def cost(spec: Any, demos: list[str], total_frames: int) -> dict[str, Any]:
    """Seconds per cloud and per FPS call; dataset bytes, raw and gzip-4 on a sample."""
    import h5py

    r = DemoFrameRenderer(TASK)
    t_cloud, t_fps, t_depth, n = 0.0, 0.0, 0.0, 0
    clouds = {}
    for demo in iter_raw_demos(TASK, demos):
        r.load_model(demo.model_xml, demo.states[0])
        mk = CloudMaker(r.model, spec)
        mk.cloud(demo.states[0])  # contexts exist and are warm before timing
        cs = []
        for s in demo.states:
            t0 = time.perf_counter()
            depths = mk.depths(s)
            t1 = time.perf_counter()
            cams = mk.cameras(s)
            pts = np.concatenate([w[v & in_box(w, spec)] for w, v in (
                depth_to_world(cams[c], depths[c], mk.depth.convention["far_m"])
                for c in spec.cameras)])
            t2 = time.perf_counter()
            farthest_point_sample(pts, spec.n_points)
            t3 = time.perf_counter()
            c, _ = mk.cloud(s)
            t_depth += t1 - t0
            t_fps += t3 - t2
            t_cloud += (t3 - t0)
            n += 1
            cs.append(c)
        clouds[demo.demo] = np.stack(cs)
        mk.close()
    r.close()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "sample.hdf5"
        with h5py.File(p, "w") as f:
            for d, a in clouds.items():
                f.create_dataset(d, data=a, chunks=(1, *a.shape[1:]), compression="gzip",
                                 compression_opts=4)
        gz = p.stat().st_size
    raw_per_frame = spec.n_points * 3 * 4
    gz_per_frame = gz / n
    return {"frames_timed": n, "seconds_per_cloud": t_cloud / n,
            "seconds_per_two_depth_renders": t_depth / n, "seconds_per_fps": t_fps / n,
            "bytes_per_frame_raw": raw_per_frame, "bytes_per_frame_gzip4": gz_per_frame,
            "dataset_frames": total_frames,
            "dataset_bytes_raw": raw_per_frame * total_frames,
            "dataset_bytes_gzip4_estimate": gz_per_frame * total_frames,
            "dataset_build_seconds_estimate": t_cloud / n * total_frames,
            "note": "the build time and the gzip size are estimates scaled from the sample"}


def _draw(frame: np.ndarray, cam: dict[str, Any], cloud: np.ndarray, src: np.ndarray,
          scale: int) -> Any:
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame).resize((frame.shape[1] * scale, frame.shape[0] * scale),
                                        Image.NEAREST)
    d = ImageDraw.Draw(img)
    for p, s in zip(cloud, src, strict=True):
        u, v, z = project_world_point(cam, p)
        if z > 0 and np.isfinite(u):
            x, y = u * scale, v * scale
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=SOURCE_COLOURS[int(s)])
    return img


def overlays(spec: Any, demo_name: str, out: Path) -> list[str]:
    """Clouds over frontview / agentview RGB (256²) and a 3-D scatter, per declared frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    demo = next(iter_raw_demos(TASK, [demo_name]))
    r = DemoFrameRenderer(TASK)
    r.load_model(demo.model_xml, demo.states[0])
    mk = CloudMaker(r.model, spec)
    files = []
    n = len(demo.states)
    for t in sorted({0, n // 2, n - 1}):
        s = demo.states[t]
        cloud, info = mk.cloud(s)
        src = info["source_camera"]
        panels = []
        for cam in ("frontview", "agentview"):
            frame = r.rgb(s, cam, 256)
            panels.append(_draw(frame, r.camera(s, cam, 256), cloud, src, 2))
        w = sum(p.width for p in panels)
        canvas = Image.new("RGB", (w, panels[0].height + 24), (20, 20, 20))
        x = 0
        for p, cam in zip(panels, ("frontview", "agentview"), strict=True):
            canvas.paste(p, (x, 24))
            ImageDraw.Draw(canvas).text((x + 6, 6), f"{demo_name} t={t} {cam} 256^2: "
                                        f"{spec.n_points} sampled points (orange agentview, "
                                        f"cyan wrist)", fill=(240, 240, 240))
            x += p.width
        f1 = out / f"lift_overlay_{demo_name}_t{t:03d}.png"
        canvas.save(f1)
        files.append(repo_relative(f1))
        box = pc_checks.box_pose(mk, s, pc_checks.CUBE_VISUAL_GEOM)
        fig = plt.figure(figsize=(6, 5), dpi=110)
        ax = fig.add_subplot(projection="3d")
        for k, colour in ((0, "tab:orange"), (1, "tab:cyan")):
            m = src == k
            ax.scatter(cloud[m, 0], cloud[m, 1], cloud[m, 2], s=3, c=colour,
                       label=spec.cameras[k])
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                            for sz in (-1, 1)]) * box["half"] @ box["rot"].T + box["pos"]
        for i in range(8):
            for j in range(i + 1, 8):
                if bin(i ^ j).count("1") == 1:
                    ax.plot(*zip(corners[i], corners[j], strict=True), c="k", lw=0.8)
        ax.set_xlim(spec.box_lo[0], spec.box_hi[0])
        ax.set_ylim(spec.box_lo[1], spec.box_hi[1])
        ax.set_zlim(spec.box_lo[2], spec.box_hi[2])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(f"{demo_name} t={t}: sampled cloud (world); black = true cube box "
                     "(diagnostic)", fontsize=8)
        ax.legend(fontsize=7)
        f2 = out / f"lift_scatter_{demo_name}_t{t:03d}.png"
        fig.savefig(f2)
        plt.close(fig)
        files.append(repo_relative(f2))
    mk.close()
    r.close()
    return files


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (see the module docstring)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--coverage-stride", type=int, default=1)
    p.add_argument("--cost-demos", type=int, default=3)
    p.add_argument("--out-dir", type=Path)
    args = p.parse_args(argv)
    plat = evaluation_platform()
    out = args.out_dir or evidence_reports_dir("pc_feasibility", plat["platform_id"])
    out.mkdir(parents=True, exist_ok=True)
    spec = spec_from_config()
    names = demo_names()
    sample = [names[i] for i in range(0, len(names), 20)]
    vdemo = valid_demos()[0]
    t0 = time.perf_counter()
    cal, hashes = calibration(spec, sample)
    write_json(out / "lift_calibration.json", cal)
    write_json(out / "lift_cloud_hashes.json", hashes)
    inv = invariants(spec, vdemo)
    write_json(out / "lift_invariants.json", inv)
    cov = coverage(spec, args.coverage_stride)
    write_json(out / "lift_coverage.json", cov)
    total = sum(len(d.states) for d in iter_raw_demos(TASK))
    cst = cost(spec, names[:args.cost_demos], total)
    write_json(out / "lift_cost.json", cst)
    figs = overlays(spec, vdemo, out)
    summary = {"schema": "roboforge.pc_feasibility/1", "task": TASK, "backend": BACKEND,
               "platform": plat, "environment": environment_snapshot(),
               "spec": spec.to_dict(), "spec_sha256": spec.sha256(),
               "calibration": cal["summary"], "convention": cal["convention"],
               "invariants_pass": inv["pass"], "coverage": {
                   k: ({kk: vv for kk, vv in v.items() if kk != "states"}
                       if isinstance(v, dict) else v) for k, v in cov.items()},
               "cost": cst, "overlays": figs, "sample_demos": sample, "overlay_demo": vdemo,
               "privilege": ("sensor-like: depth of two cameras + their calibration (wrist "
                             "pose from robot kinematics) + a fixed workspace box; no "
                             "segmentation, object pose or simulator label enters a cloud"),
               "wall_s": time.perf_counter() - t0}
    summary["pass"] = bool(cal["summary"]["pass"] and inv["pass"] and cov["pass"])
    write_json(out / "summary.json", summary)
    print(json.dumps({"pass": summary["pass"], "calibration": cal["summary"]["pass"],
                      "invariants": inv["pass"], "coverage": cov["pass"],
                      "out": repo_relative(out), "wall_s": round(summary["wall_s"], 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
