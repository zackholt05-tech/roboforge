"""Session 8 RGB feasibility: aligned frames from recorded demo states, measured (MASTER_SPEC §7.2).

No training and no RGB dataset: this renders frames of recorded demonstration states and
measures whether they can be trusted and what they cost. Per task it writes, under
``reports/session<N>/rgb_feasibility/<platform_id>/``:

* ``<task>_alignment.json``: raw file vs low-dim file (model XML, states, actions) bitwise for
  every demo; features of the raw states vs the committed processed observations;
* ``<task>_sample.jsonl`` + ``frames/`` + ``<task>_inspection.png``: the fixed inspection
  sample (rule below, declared before any frame was looked at): every camera and size, raw
  PNGs without overlay, one alignment record per frame, the projection check per frame, and
  an inspection montage whose overlays (projected object crosshair, labels) exist only there;
* ``<task>_projection.json``: the projection test on the t = 0 frame of every validation-split
  demo (object unoccluded) with its negative controls;
* ``<task>_invariants.json``: re-render determinism, hidden sites, live-vs-restored equality,
  the first-render-of-a-context check and evaluation-model parity;
* ``<task>_clip.mp4`` + ``<task>_clip.jsonl``: an aligned clip of one demo with its ground-truth
  actions (lowest-index validation demo);
* ``<task>_cost.json``: render throughput, storage per frame by compression, and the
  closed-loop control-step cost with rendering, on this platform.

Sample rule: ``numpy.random.default_rng([20261108, task_index])`` (lift 0, can 1) draws
``--sample-n`` distinct flat frame indices over all demos in numeric order. Nothing here reads
an evaluation suite other than ``validation`` (used for evaluation-model parity and step cost).
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

from roboforge.data.rgb import (  # noqa: E402
    DEFAULT_CAMERAS,
    TASK_OBJECT_POS_FEATURE,
    DemoFrameRenderer,
    iter_raw_demos,
)
from roboforge.data.rgb_checks import (  # noqa: E402
    compare_raw_lowdim,
    eval_model_parity,
    features_match_processed,
    projection_check,
    render_invariants,
)
from roboforge.paths import evidence_reports_dir, repo_relative  # noqa: E402
from roboforge.provenance import environment_snapshot, evaluation_platform, write_json  # noqa: E402

TASK_INDEX = {"lift": 0, "can": 1}


def processed_episodes(task: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash-verified processed episodes and the task manifest."""
    from roboforge.data.pipeline import load_task_artifacts
    from roboforge.data.processed import load_processed, processed_path

    docs = load_task_artifacts(task)
    proc = docs["task_manifest"]["processed"]
    eps = load_processed(processed_path(task, proc["converter_version"]),
                         expected_sha256=proc["content_sha256"])
    return eps, docs


def obs_slice(task: str) -> slice:
    """Slice of the committed observation holding the task object's world position."""
    from roboforge.data.pipeline import load_specs

    spec, _ = load_specs(task)
    lo, hi = spec.slices()[TASK_OBJECT_POS_FEATURE[task]]
    return slice(lo, hi)


def alignment(task: str, eps: dict[str, Any]) -> dict[str, Any]:
    """Raw vs low-dim for every demo; raw-state features vs processed obs for every demo."""
    out = {"raw_vs_low_dim": compare_raw_lowdim(task), "features_vs_processed": {}}
    r = DemoFrameRenderer(task)
    bad = []
    try:
        for d in iter_raw_demos(task):
            r.load_model(d.model_xml, d.states[0])
            res = features_match_processed(r, d.states, eps[d.demo], d.actions)
            if not res["pass"]:
                bad.append({"demo": d.demo, **res})
    finally:
        r.close()
    out["features_vs_processed"] = {"demos": len(eps), "failures": bad, "pass": not bad}
    out["pass"] = out["raw_vs_low_dim"]["pass"] and not bad
    return out


def overlay(frame: np.ndarray, uv: list[float] | None, label: str, scale: int = 3) -> Any:
    """Inspection copy: upscaled frame, crosshair at the projected object, a label band."""
    from PIL import Image, ImageDraw

    h, w, _ = frame.shape
    img = Image.fromarray(frame).resize((w * scale, h * scale), Image.NEAREST)
    canvas = Image.new("RGB", (w * scale, h * scale + 14), (20, 20, 20))
    canvas.paste(img, (0, 14))
    dr = ImageDraw.Draw(canvas)
    dr.text((2, 1), label, fill=(240, 240, 220))
    if uv is not None and np.all(np.isfinite(uv)):
        x, y = uv[0] * scale, uv[1] * scale + 14
        dr.line([(x - 6, y), (x + 6, y)], fill=(0, 255, 255), width=1)
        dr.line([(x, y - 6), (x, y + 6)], fill=(0, 255, 255), width=1)
    return canvas


def sample(task: str, eps: dict[str, Any], n: int, sizes: list[int],
           out: Path) -> dict[str, Any]:
    """The fixed inspection sample: frames, records, projection checks and a montage."""
    from PIL import Image

    names = sorted(eps, key=lambda k: int(k.split("_")[-1]))
    flat = [(d, t) for d in names for t in range(len(eps[d]["obs"]))]
    rng = np.random.default_rng([20261108, TASK_INDEX[task]])
    picks = sorted((flat[i] for i in rng.choice(len(flat), size=n, replace=False)),
                   key=lambda p: (int(p[0].split("_")[-1]), p[1]))
    (out / "frames").mkdir(parents=True, exist_ok=True)
    sl, r, tiles, records = obs_slice(task), DemoFrameRenderer(task), [], []
    try:
        demos = {d.demo: d for d in iter_raw_demos(task, sorted({p[0] for p in picks},
                                                         key=lambda k: int(k.split("_")[-1])))}
        for demo, t in picks:
            d = demos[demo]
            r.load_model(d.model_xml, d.states[0])
            state, pos = d.states[t], eps[demo]["obs"][t, sl].astype(np.float64)
            for size in sizes:
                for ci, cam in enumerate(DEFAULT_CAMERAS):
                    frame = r.rgb(state, cam, size)
                    other = DEFAULT_CAMERAS[1 - ci]
                    proj = projection_check(r, state, cam, size, object_pos=pos,
                                            other_camera=other)
                    png = out / "frames" / f"{task}_{demo}_t{t:03d}_{cam}_{size}.png"
                    Image.fromarray(frame).save(png)
                    rec = {**r.frame_record(demo, t, state, cam, size, frame),
                           "png": repo_relative(png), "action_t": d.actions[t].tolist(),
                           "processed_object_pos": pos.tolist(), "projection": proj}
                    records.append(rec)
                    if size == max(sizes):
                        tiles.append(overlay(frame, proj["projection_uv"],
                                             f"{demo} t={t} {cam[:9]}", scale=2))
    finally:
        r.close()
    (out / f"{task}_sample.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    cols = 2 * 4
    tw, th = tiles[0].size
    sheet = Image.new("RGB", (tw * cols, th * ((len(tiles) + cols - 1) // cols)), (0, 0, 0))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * tw, (i // cols) * th))
    sheet.save(out / f"{task}_inspection.png")
    agent = [x["projection"] for x in records if x["camera"]["name"] == "agentview"]
    return {"rule": "default_rng([20261108, task_index]).choice(n_frames, n, replace=False)",
            "picks": [list(p) for p in picks], "frames": len(records),
            "agentview_on_object": sum(p["on_object"] for p in agent),
            "agentview_frames": len(agent)}


def projection_test(task: str, eps: dict[str, Any], sizes: list[int],
                    split_demos: list[str]) -> dict[str, Any]:
    """t = 0 of every validation-split demo: projection on the object; controls must fail."""
    sl, r, rows = obs_slice(task), DemoFrameRenderer(task), []
    try:
        for d in iter_raw_demos(task, split_demos):
            r.load_model(d.model_xml, d.states[0])
            pos = eps[d.demo]["obs"][0, sl].astype(np.float64)
            for size in sizes:
                for ci, cam in enumerate(DEFAULT_CAMERAS):
                    p = projection_check(r, d.states[0], cam, size, object_pos=pos,
                                         other_camera=DEFAULT_CAMERAS[1 - ci])
                    rows.append({"demo": d.demo, **p})
    finally:
        r.close()
    # The wrist camera does not see the object at t = 0 in some demos: those frames have no
    # visible pixels or project outside the image, and are counted, not tested.
    visible = [x for x in rows if x["area_px"] > 0 and x["projection_in_image"]]
    flip = [x for x in visible if x["control_flipped_applicable"]]
    other = [x for x in visible if x.get("control_other_camera_applicable")]
    summary = {"frames": len(rows), "object_visible": len(visible),
               "on_object": sum(x["on_object"] for x in visible),
               "max_dist_to_mask_px": max(x["dist_to_mask_px"] for x in visible),
               "max_dist_to_centroid_px": max(x["dist_to_centroid_px"] for x in visible),
               "flip_control_frames": len(flip),
               "flip_control_on_object": sum(bool(x["control_flipped_on_object"]) for x in flip),
               "other_camera_control_frames": len(other),
               "other_camera_control_on_object": sum(bool(x["control_other_camera_on_object"])
                                                     for x in other)}
    agent = [x for x in rows if x["camera"] == "agentview"]
    summary["agentview_visible"] = sum(x in visible for x in agent)
    summary["pass"] = (summary["on_object"] == len(visible) > 0
                       and summary["agentview_visible"] == len(agent)
                       and summary["flip_control_on_object"] == 0
                       and summary["other_camera_control_on_object"] == 0)
    return {"summary": summary, "rows": rows}


def first_render_check(task: str, demos: list[str], sizes: list[int]) -> list[dict[str, Any]]:
    """Pixels where the first render of a fresh GL context differs from the second."""
    out = []
    for d in iter_raw_demos(task, demos):
        for size in sizes:
            for cam in DEFAULT_CAMERAS:
                r = DemoFrameRenderer(task)
                try:
                    r.load_model(d.model_xml, d.states[0])
                    f = [r.rgb(d.states[0], cam, size, warm=False) for _ in range(3)]
                finally:
                    r.close()
                out.append({"demo": d.demo, "camera": cam, "size": size,
                            "first_vs_second_px": int(np.any(f[0] != f[1], axis=-1).sum()),
                            "second_vs_third_px": int(np.any(f[1] != f[2], axis=-1).sum())})
    return out


def clip(task: str, demo: str, out: Path, size: int = 128) -> dict[str, Any]:
    """Aligned clip: both cameras, labels with step, state hash and ground-truth action."""
    import imageio.v2 as iio
    from PIL import Image, ImageDraw

    d = next(iter_raw_demos(task, [demo]))
    r, frames, recs = DemoFrameRenderer(task), [], []
    try:
        r.load_model(d.model_xml, d.states[0])
        for t, state in enumerate(d.states):
            pair = []
            for cam in DEFAULT_CAMERAS:
                f = r.rgb(state, cam, size)
                rec = r.frame_record(demo, t, state, cam, size, f)
                rec["action_t"] = d.actions[t].tolist()
                recs.append(rec)
                pair.append(overlay(f, None, cam, scale=2))
            w, h = pair[0].size
            canvas = Image.new("RGB", (2 * w, h + 30), (20, 20, 20))
            canvas.paste(pair[0], (0, 0))
            canvas.paste(pair[1], (w, 0))
            a = " ".join(f"{x:+.2f}" for x in d.actions[t])
            dr = ImageDraw.Draw(canvas)
            dr.text((4, h + 2), f"{task} {demo} t={t:03d} state={recs[-1]['state_sha256'][:12]}",
                    fill=(240, 240, 220))
            dr.text((4, h + 16), f"ground-truth action a[t]: {a}", fill=(255, 200, 120))
            frames.append(np.asarray(canvas))
    finally:
        r.close()
    mp4 = out / f"{task}_clip.mp4"
    iio.mimwrite(mp4, frames, fps=20, macro_block_size=1)
    (out / f"{task}_clip.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
    return {"demo": demo, "frames": len(d.states), "size": size, "mp4": repo_relative(mp4)}


def throughput(task: str, demos: list[str], sizes: list[int]) -> dict[str, Any]:
    """ms per rendered frame (first render of each context excluded), per camera and size."""
    out: dict[str, Any] = {}
    raw = list(iter_raw_demos(task, demos))
    for size in sizes:
        for cam in DEFAULT_CAMERAS:
            r, ms = DemoFrameRenderer(task), []
            try:
                for d in raw:
                    r.load_model(d.model_xml, d.states[0])
                    r.rgb(d.states[0], cam, size)  # warm-up of a new context, not timed
                    for s in d.states:
                        t0 = time.perf_counter()
                        r.rgb(s, cam, size)
                        ms.append(1e3 * (time.perf_counter() - t0))
            finally:
                r.close()
            a = np.array(ms)
            out[f"{cam}_{size}"] = {"frames": int(a.size), "median_ms": float(np.median(a)),
                                    "mean_ms": float(a.mean()),
                                    "p90_ms": float(np.percentile(a, 90)),
                                    "frames_per_s": float(1e3 / a.mean())}
    return out


def storage(task: str, demos: list[str], sizes: list[int], total_frames: int) -> dict[str, Any]:
    """Bytes per frame in HDF5 (chunk = 1 frame) by compression, extrapolated to the task."""
    import h5py

    out: dict[str, Any] = {"demos": demos, "total_frames_task": total_frames}
    raw = list(iter_raw_demos(task, demos))
    for size in sizes:
        r = DemoFrameRenderer(task)
        try:
            imgs = {cam: [] for cam in DEFAULT_CAMERAS}
            for d in raw:
                r.load_model(d.model_xml, d.states[0])
                for s in d.states:
                    for cam in DEFAULT_CAMERAS:
                        imgs[cam].append(r.rgb(s, cam, size))
        finally:
            r.close()
        n = len(imgs[DEFAULT_CAMERAS[0]])
        for comp, opts in (("none", None), ("gzip4", 4), ("lzf", "lzf")):
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "x.hdf5"
                with h5py.File(p, "w") as f:
                    for cam, frames in imgs.items():
                        kw = {} if opts is None else (
                            {"compression": "gzip", "compression_opts": opts}
                            if isinstance(opts, int) else {"compression": opts})
                        f.create_dataset(cam, data=np.stack(frames), chunks=(1, size, size, 3),
                                         **kw)
                nbytes = p.stat().st_size
            per_frame_pair = nbytes / n
            out[f"{size}_{comp}"] = {"frames_sampled": n, "bytes_per_frame_both_cameras":
                                     per_frame_pair,
                                     "projected_task_gib": per_frame_pair * total_frames / 2**30}
    return out


def step_cost(task: str, sizes: list[int], n_steps: int = 60) -> dict[str, Any]:
    """Closed-loop control step (RoboForgeEnv) with and without rendering both cameras."""
    from roboforge.sim.env_adapter import RoboForgeEnv
    from roboforge.sim.eval_states import load_suite

    env = RoboForgeEnv(task)
    rng = np.random.default_rng(0)
    out: dict[str, Any] = {"validation_state": None}
    try:
        init = load_suite(task, "validation")[0]
        out["validation_state"] = init.state_id
        for size in [None, *sizes]:
            env.reset(init)
            if size is not None:
                for cam in DEFAULT_CAMERAS:
                    env.render(cam, size, size)  # warm-up of this episode's contexts
            ms = []
            for _ in range(n_steps):
                t0 = time.perf_counter()
                env.step(rng.uniform(-1, 1, env.action_dim))
                if size is not None:
                    for cam in DEFAULT_CAMERAS:
                        env.render(cam, size, size)
                ms.append(1e3 * (time.perf_counter() - t0))
            out["physics_only" if size is None else f"step_plus_2_cameras_{size}"] = {
                "steps": n_steps, "median_ms": float(np.median(ms)), "mean_ms": float(np.mean(ms))}
    finally:
        env.close()
    return out


def run_task(task: str, args: argparse.Namespace, out: Path) -> dict[str, Any]:
    """Every check and measurement for one task."""
    from roboforge.sim.eval_states import load_suite

    eps, docs = processed_episodes(task)
    split = docs["split"]["splits"]
    valid = sorted(split["valid"], key=lambda k: int(k.split("_")[-1]))
    train = sorted(split["train"], key=lambda k: int(k.split("_")[-1]))
    res: dict[str, Any] = {"task": task}
    t0 = time.perf_counter()
    res["alignment"] = alignment(task, eps)
    write_json(out / f"{task}_alignment.json", res["alignment"])
    res["sample"] = sample(task, eps, args.sample_n, args.sizes, out)
    proj = projection_test(task, eps, args.sizes, valid)
    write_json(out / f"{task}_projection.json", proj)
    res["projection"] = proj["summary"]
    d0 = next(iter_raw_demos(task, [valid[0]]))
    inv = {"render_invariants": [render_invariants(task, d0.model_xml, d0.states, d0.actions,
                                                   cam, size)
                                 for size in args.sizes for cam in DEFAULT_CAMERAS],
           "first_render": first_render_check(task, valid[:3], args.sizes),
           "eval_model_parity": [eval_model_parity(task, d0.model_xml, d0.states[0],
                                                   load_suite(task, "validation")[0], cam, size)
                                 for size in args.sizes for cam in DEFAULT_CAMERAS]}
    write_json(out / f"{task}_invariants.json", inv)
    res["invariants_pass"] = (all(x["pass"] for x in inv["render_invariants"])
                              and all(x["pass"] for x in inv["eval_model_parity"]))
    res["first_render_max_px"] = max(x["first_vs_second_px"] for x in inv["first_render"])
    res["steady_render_max_px"] = max(x["second_vs_third_px"] for x in inv["first_render"])
    if not args.skip_clip:
        res["clip"] = clip(task, valid[0], out)
    total = sum(len(e["obs"]) for e in eps.values())
    cost = {"throughput": throughput(task, valid[:args.storage_demos], args.sizes),
            "storage": storage(task, train[:args.storage_demos], args.sizes, total),
            "step_cost": step_cost(task, args.sizes)}
    write_json(out / f"{task}_cost.json", cost)
    res["cost"] = cost
    res["seconds"] = time.perf_counter() - t0
    return res


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tasks", nargs="+", default=["lift", "can"], choices=["lift", "can"])
    p.add_argument("--sizes", nargs="+", type=int, default=[84, 128])
    p.add_argument("--sample-n", type=int, default=12)
    p.add_argument("--storage-demos", type=int, default=10)
    p.add_argument("--skip-clip", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    plat = evaluation_platform()
    out = args.out_dir or evidence_reports_dir("rgb_feasibility", plat["platform_id"])
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"schema": "roboforge.rgb_feasibility/1", "backend": BACKEND,
                               "platform": plat, "environment": environment_snapshot(),
                               "cameras": list(DEFAULT_CAMERAS), "sizes": args.sizes,
                               "tasks": {}}
    for task in args.tasks:
        res = run_task(task, args, out)
        summary["tasks"][task] = {k: v for k, v in res.items() if k != "cost"}
        print(f"[{task}] alignment {res['alignment']['pass']} projection "
              f"{res['projection']['pass']} invariants {res['invariants_pass']} "
              f"first-render max {res['first_render_max_px']} px "
              f"({res['seconds']:.0f} s)", flush=True)
    summary["all_passed"] = all(t["alignment"]["pass"] and t["projection"]["pass"]
                                and t["invariants_pass"] for t in summary["tasks"].values())
    write_json(out / "summary.json", summary)
    print(f"summary: {repo_relative(out / 'summary.json')} all_passed={summary['all_passed']}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
