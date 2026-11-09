"""Build the point clouds of one task's demonstrations (Session 9, Level 2; D-074 path).

Run on the canonical container (`make pc-dataset`). The container mounts ``data/`` read-only, so
the file is written under ``runs/pc_data/<task>/`` with its manifest. ``--install`` then runs on
the host. It re-verifies the file's content hash, copies the file next to the low-dim file
(``data/processed/robomimic_v1.5_ph/<task>/pc<N>_v1.hdf5``) and writes the committed manifest
``data/manifests/pointcloud/<task>_pc<N>_v1.json``. Installing needs a recorded decision.

Each demo goes through these steps:
1. its raw states are checked against the committed processed observations, bitwise;
2. every state becomes a cloud through :class:`roboforge.data.pointcloud.CloudMaker`. That
   covers depth from both cameras, calibration, the workspace box and farthest-point
   sampling: the closed-loop policy's path;
3. the state hashes are kept for alignment.

**Diagnostics, never data.** Per frame, the builder also counts how many sampled points lie
on the cube. It measures this with the cube's true pose, a simulator label the policy never
sees. The counts go to ``<file>.coverage.json`` beside the manifest, never into the dataset.

No evaluation state is read.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

BACKEND = configure_mujoco_gl()

from roboforge.data.pc_dataset import load_pc_dataset, pc_relpath, write_pc_dataset  # noqa: E402
from roboforge.data.rgb import DemoFrameRenderer, iter_raw_demos, raw_file  # noqa: E402
from roboforge.paths import REPO_ROOT, repo_relative  # noqa: E402
from roboforge.provenance import evaluation_platform, git_revision, sha256_file, write_json  # noqa: E402

MANIFEST_SCHEMA = "roboforge.pc_dataset_manifest/1"


def build(task: str, out: Path, limit: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build, verify and write; returns the manifest (not installed) and the coverage record."""
    import numpy as np

    from roboforge.data import pc_checks
    from roboforge.data.pipeline import load_task_artifacts
    from roboforge.data.pointcloud import CloudMaker
    from roboforge.data.processed import load_processed, processed_path
    from roboforge.data.rgb_checks import features_match_processed
    from roboforge.sim.snapshot import sha256_state
    from roboforge.train.pointcloud import PCConfig, config_path

    spec = PCConfig.load(config_path()).cloud_spec
    docs = load_task_artifacts(task)
    proc = docs["task_manifest"]["processed"]
    eps = load_processed(processed_path(task, proc["converter_version"]),
                         expected_sha256=proc["content_sha256"])
    _, raw_sha = raw_file(task, "raw")
    renderer = DemoFrameRenderer(task)
    points: dict[str, np.ndarray] = {}
    n_crop: dict[str, np.ndarray] = {}
    states: dict[str, list[str]] = {}
    cov: dict[str, dict[str, list[int]]] = {}
    convention = None
    t0 = time.perf_counter()
    t_cloud = 0.0
    for i, demo in enumerate(iter_raw_demos(task)):
        if limit is not None and i >= limit:
            break
        renderer.load_model(demo.model_xml, demo.states[0])
        check = features_match_processed(renderer, demo.states, eps[demo.demo], demo.actions)
        if not check["pass"]:
            raise AssertionError(f"{demo.demo}: raw states do not match the processed obs: "
                                 f"{check}")
        maker = CloudMaker(renderer.model, spec)
        convention = convention or maker.depth.convention
        clouds, counts, on_cube, cube_z = [], [], [], []
        for s in demo.states:
            tc = time.perf_counter()
            c, info = maker.cloud(s)
            t_cloud += time.perf_counter() - tc
            clouds.append(c)
            counts.append(info["n_crop"])
            box = pc_checks.box_pose(maker, s, pc_checks.CUBE_VISUAL_GEOM)  # diagnostic only
            on_cube.append(int(np.sum(np.abs(pc_checks.box_signed_distance(c, box))
                                      <= pc_checks.ON_CUBE_TOL_M)))
            cube_z.append(float(box["pos"][2]))
        maker.close()
        points[demo.demo] = np.stack(clouds)
        n_crop[demo.demo] = np.asarray(counts, dtype=np.int32)
        states[demo.demo] = [sha256_state(s) for s in demo.states]
        cov[demo.demo] = {"on_cube_sampled": on_cube, "n_crop": counts,
                          "cube_z_m": [round(z, 6) for z in cube_z]}
        if (i + 1) % 20 == 0:
            print(f"{task}: {i + 1} demos, {time.perf_counter() - t0:.0f} s", flush=True)
    renderer.close()
    render = {"backend": BACKEND, "depth": "roboforge.data.pointcloud.DepthRenderer",
              "warm_up_renders": 1, "geomgroup": [0, 1, 1, 0, 0, 0], "convention": convention,
              "path": "roboforge.data.pointcloud.CloudMaker", "decision": "D-074 path"}
    digest = write_pc_dataset(out, points, n_crop, states, {
        "task": task, "spec": spec.to_dict(), "spec_sha256": spec.sha256(),
        "raw_sha256": raw_sha, "processed_content_sha256": proc["content_sha256"],
        "render": render})
    n_frames = sum(len(v) for v in states.values())
    oc = np.concatenate([np.asarray(v["on_cube_sampled"]) for v in cov.values()])
    nc = np.concatenate([np.asarray(v["n_crop"]) for v in cov.values()])
    cz = np.concatenate([np.asarray(v["cube_z_m"]) for v in cov.values()])
    lifted = cz > 0.84
    summary = {"frames": int(n_frames), "n_crop_min": int(nc.min()),
               "n_crop_median": float(np.median(nc)), "n_crop_max": int(nc.max()),
               "frames_below_n_points": int(np.sum(nc < spec.n_points)),
               "on_cube_min": int(oc.min()), "on_cube_p5": float(np.percentile(oc, 5)),
               "on_cube_median": float(np.median(oc)), "on_cube_zero_frames": int(np.sum(oc == 0)),
               "lifted_frames": int(lifted.sum()),
               "lifted_on_cube_min": int(oc[lifted].min()) if lifted.any() else None,
               "cloud_seconds_per_frame": t_cloud / max(1, n_frames)}
    manifest = {"schema": MANIFEST_SCHEMA, "task": task, "spec": spec.to_dict(),
                "spec_sha256": spec.sha256(), "content_sha256": digest,
                "file_sha256": sha256_file(out), "bytes": out.stat().st_size,
                "relpath": pc_relpath(task, spec.n_points), "demos": len(points),
                "frames": n_frames, "complete": limit is None, "raw_sha256": raw_sha,
                "processed_content_sha256": proc["content_sha256"], "render": render,
                "coverage_summary_diagnostic": summary,
                "platform": evaluation_platform(), "git": git_revision(),
                "build_seconds": time.perf_counter() - t0}
    coverage = {"schema": "roboforge.pc_coverage_diagnostic/1",
                "note": ("diagnostic only: sampled points within "
                         f"{pc_checks.ON_CUBE_TOL_M} m of the cube's true box surface, "
                         "measured with the cube pose (a simulator label the policy never "
                         "sees)"), "summary": summary, "per_demo": cov}
    return manifest, coverage


def install(manifest_path: Path) -> Path:
    """Host side: re-verify the staged file, copy it into data/ and write the manifest."""
    m = json.loads(manifest_path.read_text())
    if not m["complete"]:
        raise SystemExit("refusing to install a partial (--limit) build")
    staged = manifest_path.with_name(Path(m["relpath"]).name)
    if sha256_file(staged) != m["file_sha256"]:
        raise SystemExit(f"{staged}: file hash differs from its manifest")
    load_pc_dataset(staged, expected_sha256=m["content_sha256"])
    dest = REPO_ROOT / "data" / m["relpath"]
    if dest.exists():
        raise SystemExit(f"{dest} exists; installed point-cloud datasets are never overwritten")
    shutil.copy2(staged, dest)
    if sha256_file(dest) != m["file_sha256"]:
        raise SystemExit(f"{dest}: copy hash mismatch")
    name = Path(m["relpath"]).stem  # pc512_v1
    out = REPO_ROOT / "data" / "manifests" / "pointcloud" / f"{m['task']}_{name}.json"
    write_json(out, m)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (see the module docstring)."""
    from roboforge.train.pointcloud import PCConfig, config_path

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="lift")
    p.add_argument("--limit", type=int, help="first N demos only (smoke; never installed)")
    p.add_argument("--out-dir", type=Path, help="default: runs/pc_data/<task>")
    p.add_argument("--install", type=Path, metavar="MANIFEST",
                   help="host: install a staged build from its manifest")
    args = p.parse_args(argv)
    if args.install:
        print(f"installed; manifest {repo_relative(install(args.install))}")
        return 0
    n = PCConfig.load(config_path()).cloud_spec.n_points
    name = Path(pc_relpath(args.task, n)).name
    out_dir = args.out_dir or REPO_ROOT / "runs" / "pc_data" / args.task
    out = out_dir / (name if args.limit is None else f"smoke{args.limit}_{name}")
    m, cov = build(args.task, out, args.limit)
    write_json(out.with_suffix(".manifest.json"), m)
    write_json(out.with_suffix(".coverage.json"), cov)
    print(json.dumps({k: m[k] for k in ("task", "demos", "frames", "content_sha256",
                                        "file_sha256", "bytes", "build_seconds")}))
    print(json.dumps(m["coverage_summary_diagnostic"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
