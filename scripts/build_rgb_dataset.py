"""Build the RGB frames of one task's demonstrations (Session 8 RGB study; D-074, D-079).

Run on the canonical container (`make rgb-dataset`), which mounts ``data/`` read-only, so the
file is written under ``runs/rgb_data/<task>/`` with its manifest. ``--install`` then runs on the
host. It re-verifies the file's content hash, copies the file next to the low-dim file
(``data/processed/robomimic_v1.5_ph/<task>/rgb<size>_v1.hdf5``) and writes the committed
manifest ``data/manifests/rgb/<task>_rgb<size>_v1.json``.

Each demo goes through these steps:
1. its raw states are checked against the committed processed observations, bitwise
   (``features_match_processed``);
2. every state is rendered from both cameras through ``DemoFrameRenderer``, with one discarded
   warm-up render per GL context;
3. the state hashes are kept for alignment.

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

from roboforge.data.rgb import DEFAULT_CAMERAS, DemoFrameRenderer, iter_raw_demos, raw_file  # noqa: E402
from roboforge.data.rgb_dataset import load_rgb_dataset, rgb_relpath, write_rgb_dataset  # noqa: E402
from roboforge.paths import REPO_ROOT, repo_relative  # noqa: E402
from roboforge.provenance import evaluation_platform, git_revision, sha256_file, write_json  # noqa: E402

MANIFEST_SCHEMA = "roboforge.rgb_dataset_manifest/1"


def build(task: str, size: int, out: Path, limit: int | None) -> dict[str, Any]:
    """Render, verify and write; returns the manifest (not yet installed)."""
    import numpy as np

    from roboforge.data.pipeline import load_task_artifacts
    from roboforge.data.processed import load_processed, processed_path
    from roboforge.data.rgb_checks import features_match_processed
    from roboforge.sim.snapshot import sha256_state

    docs = load_task_artifacts(task)
    proc = docs["task_manifest"]["processed"]
    eps = load_processed(processed_path(task, proc["converter_version"]),
                         expected_sha256=proc["content_sha256"])
    _, raw_sha = raw_file(task, "raw")
    renderer = DemoFrameRenderer(task)
    frames: dict[str, dict[str, np.ndarray]] = {}
    states: dict[str, list[str]] = {}
    t0 = time.perf_counter()
    for i, demo in enumerate(iter_raw_demos(task)):
        if limit is not None and i >= limit:
            break
        renderer.load_model(demo.model_xml, demo.states[0])
        check = features_match_processed(renderer, demo.states, eps[demo.demo], demo.actions)
        if not check["pass"]:
            raise AssertionError(f"{demo.demo}: raw states do not match the processed obs: "
                                 f"{check}")
        frames[demo.demo] = {c: np.stack([renderer.rgb(s, c, size) for s in demo.states])
                             for c in DEFAULT_CAMERAS}
        states[demo.demo] = [sha256_state(s) for s in demo.states]
        if (i + 1) % 20 == 0:
            print(f"{task}: {i + 1} demos, {time.perf_counter() - t0:.0f} s", flush=True)
    renderer.close()
    render = {"backend": BACKEND, "renderer": "roboforge.data.rgb.DemoFrameRenderer",
              "warm_up_renders": 1, "geomgroup": [0, 1, 1, 0, 0, 0], "decision": "D-074"}
    digest = write_rgb_dataset(out, frames, states, DEFAULT_CAMERAS, {
        "task": task, "size": size, "raw_sha256": raw_sha,
        "processed_content_sha256": proc["content_sha256"], "render": render})
    n_frames = sum(len(v) for v in states.values())
    return {"schema": MANIFEST_SCHEMA, "task": task, "size": size, "cameras": DEFAULT_CAMERAS,
            "content_sha256": digest, "file_sha256": sha256_file(out),
            "relpath": rgb_relpath(task, size), "demos": len(frames), "frames": n_frames,
            "complete": limit is None, "raw_sha256": raw_sha,
            "processed_content_sha256": proc["content_sha256"], "render": render,
            "platform": evaluation_platform(), "git": git_revision(),
            "build_seconds": time.perf_counter() - t0}


def install(manifest_path: Path) -> Path:
    """Host side: re-verify the staged file, copy it into data/ and write the manifest."""
    m = json.loads(manifest_path.read_text())
    if not m["complete"]:
        raise SystemExit("refusing to install a partial (--limit) build")
    staged = manifest_path.with_name(Path(m["relpath"]).name)
    if sha256_file(staged) != m["file_sha256"]:
        raise SystemExit(f"{staged}: file hash differs from its manifest")
    load_rgb_dataset(staged, expected_sha256=m["content_sha256"])
    dest = REPO_ROOT / "data" / m["relpath"]
    if dest.exists():
        raise SystemExit(f"{dest} exists; installed RGB datasets are never overwritten")
    shutil.copy2(staged, dest)
    if sha256_file(dest) != m["file_sha256"]:
        raise SystemExit(f"{dest}: copy hash mismatch")
    out = REPO_ROOT / "data" / "manifests" / "rgb" / f"{m['task']}_rgb{m['size']}_v1.json"
    write_json(out, m)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (see the module docstring)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="lift")
    p.add_argument("--size", type=int, default=84)
    p.add_argument("--limit", type=int, help="first N demos only (smoke; never installed)")
    p.add_argument("--out-dir", type=Path, help="default: runs/rgb_data/<task>")
    p.add_argument("--install", type=Path, metavar="MANIFEST",
                   help="host: install a staged build from its manifest")
    args = p.parse_args(argv)
    if args.install:
        print(f"installed; manifest {repo_relative(install(args.install))}")
        return 0
    name = Path(rgb_relpath(args.task, args.size)).name
    out_dir = args.out_dir or REPO_ROOT / "runs" / "rgb_data" / args.task
    out = out_dir / (name if args.limit is None else f"smoke{args.limit}_{name}")
    m = build(args.task, args.size, out, args.limit)
    write_json(out.with_suffix(".manifest.json"), m)
    print(json.dumps({k: m[k] for k in ("task", "size", "demos", "frames", "content_sha256",
                                        "file_sha256", "build_seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
