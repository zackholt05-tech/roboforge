"""Rendering-backend probe: offscreen RGB frames of a real task, nothing else (Session 5).

    MUJOCO_GL=egl python scripts/render_probe.py --task lift --out probe.json

Builds the task with an offscreen renderer, resets it, and renders ``--frames`` frames from
each camera at ``--size``². It needs no dataset and no evaluation suite. The Gate A probe
uses it to compare EGL with OSMesa inside a Modal GPU container (KNOWN_ISSUES KI-001), whose
image carries neither (KI-033). A camera passes if its frame has the expected shape and is
not blank (pixel std > 5). The exit code is 0 only if every camera passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from roboforge.sim.backend import configure_mujoco_gl

BACKEND = configure_mujoco_gl()

import numpy as np  # noqa: E402

from roboforge.sim.factory import load_sim_config, make_env  # noqa: E402
from roboforge.sim.renderer import render_rgb  # noqa: E402


def probe(task: str, cameras: list[str], size: int, frames: int) -> dict:
    """Construct, reset and render; per-camera timing and a non-blank check."""
    t0 = time.perf_counter()
    env = make_env(load_sim_config(task), offscreen=True)
    construct_s = time.perf_counter() - t0
    env.reset()
    per_camera = {}
    for cam in cameras:
        first = time.perf_counter()
        frame = render_rgb(env, cam, size, size)  # first frame includes context creation
        first_s = time.perf_counter() - first
        times = []
        for _ in range(frames):
            t = time.perf_counter()
            frame = render_rgb(env, cam, size, size)
            times.append(time.perf_counter() - t)
        std = float(np.asarray(frame, dtype=np.float64).std())
        per_camera[cam] = {"shape": list(frame.shape), "std": std, "ok": std > 5.0,
                           "first_frame_s": first_s, "median_frame_s": float(np.median(times)),
                           "frames_timed": frames}
    env.close()
    return {"schema": "roboforge.render_probe/1", "backend": BACKEND, "task": task,
            "size": size, "construct_s": construct_s, "cameras": per_camera,
            "ok": all(c["ok"] for c in per_camera.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="lift")
    ap.add_argument("--cameras", nargs="+", default=["agentview", "frontview"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    res = probe(a.task, a.cameras, a.size, a.frames)
    text = json.dumps(res, indent=1)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    print(text)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
