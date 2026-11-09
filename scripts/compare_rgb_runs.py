"""Compare two ``scripts/rgb_feasibility.py`` runs frame by frame (Session 8).

* Two runs on the **same platform** (e.g. two container processes): every sample and clip frame
  hash must agree, i.e. rendering is bitwise reproducible across processes. Exit code 1 if not.
* Runs on **different platforms** (macOS CGL vs container OSMesa): reports the per-frame mean
  absolute grey-level difference and the share of pixels differing by more than 16 levels,
  from the saved sample PNGs. This is why reported RGB data come from one platform (D-074).

    python scripts/compare_rgb_runs.py RUN_A RUN_B --out reports/session8/rgb_feasibility/x.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import write_json


def compare_runs(a: Path, b: Path, out: Path) -> dict[str, Any]:
    """Frame-by-frame comparison of two runs' inspection samples and clips (same records).

    Same platform: every frame hash must agree (bitwise reproducible across processes).
    Different platforms: reports per-frame mean absolute grey-level difference and the share
    of pixels differing by more than 16 levels, from the saved sample PNGs.
    """
    from PIL import Image

    res: dict[str, Any] = {"a": repo_relative(a), "b": repo_relative(b), "tasks": {}}
    sa = json.loads((a / "summary.json").read_text())
    sb = json.loads((b / "summary.json").read_text())
    same = sa["platform"]["platform_id"] == sb["platform"]["platform_id"]
    res["same_platform"] = same
    for task in sorted(set(sa["tasks"]) & set(sb["tasks"])):
        t: dict[str, Any] = {}
        for kind in ("sample", "clip"):
            fa, fb = a / f"{task}_{kind}.jsonl", b / f"{task}_{kind}.jsonl"
            if not (fa.exists() and fb.exists()):
                continue
            ra = [json.loads(x) for x in fa.read_text().splitlines()]
            rb = [json.loads(x) for x in fb.read_text().splitlines()]
            keys = [(x["demo"], x["step_index"], x["camera"]["name"], x["frame_shape"][0])
                    for x in ra]
            if keys != [(x["demo"], x["step_index"], x["camera"]["name"], x["frame_shape"][0])
                        for x in rb]:
                raise ValueError(f"{task}/{kind}: the runs rendered different frames")
            states_equal = all(x["state_sha256"] == y["state_sha256"] for x, y in zip(ra, rb,
                                                                                     strict=True))
            equal = sum(x["frame_sha256"] == y["frame_sha256"] for x, y in zip(ra, rb,
                                                                               strict=True))
            entry = {"frames": len(ra), "frame_hash_equal": equal, "state_hash_equal": states_equal}
            if kind == "sample" and "png" in ra[0]:
                diffs, big = [], []
                for x, y in zip(ra, rb, strict=True):
                    ia = np.asarray(Image.open(REPO_ROOT / x["png"]), dtype=np.int16)
                    ib = np.asarray(Image.open(REPO_ROOT / y["png"]), dtype=np.int16)
                    d = np.abs(ia - ib)
                    diffs.append(float(d.mean()))
                    big.append(float((d.max(axis=-1) > 16).mean()))
                entry |= {"mean_abs_diff_grey_levels": {"min": min(diffs), "max": max(diffs),
                                                        "mean": float(np.mean(diffs))},
                          "share_pixels_diff_gt16": {"max": max(big),
                                                     "mean": float(np.mean(big))}}
            t[kind] = entry
        res["tasks"][task] = t
    if same:
        res["pass"] = all(e["frame_hash_equal"] == e["frames"] and e["state_hash_equal"]
                          for t in res["tasks"].values() for e in t.values())
    write_json(out, res)
    return res


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    res = compare_runs(args.run_a, args.run_b, args.out)
    print(json.dumps(res, indent=1))
    return 0 if res.get("pass", True) else 1


if __name__ == "__main__":
    sys.exit(main())
