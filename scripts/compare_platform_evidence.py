"""Compare Session 0 evidence produced on two platforms (e.g. macOS host vs Linux container).

Reads ``data_report_{task}.json``, ``smoke_sim_{task}.json`` and ``frames/*.png`` from two
evidence directories and reports whether pass/fail outcomes agree, how far per-demo replay
errors differ, and how far rendered frames differ pixel-wise. Writes one JSON summary.

    uv run python scripts/compare_platform_evidence.py \
        --a reports/feasibility --b reports/feasibility/linux_aarch64_osmesa \
        --out reports/feasibility/linux_aarch64_osmesa/platform_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from roboforge.paths import repo_relative
from roboforge.provenance import git_revision


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _flat_errors(demo: dict[str, Any]) -> dict[str, float]:
    out = {f"{g}.{k}": float(v) for g in ("max_errors", "final_errors")
           for k, v in demo[g].items()}
    out["replay_first_success_step"] = float(demo["replay_first_success_step"] or -1)
    return out


def compare_task(a_dir: Path, b_dir: Path, task: str) -> dict[str, Any]:
    """Return the outcome and numeric differences between two platforms for one task."""
    a, b = _load(a_dir / f"data_report_{task}.json"), _load(b_dir / f"data_report_{task}.json")
    pa = {d["demo"]: d for d in a["replay"]["per_demo"]}
    pb = {d["demo"]: d for d in b["replay"]["per_demo"]}
    if pa.keys() != pb.keys():
        raise ValueError(f"{task}: the two reports replayed different demo sets")
    failed = {name: sorted(k for k, d in p.items() if not d["replay_success_at_end"])
              for name, p in (("a", pa), ("b", pb))}
    fields = _flat_errors(next(iter(pa.values()))).keys()
    max_diff = {f: max(abs(_flat_errors(pa[d])[f] - _flat_errors(pb[d])[f]) for d in pa)
                for f in fields}
    sa, sb = _load(a_dir / f"smoke_sim_{task}.json"), _load(b_dir / f"smoke_sim_{task}.json")
    return {
        "n_demos": len(pa),
        "replay_success": {"a": a["replay"]["replay_success_count"],
                           "b": b["replay"]["replay_success_count"]},
        "failed_demos": failed,
        "same_failed_demos": failed["a"] == failed["b"],
        "per_demo_errors_bit_identical": all(v == 0.0 for v in max_diff.values()),
        "per_demo_error_max_abs_diff": max_diff,
        "integrity_checks": {"a": a["integrity_checks"], "b": b["integrity_checks"]},
        "obs_state_alignment_demo0": {"a": a["obs_state_alignment_demo0"],
                                      "b": b["obs_state_alignment_demo0"]},
        "smoke_checks_equal": sa["checks"] == sb["checks"],
        "smoke_all_passed": {"a": sa["all_passed"], "b": sb["all_passed"]},
        "render_backend": {"a": a["render"]["backend"], "b": b["render"]["backend"]},
        "render_ms_mean": {"a": a["render"]["render_ms_mean"], "b": b["render"]["render_ms_mean"]},
        "control_steps_per_sec_median": {"a": a["replay"]["control_steps_per_sec_median"],
                                         "b": b["replay"]["control_steps_per_sec_median"]},
    }


def compare_frames(a_dir: Path, b_dir: Path) -> dict[str, Any]:
    """Return per-frame mean absolute pixel difference for frames present in both dirs."""
    out: dict[str, Any] = {}
    for fb in sorted((b_dir / "frames").glob("*.png")):
        fa = a_dir / "frames" / fb.name
        if not fa.exists():
            continue
        x, y = iio.imread(fa).astype(np.int16), iio.imread(fb).astype(np.int16)
        if x.shape != y.shape:
            out[fb.name] = {"shape_a": list(x.shape), "shape_b": list(y.shape)}
            continue
        d = np.abs(x - y)
        out[fb.name] = {"mean_abs_diff": float(d.mean()),
                        "frac_pixels_diff_gt_16": float((d.max(axis=-1) > 16).mean())}
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--a", type=Path, required=True, help="reference evidence directory")
    p.add_argument("--b", type=Path, required=True, help="evidence directory to compare")
    p.add_argument("--tasks", nargs="+", default=["lift", "can"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    summary = {
        "a": repo_relative(args.a), "b": repo_relative(args.b), "git": git_revision(),
        "tasks": {t: compare_task(args.a, args.b, t) for t in args.tasks},
        "frames": compare_frames(args.a, args.b),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    for t, r in summary["tasks"].items():
        print(f"[{t}] replay success a={r['replay_success']['a']} b={r['replay_success']['b']} "
              f"same failed demos={r['same_failed_demos']} "
              f"errors bit-identical={r['per_demo_errors_bit_identical']} "
              f"smoke checks equal={r['smoke_checks_equal']}")
    worst = max(v.get("mean_abs_diff", float("inf")) for v in summary["frames"].values())
    print(f"frames compared: {len(summary['frames'])}, worst mean |pixel diff| {worst:.2f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
