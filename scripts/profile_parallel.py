"""Run several canonical evaluation containers in parallel and measure throughput (Session 5).

    python scripts/profile_parallel.py --levels 1 2 4 --out-dir reports/session5/rollout_profile

Engineering profile on **validation starts only**. Each level P starts P containers at once
(``infra/containers/run_sim_cpu.sh`` with ``ROBOFORGE_EVIDENCE_SESSION=5``), and each runs
``scripts/profile_rollouts.py`` for every job in ``--jobs``. Recorded per level:

* each container's process wall time (container start included) and its in-process
  evaluation wall time;
* the aggregate throughput, in evaluations per hour and control steps per second;
* the speed-up over P = 1;
* one ``docker stats`` sample taken mid-run (CPU % and memory per container);
* whether every episode's ``physics_sha256`` equals the recorded canonical evaluation.

The Docker VM's CPU and memory limits (``docker info``) bound the useful P.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import git_revision

RUNNER = REPO_ROOT / "infra" / "containers" / "run_sim_cpu.sh"
DIFF_RUN = "runs/s4_diffusion_unet_lift_seed0_336e15e9"
DIFF_REF = ("runs/s4_diffusion_unet_lift_seed0_336e15e9__val_step0015000"
            "__linux-aarch64-container-1f685638f28c__ps0r0/manifest.json")
JOBS = {
    "lift_diffusion": ["--task", "lift", "--train-run", DIFF_RUN, "--step", "15000",
                       "--n-states", "50", "--reference", DIFF_REF, "--render-frames", "10"],
    "can_random": ["--task", "can", "--policy", "random_uniform", "--n-states", "10",
                   "--render-frames", "10"],
}


def docker_stats() -> list[dict[str, Any]]:
    """One ``docker stats`` sample of running containers."""
    p = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}"],
                       capture_output=True, text=True)
    return [json.loads(x) for x in p.stdout.splitlines() if x.strip()]


def level(p: int, job: str, out_dir: Path) -> dict[str, Any]:
    """Run ``p`` containers of ``job`` at once."""
    env = {**os.environ, "ROBOFORGE_EVIDENCE_SESSION": "5"}
    procs = []
    t0 = time.perf_counter()
    for i in range(p):
        out = out_dir / "raw" / f"P{p}_{job}_{i}.json"
        cmd = [str(RUNNER), "python", "scripts/profile_rollouts.py", *JOBS[job], "--tag",
               f"P{p}_{job}_{i}", "--out", repo_relative(out)]
        procs.append((out, time.perf_counter(), subprocess.Popen(
            cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)))
    time.sleep(20)
    stats = docker_stats()
    rows = []
    for out, ts, proc in procs:
        log = proc.communicate()[0]
        rows.append({"out": repo_relative(out), "returncode": proc.returncode,
                     "process_wall_s": time.perf_counter() - ts,
                     "profile": json.loads(out.read_text()) if out.is_file() else None,
                     "log_tail": log.splitlines()[-5:]})
    wall = time.perf_counter() - t0
    ok = [r["profile"] for r in rows if r["profile"]]
    steps = sum(r["n_steps"] for r in ok)
    return {"P": p, "job": job, "wall_s": wall, "n_ok": len(ok),
            "evaluations_per_hour": 3600 * len(ok) / wall,
            "aggregate_steps_per_s": steps / wall,
            "mean_eval_wall_s": sum(r["eval_wall_s"] for r in ok) / max(len(ok), 1),
            "mean_process_wall_s": sum(r["process_wall_s"] for r in rows) / len(rows),
            "physics_sha256_all_match": [r["physics_sha256_all_match"] for r in ok],
            "docker_stats_mid_run": stats, "containers": rows}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--jobs", nargs="+", default=list(JOBS), choices=list(JOBS))
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "session5" /
                    "rollout_profile")
    args = ap.parse_args(argv)
    info = subprocess.run(["docker", "info", "--format", "{{json .}}"], capture_output=True,
                          text=True)
    d = json.loads(info.stdout)
    results = [level(p, j, args.out_dir) for j in args.jobs for p in args.levels]
    for j in args.jobs:
        base = next((r for r in results if r["job"] == j and r["P"] == 1), None)
        for r in results:
            if r["job"] == j and base:
                r["speedup_vs_P1"] = r["evaluations_per_hour"] / base["evaluations_per_hour"]
    summary = {"schema": "roboforge.parallel_rollout_profile/1",
               "kind": "engineering profile on validation starts (not a result)",
               "docker_vm": {"ncpu": d.get("NCPU"), "mem_bytes": d.get("MemTotal"),
                             "server_version": d.get("ServerVersion")},
               "levels": results, "git": git_revision()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for r in results:
        print(f"{r['job']:15s} P={r['P']}: wall {r['wall_s']:.0f} s, "
              f"{r['evaluations_per_hour']:.1f} evals/h, {r['aggregate_steps_per_s']:.0f} "
              f"steps/s, mean eval {r['mean_eval_wall_s']:.0f} s, speed-up "
              f"{r.get('speedup_vs_P1', float('nan')):.2f}, sha match "
              f"{r['physics_sha256_all_match']}")
    return 0 if all(r["n_ok"] == r["P"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
