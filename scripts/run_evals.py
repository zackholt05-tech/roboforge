"""Run the preregistered robustness and ablation evaluations (RESEARCH_PROTOCOL §14.3-14.4).

    ROBOFORGE_EVIDENCE_SESSION=7 uv run --frozen python scripts/run_evals.py \
        --batches robustness a1 a2 \
        --config lift=reports/session6/lift_test_configuration.json \
        --config can=reports/session7/can_test_configuration.json

Every job evaluates one already-selected checkpoint (the 12 per task of the test
configuration records; A3: the selected checkpoints of the A3 suite) on one robustness suite,
optionally with a preregistered inference variant, through
``scripts/eval_policy.py --suite S [--variant V]`` in its own canonical container, at most 4
at a time (Gate C, D-051). Nothing here selects. Batches:

* ``robustness``: every checkpoint on every suite of ``robustness.sets_for(task)`` (H4);
* ``a1``: diffusion checkpoints on ``robustness_base`` with ``--variant trained-weights``;
* ``a2``: diffusion checkpoints on ``robustness_base`` with ``--variant ddpm100``;
* ``a3``: the A3 suite's selected checkpoints (``--a3-summary``) on ``robustness_base``.

A job whose summary exists (``reports/session<N>/robustness/evals/<eval_run_id>.json``) is
skipped; a failed job is recorded and not relaunched. The tree must be clean at every launch.
Events: ``logs/session<N>/robustness/events.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboforge.eval.validation import eval_run_id
from roboforge.paths import REPO_ROOT, evidence_reports_dir, evidence_session, repo_relative
from roboforge.provenance import git_revision
from roboforge.sim import robustness as rb

PLATFORM = "linux-aarch64-container-1f685638f28c"
DIFFUSION = "diffusion_unet"
VARIANT = {"a1": "trained-weights", "a2": "ddpm100"}


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def checkpoints(config_path: Path) -> list[dict[str, Any]]:
    """The 12 selected checkpoints of a test-configuration record, in family/seed order."""
    rec = json.loads(config_path.read_text())
    return [{"task": rec["task"], "family": fam, "seed": s["seed"],
             "train_run_id": s["train_run_id"], "step": int(s["selected_step"])}
            for fam, f in rec["families"].items() for s in f["seeds"]]


def a3_checkpoints(summary_path: Path) -> list[dict[str, Any]]:
    """The selected checkpoints of every run of the A3 suite summary (all seeds)."""
    s = json.loads(summary_path.read_text())
    out = []
    for fam in s["families"]:
        for r in fam["runs"]:
            if r.get("selection") is None:
                raise SystemExit(f"A3 run {r['run_id']} has no selection")
            out.append({"task": s["task"], "family": DIFFUSION, "seed": r["seed"],
                        "train_run_id": r["run_id"], "step": int(r["selection"]["step"]),
                        "arm": "T_o=1"})
    return out


def jobs(batches: list[str], configs: dict[str, Path], a3: Path | None) -> list[dict[str, Any]]:
    """Every job of the requested batches, in batch, task, family, seed, suite order."""
    out = []
    for batch in batches:
        if batch == "a3":
            if a3 is None:
                raise SystemExit("--batches a3 needs --a3-summary")
            out += [{**c, "batch": "a3", "suite": rb.base_name(), "variant": None}
                    for c in a3_checkpoints(a3)]
            continue
        for task, path in configs.items():
            for c in checkpoints(path):
                if c["task"] != task:
                    raise SystemExit(f"{path}: task {c['task']} != {task}")
                if batch == "robustness":
                    out += [{**c, "batch": batch, "suite": s, "variant": None}
                            for s in rb.sets_for(task)]
                elif c["family"] == DIFFUSION:
                    out.append({**c, "batch": batch, "suite": rb.base_name(),
                                "variant": VARIANT[batch]})
    for j in out:
        j["eval_run_id"] = eval_run_id(j["train_run_id"], j["step"], PLATFORM,
                                       stochastic=j["family"] == DIFFUSION, suite=j["suite"],
                                       variant=j["variant"])
    if len({j["eval_run_id"] for j in out}) != len(out):
        raise SystemExit("duplicate jobs")
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--batches", nargs="+", required=True,
                   choices=["robustness", "a1", "a2", "a3"])
    p.add_argument("--config", action="append", default=[], metavar="TASK=PATH")
    p.add_argument("--a3-summary", type=Path, default=None)
    p.add_argument("--max-eval", type=int, default=4)
    p.add_argument("--poll-s", type=float, default=10.0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.max_eval > 4:
        p.error("Gate C (D-051) allows at most 4 containers")
    configs = {k: REPO_ROOT / v for k, v in (c.split("=", 1) for c in a.config)}
    todo = jobs(a.batches, configs, a.a3_summary)
    reports = evidence_reports_dir("robustness") / "evals"
    logs = REPO_ROOT / "logs" / f"session{evidence_session()}" / "robustness"
    if a.dry_run:
        for j in todo:
            done = (reports / f"{j['eval_run_id']}.json").exists()
            print(("done " if done else "todo ") + j["eval_run_id"])
        print(f"{len(todo)} jobs")
        return 0
    if git_revision()["dirty"]:
        raise SystemExit("working tree is dirty (code/configs/docs): refusing to launch")
    logs.mkdir(parents=True, exist_ok=True)

    def event(ev: str, **kw: object) -> None:
        rec = {"utc": _now(), "event": ev, **kw}
        with (logs / "events.jsonl").open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        print(json.dumps(rec, sort_keys=True), flush=True)

    pending = []
    for j in todo:
        if (reports / f"{j['eval_run_id']}.json").exists():
            event("skip_done", eval_run_id=j["eval_run_id"])
        else:
            pending.append(j)
    event("start", batches=a.batches, n=len(pending), git=git_revision(),
          configs={k: repo_relative(v) for k, v in configs.items()},
          a3_summary=repo_relative(a.a3_summary) if a.a3_summary else None)
    running: list[tuple[subprocess.Popen, dict[str, Any], float]] = []
    failed = []
    while pending or running:
        while pending and len(running) < a.max_eval:
            j = pending.pop(0)
            if git_revision()["dirty"]:
                raise SystemExit("working tree became dirty: refusing to launch more")
            cmd = ["infra/containers/run_sim_cpu.sh", "python", "scripts/eval_policy.py",
                   "--train-run", f"runs/{j['train_run_id']}", "--steps", str(j["step"]),
                   "--suite", j["suite"], "--reports-dir",
                   f"reports/session{evidence_session()}"]
            if j["variant"]:
                cmd += ["--variant", j["variant"]]
            log = logs / "jobs" / f"{j['eval_run_id']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log.open("a"),
                                    stderr=subprocess.STDOUT,
                                    env={**os.environ,
                                         "ROBOFORGE_EVIDENCE_SESSION": str(evidence_session())})
            running.append((proc, j, time.time()))
            event("launch", pid=proc.pid, log=repo_relative(log), argv=cmd[1:],
                  **{k: j[k] for k in ("batch", "task", "family", "seed", "suite", "variant",
                                       "eval_run_id")})
        time.sleep(a.poll_s)
        for item in list(running):
            proc, j, t0 = item
            if proc.poll() is None:
                continue
            running.remove(item)
            ok = proc.returncode == 0 and (reports / f"{j['eval_run_id']}.json").exists()
            event("finished", eval_run_id=j["eval_run_id"], returncode=proc.returncode,
                  wall_s=round(time.time() - t0, 1), ok=ok)
            if not ok:
                failed.append(j["eval_run_id"])
    event("end", failed=failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
