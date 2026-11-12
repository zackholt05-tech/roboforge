"""X2-a headroom pilot: train, evaluate on X2 validation starts, measure, read (DECISIONS D-124).

Development numbers only, never a result. From a clean tree, this host orchestrator:

1. launches the four pilot trainings (``scripts/x2_train.py``, the frozen chosen Can
   configurations, seed 0, on the X2 data), each under ``/usr/bin/time -l`` for its peak RSS;
2. as each run indexes a pilot checkpoint (``eval_steps`` of ``configs/extensions/x2_pilot.yaml``),
   evaluates it on the 50 X2 validation starts in the canonical container
   (``scripts/x2_eval.py``), at most 2 containers while the diffusion run trains and 4 after it
   ends, no launch below 4 GiB free disk, on a clean tree, and none while the host swaps hard
   (the D-113 lesson; threshold :data:`SWAP_PAGES_PER_S`);
3. samples host memory, swap, load, disk and container memory every 5 minutes;
4. when everything has ended, writes ``reports/session<N>/x2/pilot/pilot.json`` and
   ``tables.md``: per-run training speed, time, peak RSS and disk; per-evaluation success,
   wall time and throughput; each family's best evaluated checkpoint; and the preregistered
   reading (:func:`roboforge.x2.pilot.reading`).

No evaluation is ever rerun automatically. ``--report`` rewrites the report from the files only.
A first SIGINT stops new launches and asks the trainings to stop with a resume checkpoint;
running evaluations finish::

    ROBOFORGE_EVIDENCE_SESSION=15 python scripts/x2_pilot.py
    ROBOFORGE_EVIDENCE_SESSION=15 python scripts/x2_pilot.py --report
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT, evidence_reports_dir, evidence_session, repo_relative
from roboforge.provenance import git_revision, write_json
from roboforge.train.journal import RunJournal
from roboforge.x2 import states as x2s
from roboforge.x2.pilot import FAMILIES, pilot_config, pilot_runs, reading

RUNS = REPO_ROOT / "runs" / "x2"
EVALS = RUNS / "evals"
POLL_S = 20
HOST_EVERY_S = 300
SWAP_PAGES_PER_S = 3000.0   # swap-ins + swap-outs per second; D-113 saw about 10,000
_STOP = {"requested": False}


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def log_dir() -> Path:
    """``logs/session<N>/x2/pilot``."""
    d = REPO_ROOT / "logs" / f"session{evidence_session()}" / "x2" / "pilot"
    d.mkdir(parents=True, exist_ok=True)
    return d


def event(kind: str, **fields: Any) -> None:
    """Append one event to ``events.jsonl`` and echo it."""
    row = {"utc": _now(), "event": kind, **fields}
    with (log_dir() / "events.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def tree_clean() -> bool:
    """No change outside reports/, logs/ and runs/ (the container's dirty-flag rule)."""
    out = subprocess.run(["git", "status", "--porcelain", "--", ".", ":(exclude)reports",
                          ":(exclude)logs", ":(exclude)runs"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout
    return not out.strip()


def free_gib() -> float:
    """Free disk space of the repository volume, GiB."""
    return shutil.disk_usage(REPO_ROOT).free / 2**30


def swap_counters() -> tuple[int, int]:
    """macOS cumulative (swap-ins, swap-outs) from ``vm_stat``."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    get = lambda k: int(re.search(rf"{k}:\s+(\d+)", out).group(1))  # noqa: E731
    return get("Swapins"), get("Swapouts")


def host_sample(prev: dict[str, Any] | None) -> dict[str, Any]:
    """Swap rate since ``prev``, swap use, load, free disk and container memory."""
    si, so = swap_counters()
    t = time.monotonic()
    rate = None
    if prev is not None and t > prev["t"]:
        rate = ((si - prev["si"]) + (so - prev["so"])) / (t - prev["t"])
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                          text=True).stdout.strip()
    stats = subprocess.run(["docker", "stats", "--no-stream", "--format",
                            "{{.Name}} {{.MemUsage}} {{.CPUPerc}}"], capture_output=True,
                           text=True).stdout.strip().splitlines()
    return {"t": t, "si": si, "so": so, "utc": _now(), "swap_pages_per_s": rate,
            "swapusage": swap, "loadavg": os.getloadavg(), "free_gib": round(free_gib(), 2),
            "containers": stats}


def preflight() -> list[str]:
    """Reasons not to start (empty = go)."""
    problems = []
    if not tree_clean():
        problems.append("working tree is dirty outside reports/logs/runs")
    for rel in ("configs/extensions/x2_pilot.yaml", "data/manifests/tasks/can_mh.json",
                f"data/eval_states/can/{x2s.VALIDATION_SUITE}.json"):
        r = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO_ROOT,
                           capture_output=True)
        if r.returncode:
            problems.append(f"{rel} is not committed")
    if free_gib() < float(pilot_config()["limits"]["min_free_gib"]):
        problems.append(f"free disk {free_gib():.1f} GiB below the limit")
    return problems


def run_dir(run_id: str) -> Path:
    """``runs/x2/<run_id>``."""
    return RUNS / run_id


def indexed(run_id: str) -> dict[int, dict[str, Any]]:
    """Cadence checkpoints indexed so far by a run's journal."""
    d = run_dir(run_id)
    if not (d / "checkpoints.jsonl").exists():
        return {}
    return {int(r["step"]): r for r in RunJournal(d).checkpoints()}


def summary_paths() -> list[Path]:
    """Every X2 pilot evaluation summary written so far."""
    return sorted(evidence_reports_dir("x2", "pilot", "evals").glob("*.json"))


def done_keys() -> set[tuple[str, int]]:
    """(train run, step) pairs that already have a summary."""
    out = set()
    for p in summary_paths():
        s = json.loads(p.read_text())
        out.add((s["train_run_id"], int(s["checkpoint"]["step"])))
    return out


def launch_train(family: str) -> subprocess.Popen:
    """Start one pilot training under ``/usr/bin/time -l``."""
    f = (log_dir() / f"train_{family}.log").open("a")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    return subprocess.Popen(["/usr/bin/time", "-l", sys.executable, "scripts/x2_train.py",
                             "--family", family], cwd=REPO_ROOT, stdout=f,
                            stderr=subprocess.STDOUT, env=env)


def launch_eval(family: str, run_id: str, step: int, sha: str) -> subprocess.Popen:
    """Start one container evaluation on the X2 validation starts."""
    f = (log_dir() / f"eval_{family}_step{step:06d}.log").open("a")
    env = {**os.environ, "ROBOFORGE_EVIDENCE_SESSION": str(evidence_session())}
    cmd = ["infra/containers/run_sim_cpu.sh", "python", "scripts/x2_eval.py", "--train-run",
           f"runs/x2/{run_id}", "--step", str(step), "--family", family,
           "--expected-sha256", sha]
    return subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)


def orchestrate() -> int:
    """Run the pilot to the end (or until SIGINT); then write the report."""
    cfg = pilot_config()
    lim = cfg["limits"]
    problems = preflight()
    if problems:
        event("preflight_failed", problems=problems)
        return 2
    runs = {r.family: r for r in pilot_runs()}
    event("start", git=git_revision(), config=cfg, run_ids={f: r.run_id for f, r in runs.items()})
    trains: dict[str, subprocess.Popen] = {}
    for fam in FAMILIES:
        d = run_dir(runs[fam].run_id)
        if (d / "manifest.json").exists():
            event("train_already_ended", family=fam, run_id=runs[fam].run_id)
            continue
        trains[fam] = launch_train(fam)
        event("train_launched", family=fam, run_id=runs[fam].run_id, pid=trains[fam].pid)
    exits: dict[str, int] = {}
    evals: dict[tuple[str, int], subprocess.Popen] = {}
    launched: set[tuple[str, int]] = set(done_keys())
    eval_exits: dict[str, int] = {}
    prev = host_sample(None)
    last_host = time.monotonic()
    swap_rate: float | None = None
    signal.signal(signal.SIGINT, lambda *_: _STOP.update(requested=True))
    stop_sent = False
    while True:
        if _STOP["requested"] and not stop_sent:
            event("stop_requested")
            for p in trains.values():
                if p.poll() is None:
                    p.send_signal(signal.SIGINT)
            stop_sent = True
        for fam, p in trains.items():
            if fam not in exits and p.poll() is not None:
                exits[fam] = p.returncode
                event("train_ended", family=fam, exit=p.returncode)
        for key, p in list(evals.items()):
            if p.poll() is not None:
                eval_exits[f"{key[0]}@{key[1]}"] = p.returncode
                event("eval_ended", run_id=key[0], step=key[1], exit=p.returncode)
                del evals[key]
        if time.monotonic() - last_host >= 60:
            s = host_sample(prev)
            swap_rate = s["swap_pages_per_s"]
            prev, last_host = s, time.monotonic()
            if int(s["t"]) % HOST_EVERY_S < 60 or (swap_rate or 0) > SWAP_PAGES_PER_S:
                with (log_dir() / "host.jsonl").open("a") as f:
                    f.write(json.dumps({k: v for k, v in s.items()
                                        if k not in ("t", "si", "so")}) + "\n")
        diffusion_training = "diffusion_unet" in trains and "diffusion_unet" not in exits
        limit = (lim["max_containers_during_diffusion"] if diffusion_training
                 else lim["max_containers"])
        queue = [(fam, runs[fam].run_id, step, rec["sha256"]) for fam in FAMILIES
                 for step, rec in sorted(indexed(runs[fam].run_id).items())
                 if step in cfg["eval_steps"] and (runs[fam].run_id, step) not in launched]
        while queue and len(evals) < limit and not _STOP["requested"]:
            if free_gib() < float(lim["min_free_gib"]):
                event("hold_disk", free_gib=round(free_gib(), 2))
                break
            if swap_rate is not None and swap_rate > SWAP_PAGES_PER_S:
                event("hold_swap", swap_pages_per_s=round(swap_rate, 1))
                break
            if not tree_clean():
                event("hold_dirty_tree")
                break
            fam, rid, step, sha = queue.pop(0)
            evals[(rid, step)] = launch_eval(fam, rid, step, sha)
            launched.add((rid, step))
            event("eval_launched", family=fam, run_id=rid, step=step, running=len(evals),
                  limit=limit)
        trains_done = all(f in exits for f in trains)
        if trains_done and not evals and (not queue or _STOP["requested"]):
            break
        time.sleep(POLL_S)
    event("end", train_exits=exits, eval_exits=eval_exits)
    write_report()
    bad = [k for k, v in {**exits, **eval_exits}.items() if v != 0]
    return 0 if not bad and not _STOP["requested"] else 1


def peak_rss_bytes(family: str) -> int | None:
    """Peak RSS of a training from its ``/usr/bin/time -l`` output, if it has ended."""
    p = log_dir() / f"train_{family}.log"
    if not p.exists():
        return None
    m = re.findall(r"(\d+)\s+maximum resident set size", p.read_text())
    return int(m[-1]) if m else None


def du_bytes(path: Path) -> int:
    """Total size of the files under ``path``."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def write_report() -> dict[str, Any]:
    """Pilot report from the files only (training manifests, logs, evaluation summaries)."""
    cfg = pilot_config()
    runs = {r.family: r for r in pilot_runs()}
    out: dict[str, Any] = {"schema": "roboforge.x2_pilot_report/1", "decision": "D-124",
                           "label": cfg["label"], "created_utc": _now(), "git": git_revision(),
                           "config": cfg, "runs": {}, "evaluations": [], "best": {}}
    for fam, r in runs.items():
        d = run_dir(r.run_id)
        rec: dict[str, Any] = {"run_id": r.run_id, "config_sha256": r.config.sha256(),
                               "status": None, "peak_rss_bytes": peak_rss_bytes(fam),
                               "disk_bytes": du_bytes(d)}
        if (d / "manifest.json").exists():
            m = json.loads((d / "manifest.json").read_text())
            rec.update(status=m["status"], timing=m.get("timing"),
                       dataset_id=m.get("data", {}).get("dataset_id"),
                       windows=m.get("windows"), param_count=m.get("param_count"),
                       n_checkpoints=len(m.get("checkpoints", [])),
                       git=m.get("git"))
        out["runs"][fam] = rec
    for p in summary_paths():
        s = json.loads(p.read_text())
        out["evaluations"].append({
            "family": s["family"], "train_run_id": s["train_run_id"],
            "step": s["checkpoint"]["step"], "successes": s["successes"],
            "n": s["n_episodes"], "wilson95": s["wilson95"],
            "first_success_step": s["first_success_step"], "stop_reasons": s["stop_reasons"],
            "n_errors": s["n_errors"], "steps_total": s["steps_total"], "wall_s": s["wall_s"],
            "throughput_control_steps_per_s": s["throughput_control_steps_per_s"],
            "latency_ms_median": s["latency_ms_median_of_episode_medians"],
            "canonical": s["canonical_platform"], "verify_passed": s["verify_passed"],
            "horizon": s["episode_rules"]["horizon"], "git": s["git"],
            "records_bytes": du_bytes(REPO_ROOT / s["records_dir"]),
            "summary": repo_relative(p)})
    for fam in FAMILIES:
        ev = [e for e in out["evaluations"] if e["family"] == fam]
        if ev:
            out["best"][fam] = max(e["successes"] for e in ev)
    complete = (all(out["runs"][f]["status"] == "complete" for f in FAMILIES)
                and all(sum(e["family"] == f for e in out["evaluations"])
                        == len(cfg["eval_steps"]) for f in FAMILIES))
    out["complete"] = complete
    out["reading"] = reading(out["best"]) if complete else {"label": "incomplete"}
    out["disk_bytes_total"] = du_bytes(RUNS)
    rep = evidence_reports_dir("x2", "pilot")
    write_json(rep / "pilot.json", out)
    (rep / "tables.md").write_text(tables(out))
    print(json.dumps({"reading": out["reading"], "best": out["best"],
                      "out": repo_relative(rep / "pilot.json")}, indent=1))
    return out


def tables(out: dict[str, Any]) -> str:
    """Markdown tables of the pilot report."""
    lines = ["# X2-a headroom pilot (development numbers on X2 validation starts; not a result)",
             "", f"Reading: **{out['reading']['label']}** (rule: D-124, "
             "`configs/extensions/x2_pilot.yaml`).", "",
             "| Family | Run | Steps/s | Wall h | CPU h | Peak RSS GiB | Disk GiB |",
             "|---|---|---|---|---|---|---|"]
    for fam, r in out["runs"].items():
        t = r.get("timing") or {}
        rss = r["peak_rss_bytes"]
        lines.append(f"| {fam} | `{r['run_id']}` | {t.get('steps_per_s', float('nan')):.2f} | "
                     f"{t.get('wall_s', float('nan')) / 3600:.2f} | "
                     f"{t.get('cpu_s', float('nan')) / 3600:.2f} | "
                     f"{(rss or float('nan')) / 2**30:.2f} | {r['disk_bytes'] / 2**30:.2f} |")
    lines += ["", "| Family | Step | Successes / 50 | Wilson 95 % | Mean first-success step | "
              "Wall min | Control steps/s |", "|---|---|---|---|---|---|---|"]
    for e in sorted(out["evaluations"], key=lambda e: (FAMILIES.index(e["family"]), e["step"])):
        fs = e["first_success_step"]["mean"]
        lines.append(f"| {e['family']} | {e['step']} | {e['successes']} | "
                     f"[{e['wilson95'][0]:.2f}, {e['wilson95'][1]:.2f}] | "
                     f"{'' if fs is None else f'{fs:.0f}'} | {e['wall_s'] / 60:.1f} | "
                     f"{e['throughput_control_steps_per_s']:.1f} |")
    lines += ["", "Best evaluated checkpoint per family (of 50): "
              + ", ".join(f"{f} {b}" for f, b in out["best"].items()), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--report", action="store_true", help="rewrite the report from the files")
    args = p.parse_args(argv)
    if args.report:
        write_report()
        return 0
    return orchestrate()


if __name__ == "__main__":
    sys.exit(main())
