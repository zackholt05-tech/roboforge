"""Launch plans for Modal jobs: built, checked and logged locally (Session 5).

``infra/modal_app.py`` runs under a Python that has the Modal SDK, which is kept outside the
project venv (DECISIONS D-045). It never imports RoboForge. Everything that needs the project
(config resolution, run IDs, git state, data hashes, prices, the approval gate and the cost
ledger) lives here and runs in the venv:

    uv run --frozen python -m roboforge.compute.plan train --config configs/models/bc_mlp.yaml \
        --task lift --seed 0 --prefix s5modal --timeout-s 1800 --out plan.json
    uv run --frozen python -m roboforge.compute.plan probe --out probe.json
    uv run --frozen python -m roboforge.compute.plan authorize --plan plan.json --approval A-01
    uv run --frozen python -m roboforge.compute.plan record-result --plan plan.json \
        --result result.json

``train``/``probe`` only compute a plan and, with ``--record``, log a ``dry_run`` event. They
contact no server. ``authorize`` is the only path to a ``launch`` event: it applies the
approval gate (:mod:`roboforge.compute.approval`) and the launch rules below, and refuses
otherwise.

Launch rules (MASTER_SPEC §6.2; COMPUTE_PLAN §3):

* the working tree must be clean, so the plan's git SHA is the code that runs;
* ``mode: new`` requires a run ID never used locally (``runs/``) or by an earlier launch;
  ``mode: resume`` requires one an earlier launch created;
* ``max_containers = 1``, ``retries = 0``, a 2 s scaledown window and an explicit timeout no
  larger than the function's maximum; the job's own deadline is the timeout minus a margin;
* the plan's upper-bound estimate (timeout × list price + build allowance) must fit the
  approval's remaining cap.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboforge.compute import cost_log
from roboforge.compute.approval import ApprovalError, authorize
from roboforge.compute.pricing import function_resources, launch_upper_bound, load_modal_config
from roboforge.paths import REPO_ROOT
from roboforge.provenance import git_revision, sha256_file

SCHEMA = "roboforge.modal_plan/1"


def image_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    """What the remote image is built from (all hashed, so the plan pins it)."""
    app = cfg["app"]
    return {"base": f"modal.Image.debian_slim(python_version={app['python']!r})",
            "apt": list(app["apt"]), "uv_version": app["uv_version"],
            "install": "Image.uv_sync(frozen=True, extra_options='--no-dev') from uv.lock",
            "uv_lock_sha256": sha256_file(REPO_ROOT / "uv.lock"),
            "pyproject_sha256": sha256_file(REPO_ROOT / "pyproject.toml"),
            "code_dirs": list(app["code_dirs"]), "repo_root": app["repo_root"],
            "env": {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
                    "PYTHONPATH": f"{app['repo_root']}/src",
                    "ROBOFORGE_DATA_ROOT": app["volumes"]["data"]["mount"]}}


def _checks(mode: str, run_id: str | None, events: list[dict[str, Any]]) -> list[str]:
    """Reasons a paid launch of this plan would be refused (empty: none)."""
    blocking = []
    git = git_revision()
    if git["dirty"] or not git["sha"]:
        blocking.append("working tree is dirty (or not a git checkout): commit first")
    if run_id is not None:
        used = run_id in cost_log.launched_run_ids(events)
        local = (REPO_ROOT / "runs" / run_id).exists()
        if mode == "new" and (used or local):
            blocking.append(f"run ID {run_id} already exists (launched={used}, local={local})")
        if mode == "resume" and not used:
            blocking.append(f"run ID {run_id} was never launched; nothing to resume")
    return blocking


def _base(cfg: dict[str, Any], function: str, timeout_s: int) -> dict[str, Any]:
    f, pol = cfg["functions"][function], cfg["policy"]
    if not 60 <= timeout_s <= f["max_timeout_s"]:
        raise ValueError(f"{function}: timeout {timeout_s} s outside [60, {f['max_timeout_s']}]")
    return {"schema": SCHEMA, "created_utc": datetime.now(tz=UTC).isoformat(),
            "function": function, "launch_id": uuid.uuid4().hex, "git": git_revision(),
            "app": cfg["app"]["name"], "image": image_spec(cfg),
            "volumes": cfg["app"]["volumes"],
            "resources": function_resources(cfg, function, timeout_s).to_dict(),
            "timeout_s": int(timeout_s),
            "deadline_s": int(timeout_s - pol["deadline_margin_s"]),
            "policy": {"max_containers": pol["max_containers"], "retries": pol["retries"],
                       "scaledown_window_s": pol["scaledown_window_s"],
                       "preemptible": True, "gpu_count": 1 if f["gpu"] else 0},
            "estimate": launch_upper_bound(cfg, function, timeout_s)}


def train_plan(config: Path, *, task: str | None, seed: int | None, prefix: str,
               overrides: list[str], timeout_s: int | None, mode: str = "new",
               events: list[dict[str, Any]] | None = None,
               simulate_preemption_s: int | None = None) -> dict[str, Any]:
    """Plan of one ``train_cpu`` job (CPU, 1 torch thread: RESEARCH_PROTOCOL §6.2).

    ``simulate_preemption_s`` (engineering smoke only) makes the Modal job call
    ``modal.exception.simulate_preemption`` after that many seconds: a real SIGINT to the
    container process, then a second interrupt 30 s later, as in a preemption."""
    from roboforge.data.pipeline import load_task_artifacts
    from roboforge.data.processed import processed_path, processed_relpath
    from roboforge.train.config import load_config, run_id

    cfg = load_modal_config()
    ov = list(overrides) + ([f"task={task}"] if task else []) + \
        ([f"seed={seed}"] if seed is not None else [])
    rc = load_config(config, ov)
    rid = run_id(rc, prefix)
    m = load_task_artifacts(rc.task)["task_manifest"]["processed"]
    rel = Path(processed_relpath(rc.task, m["converter_version"])).as_posix()
    plan = _base(cfg, "train_cpu", timeout_s or cfg["functions"]["train_cpu"]["default_timeout_s"])
    plan.update(mode=mode, run_id=rid, prefix=prefix, config=rc.to_dict(),
                simulate_preemption_s=simulate_preemption_s,
                config_sha256=rc.sha256(), config_path=Path(config).as_posix(), overrides=ov,
                data={"task": rc.task, "processed_relpath": rel,
                      "processed_content_sha256": m["content_sha256"],
                      "processed_file_sha256": sha256_file(processed_path(
                          rc.task, m["converter_version"]))})
    plan["blocking"] = _checks(mode, rid, events if events is not None else
                               cost_log.read_events())
    return plan


def probe_plan(timeout_s: int | None, families: list[str], steps: int,
               events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Plan of one ``gpu_probe`` job (Gate A: platform, EGL, CUDA, throughput)."""
    cfg = load_modal_config()
    plan = _base(cfg, "gpu_probe", timeout_s or cfg["functions"]["gpu_probe"]["default_timeout_s"])
    plan.update(mode="probe", run_id=None, probe={"families": families, "steps": steps,
                                                  "task": "lift"})
    plan["blocking"] = _checks("probe", None, events if events is not None else
                               cost_log.read_events())
    return plan


def summary(plan: dict[str, Any]) -> dict[str, Any]:
    """The fields a ledger event keeps from a plan."""
    return {k: plan.get(k) for k in ("function", "launch_id", "run_id", "mode", "timeout_s",
                                     "deadline_s", "resources", "policy", "git")} | {
        "estimate_upper_usd": plan["estimate"]["total_usd"],
        "pricing_fetched_utc": plan["estimate"]["pricing_fetched_utc"],
        "uv_lock_sha256": plan["image"]["uv_lock_sha256"]}


def authorize_plan(plan: dict[str, Any], approval_id: str,
                   ledger: Path = cost_log.LEDGER) -> dict[str, Any]:
    """Apply the launch rules and the approval gate; on success log the ``launch`` event."""
    events = cost_log.read_events(ledger)
    blocking = _checks(plan["mode"], plan.get("run_id"), events)
    if blocking:
        raise ApprovalError("; ".join(blocking))
    if any(e["event"] == "launch" and e["launch_id"] == plan["launch_id"] for e in events):
        raise ApprovalError(f"launch {plan['launch_id']} was already authorized")
    decision = authorize(approval_id, plan["function"], plan["estimate"]["total_usd"], events)
    return cost_log.append_event("launch", {**summary(plan), "approval_id": approval_id,
                                            "decision": decision}, ledger)


def record_result(plan: dict[str, Any], result: dict[str, Any],
                  ledger: Path = cost_log.LEDGER) -> dict[str, Any]:
    """Log a launch's ``result``: status, measured container time and its list-price cost."""
    from roboforge.compute.pricing import Resources, cost_usd

    cfg = load_modal_config()
    r = plan["resources"]
    secs = result.get("container_wall_s")
    measured = None
    if secs is not None:
        measured = cost_usd(Resources(float(secs) + cfg["policy"]["scaledown_window_s"],
                                      r["cpu"], r["memory_gib"], r["gpu"]),
                            cfg["pricing"])["total_usd"]
    return cost_log.append_event("result", {
        "launch_id": plan["launch_id"], "run_id": plan.get("run_id"),
        "status": result.get("status"), "container_wall_s": secs,
        "measured_cost_usd": measured,
        "measured_cost_kind": "estimate: measured container time x list price (excludes "
                              "image build and start-up)",
        "modal": result.get("modal"), "detail": result.get("detail")}, ledger)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (see the module docstring)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", type=Path, required=True)
    t.add_argument("--task")
    t.add_argument("--seed", type=int)
    t.add_argument("--prefix", required=True)
    t.add_argument("--set", dest="overrides", action="append", default=[])
    t.add_argument("--timeout-s", type=int)
    t.add_argument("--mode", choices=["new", "resume"], default="new")
    t.add_argument("--simulate-preemption-s", type=int)
    pr = sub.add_parser("probe")
    pr.add_argument("--timeout-s", type=int)
    pr.add_argument("--families", nargs="+",
                    default=["bc_mlp", "bc_rnn", "bc_mlp_chunk", "diffusion_unet"])
    pr.add_argument("--steps", type=int, default=300)
    for s in (t, pr):
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--record", action="store_true", help="log a dry_run event")
    a = sub.add_parser("authorize")
    a.add_argument("--plan", type=Path, required=True)
    a.add_argument("--approval", required=True)
    rr = sub.add_parser("record-result")
    rr.add_argument("--plan", type=Path, required=True)
    rr.add_argument("--result", type=Path, required=True)
    rb = sub.add_parser("record-billing", help="log a billed amount from Modal's records")
    rb.add_argument("--launch-id", required=True)
    rb.add_argument("--app-id", required=True)
    rb.add_argument("--report", type=Path, required=True,
                    help="JSON of `modal billing report --json` covering the launch")
    args = p.parse_args(argv)
    if args.cmd == "record-billing":
        rows = [r for r in json.loads(args.report.read_text())
                if r.get("object_id") == args.app_id]
        if not rows:
            print(f"no billing rows for {args.app_id} in {args.report}", file=sys.stderr)
            return 2
        ev = cost_log.append_event("billing", {
            "launch_id": args.launch_id, "app_id": args.app_id,
            "billed_usd": sum(float(r["cost"]) for r in rows),
            "rows": rows, "report": args.report.as_posix(),
            "source": "modal billing report --json (list cost before credits; Modal notes "
                      "possible collection delays)"})
        print(json.dumps(ev, indent=1, default=str))
        return 0
    if args.cmd in ("train", "probe"):
        plan = (train_plan(args.config, task=args.task, seed=args.seed, prefix=args.prefix,
                           overrides=args.overrides, timeout_s=args.timeout_s, mode=args.mode,
                           simulate_preemption_s=args.simulate_preemption_s)
                if args.cmd == "train" else probe_plan(args.timeout_s, args.families,
                                                       args.steps))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
        if args.record:
            cost_log.append_event("dry_run", {**summary(plan), "blocking": plan["blocking"],
                                              "cost_usd": 0.0})
        print(json.dumps({"plan": str(args.out), "run_id": plan.get("run_id"),
                          "estimate_upper_usd": round(plan["estimate"]["total_usd"], 4),
                          "blocking": plan["blocking"]}, indent=1))
        return 0
    plan = json.loads(args.plan.read_text())
    if args.cmd == "authorize":
        try:
            ev = authorize_plan(plan, args.approval)
        except ApprovalError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(ev["decision"], indent=1))
        return 0
    ev = record_result(plan, json.loads(args.result.read_text()))
    print(json.dumps(ev, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
