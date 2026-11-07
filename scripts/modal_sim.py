"""Local simulation of a Modal training job in a container-shaped directory (Session 5).

    python scripts/modal_sim.py --root <dir> --config configs/models/diffusion_unet.yaml \
        --prefix s5sim --set train.max_steps=2000 ... --report reports/session5/modal_sim/x.json

No Modal resource is touched. ``<root>/container/`` is laid out like the Modal container of
``infra/modal_app.py`` (``configs/compute/modal.yaml`` ``app``):

* ``container/app/``: copies of the mounted code directories, so ``REPO_ROOT`` is ``/app``'s
  stand-in and manifests record ``runs/<run_id>/...``;
* ``container/app/runs/``: the runs Volume (mounted at ``/app/runs`` on Modal);
* ``container/data/``: the data Volume (``ROBOFORGE_DATA_ROOT``), holding the uploaded
  processed file, which is hash-checked against the plan as on Modal.

The job body is the real :func:`roboforge.compute.remote.train_job`, run in a subprocess with
the container's ``PYTHONPATH``, data root and git variables. Two runs of the same config
train in separate roots:

* ``reference``: one uninterrupted launch;
* ``chain``: the failure modes a remote run can meet, in order:
  1. a **preemption**: SIGINT once the heartbeat passes ``--sigint-at``. The job saves a
     resume checkpoint and re-raises; SIGKILL follows 30 s later if it is still alive, as on
     Modal;
  2. Modal's **restart on the same input** (same launch ID), killed by **SIGKILL** without
     warning once past ``--sigkill-at``;
  3. another same-input restart, stopped by its own **deadline** (``--deadline-s``);
  4. a **duplicate** ``new`` launch (fresh launch ID), which must be refused;
  5. an approved **resume** launch (new launch ID), which completes;
  6. a duplicate ``new`` launch after completion, which must be refused.

Then the chain's run is "downloaded" (copied out of the runs Volume into a staging directory,
as ``modal volume get`` would) and installed into the local ``runs/`` by the hash-validated
:func:`roboforge.compute.retrieve.install`. The report compares every cadence checkpoint of
the chain with the reference's (:func:`roboforge.train.checkpoint.checkpoint_differences`),
and the metric rows apart from wall-clock fields.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from roboforge.compute.plan import train_plan
from roboforge.compute.pricing import load_modal_config
from roboforge.compute.retrieve import install, verify_run_dir
from roboforge.paths import REPO_ROOT, data_root, repo_relative
from roboforge.provenance import environment_snapshot, git_revision, sha256_file
from roboforge.train.checkpoint import checkpoint_differences

GRACE_S = 30.0  # Modal: SIGKILL 30 s after the preemption SIGINT


def materialize(root: Path, plan: dict[str, Any]) -> dict[str, Path]:
    """Create the container-shaped tree and 'upload' the processed data."""
    app = load_modal_config()["app"]
    c = root / "container"
    app_dir, data_dir = c / "app", c / "data"
    for d in app["code_dirs"]:
        shutil.copytree(REPO_ROOT / d, app_dir / d,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (app_dir / "runs").mkdir(parents=True)
    rel = plan["data"]["processed_relpath"]
    src = data_root() / rel
    dst = data_dir / rel
    dst.parent.mkdir(parents=True)
    shutil.copy2(src, dst)
    if sha256_file(dst) != plan["data"]["processed_file_sha256"]:
        raise RuntimeError("upload hash mismatch")
    return {"app": app_dir, "data": data_dir, "runs": app_dir / "runs"}


def _env(lay: dict[str, Path], plan: dict[str, Any]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROBOFORGE_")}
    env.update(PYTHONPATH=str(lay["app"] / "src"), ROBOFORGE_DATA_ROOT=str(lay["data"]),
               ROBOFORGE_GIT_SHA=plan["git"]["sha"],
               ROBOFORGE_GIT_DIRTY="1" if plan["git"]["dirty"] else "0",
               ROBOFORGE_GIT_SOURCE="launch plan (local Modal simulation)",
               OMP_NUM_THREADS="1")
    return env


def _progress_step(run_dir: Path) -> int:
    p = run_dir / "progress.json"
    if not p.is_file():
        return -1
    try:
        return int(json.loads(p.read_text())["step"])
    except (json.JSONDecodeError, KeyError):
        return -1


def segment(lay: dict[str, Path], plan: dict[str, Any], tag: str, *,
            signal_at: tuple[int, int] | None = None) -> dict[str, Any]:
    """Run one job process; optionally deliver ``signal_at = (signum, step)``."""
    pdir = lay["app"].parent / "plans"
    pdir.mkdir(exist_ok=True)
    pf, rf = pdir / f"{tag}.plan.json", pdir / f"{tag}.result.json"
    pf.write_text(json.dumps(plan, indent=1))
    cmd = [sys.executable, "-m", "roboforge.compute.remote", "train", "--plan", str(pf),
           "--result", str(rf)]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=lay["app"], env=_env(lay, plan),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sent = None
    run_dir = lay["runs"] / plan["run_id"]
    if signal_at is not None:
        signum, at = signal_at
        while proc.poll() is None and _progress_step(run_dir) < at:
            time.sleep(0.05)
        if proc.poll() is None:
            sent = {"signal": signal.Signals(signum).name, "heartbeat_step":
                    _progress_step(run_dir), "after_s": time.perf_counter() - t0}
            proc.send_signal(signum)
            if signum == signal.SIGINT:
                try:
                    proc.wait(timeout=GRACE_S)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    sent["sigkill_after_grace"] = True
    out = proc.communicate()[0]
    return {"tag": tag, "launch_id": plan["launch_id"], "mode": plan["mode"],
            "deadline_s": plan["deadline_s"], "returncode": proc.returncode, "signal": sent,
            "wall_s": time.perf_counter() - t0,
            "result": json.loads(rf.read_text()) if rf.is_file() else None,
            "output_tail": out.splitlines()[-6:]}


def _rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    for r in rows:
        r.pop("wall_s", None)
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--task", default="lift")
    p.add_argument("--prefix", default="s5sim")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--sigint-at", type=int, required=True)
    p.add_argument("--sigkill-at", type=int, required=True)
    p.add_argument("--deadline-s", type=float, required=True)
    p.add_argument("--install", action="store_true", help="install the chain run into runs/")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args(argv)
    args.root = args.root.resolve()  # subprocesses run with the container's cwd
    if args.root.exists():
        raise SystemExit(f"{args.root} exists; use a fresh root")
    base = train_plan(args.config, task=args.task, seed=None, prefix=args.prefix,
                      overrides=args.overrides, timeout_s=None, events=[])
    rid = base["run_id"]
    procs: dict[str, Any] = {}
    lay_ref = materialize(args.root / "reference", base)
    lay = materialize(args.root / "chain", base)
    ref_plan = {**base, "launch_id": "sim-ref-" + uuid.uuid4().hex[:8]}
    with cf.ThreadPoolExecutor(1) as pool:  # the reference trains while the chain runs
        ref_future = pool.submit(segment, lay_ref, ref_plan, "reference")
        l1 = {**base, "launch_id": "sim-L1-" + uuid.uuid4().hex[:8]}
        chain = [segment(lay, l1, "1_preempt_sigint", signal_at=(signal.SIGINT,
                                                                  args.sigint_at)),
                 segment(lay, l1, "2_restart_sigkill", signal_at=(signal.SIGKILL,
                                                                   args.sigkill_at)),
                 segment(lay, {**l1, "deadline_s": args.deadline_s}, "3_restart_deadline")]
        chain.append(segment(lay, {**base, "launch_id": "sim-L3-" + uuid.uuid4().hex[:8]},
                             "4_duplicate_new_refused"))
        chain.append(segment(lay, {**base, "mode": "resume",
                                   "launch_id": "sim-L2-" + uuid.uuid4().hex[:8]},
                             "5_resume_launch"))
        chain.append(segment(lay, {**base, "launch_id": "sim-L4-" + uuid.uuid4().hex[:8]},
                             "6_duplicate_after_complete_refused"))
        procs["reference"] = ref_future.result()
    procs["chain"] = chain
    ref_dir, ch_dir = lay_ref["runs"] / rid, lay["runs"] / rid
    ref_m = json.loads((ref_dir / "manifest.json").read_text())
    ch_m = json.loads((ch_dir / "manifest.json").read_text())
    diffs = {c["step"]: checkpoint_differences(lay["app"] / c["path"],
                                               lay_ref["app"] / r["path"])
             for c, r in zip(ch_m["checkpoints"], ref_m["checkpoints"], strict=True)}
    staging = args.root / "download_staging" / rid
    shutil.copytree(ch_dir, staging)  # what `modal volume get roboforge-runs <rid>` returns
    retrieval = (install(staging, rid) if args.install
                 else {"verified_staged": verify_run_dir(staging, rid), "installed": None})
    expect = {"1_preempt_sigint": 130, "2_restart_sigkill": -9, "3_restart_deadline": 3,
              "4_duplicate_new_refused": 1, "5_resume_launch": 0,
              "6_duplicate_after_complete_refused": 1}
    rc_ok = {s["tag"]: s["returncode"] == expect[s["tag"]] for s in chain}
    dup_refused = all("RunExistsError" in "\n".join(s["output_tail"]) for s in chain
                      if s["tag"].startswith(("4_", "6_")))
    checks = {
        "chain_returncodes_as_expected": all(rc_ok.values()),
        "duplicates_refused_with_RunExistsError": dup_refused,
        "reference_complete": procs["reference"]["returncode"] == 0,
        "cadence_checkpoints_content_identical": all(not d for d in diffs.values()),
        "metrics_rows_equal_except_wall_s": _rows(ch_dir) == _rows(ref_dir),
        "final_offline_equal": ch_m["final_offline"] == ref_m["final_offline"],
        "manifest_paths_repo_relative": all(c["path"].startswith(f"runs/{rid}/")
                                            for c in ch_m["checkpoints"]),
        "retrieval_hash_verified": bool(retrieval.get("verified_staged")),
        "segments_recorded": [e["event"] + ":" + str(e.get("reason", "")) for e in
                              ch_m["segments"]],
    }
    passed = all(v for k, v in checks.items() if k != "segments_recorded")
    report = {"schema": "roboforge.modal_sim/1", "kind": "local simulation (no Modal resource)",
              "run_id": rid, "plan": {k: base[k] for k in ("function", "timeout_s",
                                                         "deadline_s", "policy", "resources",
                                                         "image", "volumes", "estimate")},
              "segments": procs, "checkpoint_differences": diffs, "retrieval": retrieval,
              "chain_manifest_timing": ch_m["timing"], "reference_timing": ref_m["timing"],
              "resume_checkpoints": ch_m["resume_checkpoints"], "checks": checks,
              "returncode_checks": rc_ok, "passed": passed, "git": git_revision(),
              "environment": environment_snapshot(), "root": repo_relative(args.root)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({"passed": passed, "checks": checks, "report": str(args.report)},
                     indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
