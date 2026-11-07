"""Bodies of the remote jobs, shared by Modal and its local simulation (Session 5).

``infra/modal_app.py`` calls :func:`train_job` and :func:`probe_job` inside the Modal
container. ``scripts/modal_sim.py`` runs exactly the same functions in a local process laid
out like that container (:mod:`roboforge.compute.layout`), so the simulation exercises the
real job code, not a copy:

* repository root ``/app`` (code mounted read-only), so manifests record repo-relative
  ``runs/<run_id>/...`` paths that resolve unchanged after download;
* the runs Volume mounted **at** ``/app/runs`` and the data Volume at ``/data``
  (``ROBOFORGE_DATA_ROOT``);
* the plan's git SHA and dirty flag passed in (the container has no ``.git``).

:func:`train_job` trains through :func:`roboforge.train.run.train_run`, the same code as a
local run, with the plan's launch ID. A Modal preemption restart reuses that ID and resumes.
SIGINT/SIGTERM and the plan's deadline stop the run cleanly with a resume checkpoint. After a
signal the job re-raises ``KeyboardInterrupt``, so the platform treats the input as
interrupted and its preemption restart takes over. After the deadline it returns
``status: interrupted``, and continuing needs a new, approved ``resume`` launch.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

PLAN_GIT_SOURCE = "launch plan (roboforge.compute.plan; clean committed tree)"


def _set_git_env(plan: dict[str, Any]) -> None:
    os.environ.setdefault("ROBOFORGE_GIT_SHA", plan["git"]["sha"])
    os.environ.setdefault("ROBOFORGE_GIT_DIRTY", "1" if plan["git"]["dirty"] else "0")
    os.environ.setdefault("ROBOFORGE_GIT_SOURCE", PLAN_GIT_SOURCE)


def host_info() -> dict[str, Any]:
    """Where the job runs: machine, Python, CPU, and Modal's container identifiers."""
    from roboforge.provenance import _cpu_model, package_versions

    modal_env = {k: v for k, v in os.environ.items()
                 if k.startswith("MODAL_") and "TOKEN" not in k and "SECRET" not in k}
    import threading

    return {"main_thread": threading.current_thread() is threading.main_thread(),
            "executable": sys.executable,
            "machine": platform.machine(), "system": platform.system(),
            "platform": platform.platform(), "python": sys.version.split()[0],
            "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "modal_env": modal_env,
            "packages": package_versions(("torch", "mujoco", "robosuite", "numpy"))}


def verify_data(plan: dict[str, Any]) -> dict[str, Any]:
    """The uploaded processed file must equal the plan's (file SHA-256); the content hash is
    re-checked by ``SequenceDataset.from_committed`` when training loads it."""
    from roboforge.paths import data_root
    from roboforge.provenance import sha256_file

    d = plan["data"]
    f = data_root() / d["processed_relpath"]
    if not f.is_file():
        raise FileNotFoundError(f"processed data missing on the data volume: {f}")
    h = sha256_file(f)
    if h != d["processed_file_sha256"]:
        raise ValueError(f"{f}: SHA-256 {h} != plan {d['processed_file_sha256']}")
    return {"path": d["processed_relpath"], "file_sha256": h, "verified": True}


def train_job(plan: dict[str, Any], *, commit: Callable[[], None] | None = None,
              stop_at_step: int | None = None) -> dict[str, Any]:
    """Train (or continue) the plan's run inside the remote layout; return a result record.

    ``commit`` persists the runs Volume (Modal ``Volume.commit``; a no-op locally). It is
    called after training ends or stops, also when the job re-raises an interrupt.
    ``stop_at_step`` exists for the local simulation only.
    """
    from roboforge.paths import REPO_ROOT
    from roboforge.train.config import from_dict, run_id
    from roboforge.train.run import train_run
    from roboforge.train.stop import StopControl

    t0 = time.perf_counter()
    _set_git_env(plan)
    cfg = from_dict(plan["config"])
    if run_id(cfg, plan["prefix"]) != plan["run_id"] or cfg.sha256() != plan["config_sha256"]:
        raise ValueError("plan config does not reproduce its run ID")
    info = {"host": host_info(), "data": verify_data(plan), "plan_launch_id": plan["launch_id"],
            "function": plan["function"], "mode": plan["mode"]}
    stop = StopControl(deadline_s=plan["deadline_s"], stop_at_step=stop_at_step)
    signalled = installed = False
    try:
        with stop.handle_signals():
            installed = stop.signals_installed
            out = train_run(cfg, REPO_ROOT / "runs", plan["prefix"],
                            launch_id=plan["launch_id"], resume=plan["mode"] == "resume",
                            stop=stop, launch_info={**info, "signals_installed": installed})
        signalled = out.get("status") == "interrupted" and out["reason"].startswith("signal")
    finally:
        if commit is not None:
            commit()
    result = {"status": out["status"], "run_id": plan["run_id"],
              "launch_id": plan["launch_id"], "container_wall_s": time.perf_counter() - t0,
              "signals_installed": installed, "host": info["host"],
              "detail": {k: out.get(k) for k in ("step", "reason", "segment")}
              if out["status"] == "interrupted" else
              {"checkpoints": out["checkpoints"], "timing": out["timing"]}}
    if signalled:
        raise KeyboardInterrupt(f"stopped by {out['reason']} at step {out['step']}; "
                                f"resume checkpoint written")
    return result


def _smoke(backend: str, out_dir: Path) -> dict[str, Any]:
    from roboforge.paths import REPO_ROOT

    env = {**os.environ, "MUJOCO_GL": backend, "PYOPENGL_PLATFORM": backend}
    t = time.perf_counter()
    p = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "smoke_sim.py"), "--task",
                        "lift", "--out-dir", str(out_dir / backend)], env=env,
                       capture_output=True, text=True, timeout=600)
    lines = (p.stdout + p.stderr).splitlines()
    return {"backend": backend, "returncode": p.returncode,
            "passed": sum("[PASS]" in x for x in lines),
            "failed": sum("[FAIL]" in x for x in lines),
            "wall_s": time.perf_counter() - t, "tail": lines[-15:]}


def _smoke_checked(backend: str, out_dir: Path) -> dict[str, Any]:
    """:func:`_smoke`, raising (with the whole record) unless every check passed."""
    r = _smoke(backend, out_dir)
    if r["returncode"] or r["failed"] or not r["passed"]:
        raise RuntimeError(f"smoke_sim with MUJOCO_GL={backend} failed: {json.dumps(r)}")
    return r


def _section(name: str, fn: Callable[[], Any], failed: list[str]) -> dict[str, Any]:
    """Run one probe section. A failure is recorded with its traceback, not raised, so the
    sections after it still run; the probe's status then says which sections failed."""
    import traceback

    t = time.perf_counter()
    try:
        rec = {"ok": True, "value": fn(), "seconds": time.perf_counter() - t}
    except Exception as exc:  # recorded in the result and in its status
        failed.append(name)
        rec = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc(), "seconds": time.perf_counter() - t}
    # One JSON line per section on stdout, so the container log keeps the evidence even if
    # the job is later lost to a timeout or preemption.
    print(json.dumps({"probe_section": name, **rec}, default=str), flush=True)
    return rec


def probe_job(plan: dict[str, Any], *, out_dir: Path) -> dict[str, Any]:
    """Gate A probe: platform (KI-001), CUDA, EGL vs OSMesa rendering, training throughput.

    Every section (and every family of each training profile) runs separately through
    :func:`_section`. The status is ``complete`` only if all of them succeeded, otherwise
    ``partial`` with the failed sections listed.
    """
    import torch

    from roboforge.compute.profile import profile_training

    t0 = time.perf_counter()
    _set_git_env(plan)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    def cuda_info() -> dict[str, Any]:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True)
        return {"available": torch.cuda.is_available(), "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "nvidia_smi": smi.stdout.strip() or smi.stderr.strip()}

    cuda = _section("cuda", cuda_info, failed)
    render = {b: _section(f"render:{b}", lambda b=b: _smoke_checked(b, out_dir), failed)
              for b in ("egl", "osmesa")}
    p = plan["probe"]
    devices = ["cpu"] + (["cuda"] if cuda["ok"] and cuda["value"]["available"] else [])
    prof = {d: {f: _section(f"profile:{d}:{f}",
                            lambda d=d, f=f: profile_training(f, p["task"], d, p["steps"]),
                            failed)
                for f in p["families"]} for d in devices}
    result = {"status": "partial" if failed else "complete", "failed_sections": failed,
              "launch_id": plan["launch_id"], "host": host_info(), "cuda": cuda,
              "render": render, "training_profile": prof,
              "container_wall_s": time.perf_counter() - t0}
    (out_dir / "probe_result.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    """Simulation entrypoint: ``python -m roboforge.compute.remote train --plan P --result R``."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=["train"])
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--stop-at-step", type=int)
    args = ap.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    try:
        res = train_job(plan, stop_at_step=args.stop_at_step)
    except KeyboardInterrupt as exc:
        args.result.write_text(json.dumps({"status": "interrupted_reraised",
                                           "detail": str(exc)}) + "\n")
        return 130
    args.result.write_text(json.dumps(res, indent=1, default=str) + "\n")
    return 0 if res["status"] == "complete" else 3


if __name__ == "__main__":
    sys.exit(main())
