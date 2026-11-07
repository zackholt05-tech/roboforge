"""RoboForge on Modal: budget-gated training and the Gate A probe (Session 5; D-045..D-047).

Run with a Python that has the Modal SDK, which is kept **outside** the project venv
(D-045; e.g. ``/opt/anaconda3/bin/python``). This file never imports RoboForge locally. It
asks the project venv (``uv run --frozen python -m roboforge.compute.plan``) for plans,
the approval gate and the cost ledger.

    # DRY RUN (the default): prints the plan and the upper-bound estimate, logs a dry_run
    # event, contacts no Modal server and creates nothing.
    python infra/modal_app.py train --config configs/models/diffusion_unet.yaml --task lift \
        --seed 0 --prefix s5modal --set train.max_steps=2000 --timeout-s 1800
    python infra/modal_app.py probe

    # PAID LAUNCH: only with an owner approval recorded in docs/EXPERIMENT_LEDGER.md.
    python infra/modal_app.py train ... --budget-approved A-01
    python infra/modal_app.py probe --budget-approved A-01

    # Hash-validated download into runs/ (reads the runs Volume; creates nothing).
    python infra/modal_app.py fetch --run-id <run_id>
    # Kill path for a running launch (also: Ctrl-C in the launching terminal).
    python infra/modal_app.py cancel --app-id <ap-...>

Launch policy, fixed for every function (``configs/compute/modal.yaml``):

* ``max_containers=1``, ``retries=0``, ``single_use_containers=True``;
* ``scaledown_window=2`` s and an explicit per-launch timeout (``with_options``);
* preemptible, with preemption handled by resumable training: the same launch ID resumes;
* one GPU at most (``gpu_probe``), never multi-GPU;
* a new run ID is never reused, and a duplicate launch is refused both by the gate and inside
  the job (``launch.json``).

Remote layout (identical in ``scripts/modal_sim.py``):

* code mounted at ``/app`` (``src``, ``scripts``, ``configs``, ``data/{manifests,splits,
  normalization}``);
* the ``roboforge-runs`` Volume at ``/app/runs``;
* the ``roboforge-data`` Volume at ``/data`` (``ROBOFORGE_DATA_ROOT``);
* the image is built from ``uv.lock`` with ``uv sync --frozen --no-dev``. This Modal image
  is a different platform from the canonical evaluation container (D-017). It never produces
  canonical evaluation evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal
import yaml

LOCAL = modal.is_local()
REPO = Path(__file__).resolve().parents[1] if LOCAL else Path("/app")
CFG = yaml.safe_load((REPO / "configs" / "compute" / "modal.yaml").read_text())
APP = CFG["app"]
POL = CFG["policy"]
FN = CFG["functions"]


def _image() -> modal.Image:
    img = (modal.Image.debian_slim(python_version=APP["python"])
           .apt_install(*APP["apt"])
           .uv_sync(str(REPO), frozen=True, extra_options="--no-dev",
                    uv_version=APP["uv_version"])
           .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
                 "PYTHONPATH": f"{APP['repo_root']}/src",
                 "ROBOFORGE_DATA_ROOT": APP["volumes"]["data"]["mount"]}))
    for d in APP["code_dirs"]:
        img = img.add_local_dir(str(REPO / d), f"{APP['repo_root']}/{d}",
                                ignore=["**/__pycache__", "**/*.pyc"])
    return img


app = modal.App(APP["name"])
image = _image()
data_vol = modal.Volume.from_name(APP["volumes"]["data"]["name"])
runs_vol = modal.Volume.from_name(APP["volumes"]["runs"]["name"])
VOLUMES = {APP["volumes"]["data"]["mount"]: data_vol, APP["volumes"]["runs"]["mount"]: runs_vol}
COMMON = {"image": image, "max_containers": POL["max_containers"], "retries": POL["retries"],
          "scaledown_window": POL["scaledown_window_s"], "single_use_containers": True,
          "nonpreemptible": False}


@app.function(volumes=VOLUMES, cpu=FN["train_cpu"]["cpu"],
              memory=FN["train_cpu"]["memory_mib"],
              timeout=FN["train_cpu"]["default_timeout_s"], **COMMON)
def train_cpu(plan: dict[str, Any]) -> dict[str, Any]:
    """Train or continue one run on CPU (1 torch thread), resumable on preemption."""
    from roboforge.compute.remote import train_job

    runs_vol.reload()  # see the latest committed state (a preemption restart resumes it)
    res = train_job(plan, commit=runs_vol.commit)
    res["modal"] = {"function": "train_cpu"}
    return res


@app.function(volumes={APP["volumes"]["data"]["mount"]: data_vol}, gpu=FN["gpu_probe"]["gpu"],
              cpu=FN["gpu_probe"]["cpu"], memory=FN["gpu_probe"]["memory_mib"],
              timeout=FN["gpu_probe"]["default_timeout_s"], **COMMON)
def gpu_probe(plan: dict[str, Any]) -> dict[str, Any]:
    """Gate A: platform (KI-001), CUDA, EGL vs OSMesa, CPU and GPU training throughput."""
    from roboforge.compute.remote import probe_job

    res = probe_job(plan, out_dir=Path("/tmp/roboforge_probe"))
    res["modal"] = {"function": "gpu_probe"}
    return res


# --------------------------------------------------------------------------- local side
def _venv(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["uv", "run", "--frozen", "python", "-m", *args], cwd=REPO,
                          capture_output=True, text=True)


def _plan(cmd: list[str], out: Path) -> dict[str, Any]:
    p = _venv("roboforge.compute.plan", *cmd, "--out", str(out), "--record")
    if p.returncode:
        raise SystemExit(f"plan failed:\n{p.stdout}\n{p.stderr}")
    return json.loads(out.read_text())


def _print_plan(plan: dict[str, Any]) -> None:
    keep = ("function", "mode", "run_id", "launch_id", "timeout_s", "deadline_s", "resources",
            "policy", "volumes", "git", "blocking")
    print(json.dumps({k: plan.get(k) for k in keep}, indent=1))
    print(json.dumps({"image": plan["image"]}, indent=1))
    est = plan["estimate"]
    print(f"UPPER-BOUND ESTIMATE ${est['total_usd']:.4f} (list prices fetched "
          f"{est['pricing_fetched_utc']}; job ${est['job']['total_usd']:.4f} = "
          f"{est['attempts']} attempts x ${est['one_attempt']['total_usd']:.4f} + build/start "
          f"allowance ${est['build_allowance']['total_usd']:.4f}; not a cap)")


def _upload_data(plan: dict[str, Any]) -> dict[str, Any]:
    """Create both Volumes if needed and upload the plan's processed file (paid path only).

    Runs **before** ``app.run()``: starting the app looks the Volumes up by name, so they
    must exist by then."""
    vol = modal.Volume.from_name(APP["volumes"]["data"]["name"], create_if_missing=True)
    vol.hydrate()
    modal.Volume.from_name(APP["volumes"]["runs"]["name"], create_if_missing=True).hydrate()
    rel = plan["data"]["processed_relpath"] if plan.get("data") else \
        "processed/robomimic_v1.5_ph/lift/lowdim_v1.hdf5"
    local = REPO / "data" / rel
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(local), "/" + rel)
    return {"uploaded": rel, "bytes": local.stat().st_size}


def _launch(plan_path: Path, approval: str) -> int:
    plan = json.loads(plan_path.read_text())
    auth = _venv("roboforge.compute.plan", "authorize", "--plan", str(plan_path),
                 "--approval", approval)
    print(auth.stdout, auth.stderr)
    if auth.returncode:
        return 2
    fn = train_cpu if plan["function"] == "train_cpu" else gpu_probe
    result: dict[str, Any]
    app_id = None  # read while the app runs: app.app_id is unset again after it stops
    t0 = time.time()
    try:
        upload = _upload_data(plan)
        with modal.enable_output(), app.run():
            app_id = app.app_id
            result = fn.with_options(timeout=plan["timeout_s"]).remote(plan)
            result.setdefault("modal", {}).update(app_id=app_id, upload=upload)
    except BaseException as exc:  # incl. KeyboardInterrupt (Ctrl-C, or re-raised remotely)
        result = {"status": "error", "detail": f"{type(exc).__name__}: {exc}",
                  "modal": {"app_id": app_id}}
        reraise = None if isinstance(exc, Exception) else exc
    else:
        reraise = None
    # Written and logged in every case, so no launch event is left without its result.
    result["client_wall_s"] = time.time() - t0
    out = plan_path.with_suffix(".result.json")
    out.write_text(json.dumps(result, indent=1, default=str) + "\n")
    rec = _venv("roboforge.compute.plan", "record-result", "--plan", str(plan_path),
                "--result", str(out))
    print(rec.stdout[-2000:], rec.stderr[-2000:])
    print(f"result: {out}")
    if reraise is not None:
        raise reraise
    return 0 if result.get("status") == "complete" else 1


def _fetch(run_id: str, staging_root: Path) -> int:
    vol = modal.Volume.from_name(APP["volumes"]["runs"]["name"])
    staged = staging_root / run_id
    if staged.exists():
        raise SystemExit(f"{staged} exists")
    for entry in vol.iterdir(run_id, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        dest = staging_root / entry.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            vol.read_file_into_fileobj(entry.path, fh)
    p = _venv("roboforge.compute.retrieve", "--staged", str(staged), "--run-id", run_id)
    print(p.stdout, p.stderr)
    return p.returncode


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--task")
    t.add_argument("--seed")
    t.add_argument("--prefix", required=True)
    t.add_argument("--set", dest="overrides", action="append", default=[])
    t.add_argument("--timeout-s")
    t.add_argument("--mode", choices=["new", "resume"], default="new")
    t.add_argument("--simulate-preemption-s")
    t.add_argument("--stop-at-step")
    pr = sub.add_parser("probe")
    pr.add_argument("--timeout-s")
    pr.add_argument("--steps", default="300")
    for s in (t, pr):
        s.add_argument("--budget-approved", metavar="APPROVAL_ID",
                       help="launch for real under this recorded owner approval")
        s.add_argument("--plan-dir", type=Path, default=REPO / "logs" / "tmp" / "modal_plans")
    f = sub.add_parser("fetch")
    f.add_argument("--run-id", required=True)
    f.add_argument("--staging", type=Path, default=REPO / "logs" / "tmp" / "modal_fetch")
    c = sub.add_parser("cancel")
    c.add_argument("--app-id", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        return _fetch(args.run_id, args.staging)
    if args.cmd == "cancel":
        return subprocess.run([sys.executable, "-m", "modal", "app", "stop", "--yes",
                               args.app_id]).returncode
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out = args.plan_dir / f"{args.cmd}_{stamp}.json"
    if args.cmd == "train":
        cmd = ["train", "--config", args.config, "--prefix", args.prefix, "--mode", args.mode]
        for k in ("task", "seed", "timeout_s", "simulate_preemption_s", "stop_at_step"):
            if getattr(args, k):
                cmd += [f"--{k.replace('_', '-')}", str(getattr(args, k))]
        for o in args.overrides:
            cmd += ["--set", o]
    else:
        cmd = ["probe", "--steps", args.steps] + (["--timeout-s", args.timeout_s]
                                                 if args.timeout_s else [])
    plan = _plan(cmd, out)
    _print_plan(plan)
    if not args.budget_approved:
        print(f"DRY RUN: no Modal call was made and nothing was created. Plan: {out}")
        return 0
    if plan["blocking"]:
        print("REFUSED: " + "; ".join(plan["blocking"]), file=sys.stderr)
        return 2
    return _launch(out, args.budget_approved)


if __name__ == "__main__":
    sys.exit(main())
