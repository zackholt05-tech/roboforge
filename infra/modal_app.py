"""RoboForge on Modal: budget-gated training, the Gate A probe and the Gate D RGB probe.

Sessions 5 and 8; DECISIONS D-045..D-047.

Run with a Python that has the Modal SDK, which is kept **outside** the project venv
(D-045; e.g. ``/opt/anaconda3/bin/python``). This file never imports RoboForge locally. It
asks the project venv (``uv run --frozen python -m roboforge.compute.plan``) for plans,
the approval gate and the cost ledger.

    # DRY RUN (the default): prints the plan and the upper-bound estimate, logs a dry_run
    # event, contacts no Modal server and creates nothing.
    python infra/modal_app.py train --config configs/models/diffusion_unet.yaml --task lift \
        --seed 0 --prefix s5modal --set train.max_steps=2000 --timeout-s 1800
    python infra/modal_app.py probe
    python infra/modal_app.py rgb-probe              # Session 8 Gate D: RGB timing, no data

    # PAID LAUNCH: only with an owner approval recorded in docs/EXPERIMENT_LEDGER.md.
    python infra/modal_app.py train ... --budget-approved A-01
    python infra/modal_app.py probe --budget-approved A-01
    python infra/modal_app.py rgb-probe --budget-approved A-02
    python infra/modal_app.py train-gpu --family diffusion --seed 0 --prefix s8 \
        --budget-approved A-03                  # Session 8 RGB study (one segment per launch)
    python infra/modal_app.py train-gpu-pc --seed 0 --prefix s9 \
        --budget-approved A-04                  # Session 9 Level 2 (one segment per launch)

    # Hash-validated download into runs/ (reads the runs Volume; creates nothing).
    python infra/modal_app.py fetch --run-id <run_id>
    python infra/modal_app.py fetch-rgb --run-id <run_id>   # RGB study run, hash-verified
    python infra/modal_app.py fetch-pc --run-id <run_id>    # Level 2 run, hash-verified
    python infra/modal_app.py attach --plan <plan.json>     # re-attach a detached launch (D-085)
    # Kill path for a running launch. Launches are detached (D-085): Ctrl-C stops only the
    # client, and a new launch is refused while any app of this project is still live.
    python infra/modal_app.py cancel --app-id <ap-...>

Launch policy, fixed for every function (``configs/compute/modal.yaml``):

* ``max_containers=1``, ``retries=0``, ``single_use_containers=True``;
* ``scaledown_window=2`` s and an explicit per-launch timeout (``with_options``);
* preemptible, with preemption handled by resumable training: the same launch ID resumes;
* one GPU at most (``gpu_probe``, ``rgb_probe``), never multi-GPU;
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


@app.function(volumes={APP["volumes"]["runs"]["mount"]: runs_vol}, gpu=FN["rgb_probe"]["gpu"],
              cpu=FN["rgb_probe"]["cpu"], memory=FN["rgb_probe"]["memory_mib"],
              timeout=FN["rgb_probe"]["default_timeout_s"], **COMMON)
def rgb_probe(plan: dict[str, Any]) -> dict[str, Any]:
    """Gate D: RGB policy training speed on CUDA; random inputs only (no data Volume)."""
    from roboforge.compute.rgb_probe import rgb_probe_job

    runs_vol.reload()  # see earlier attempts' markers (the attempt cap)
    res = rgb_probe_job(plan, attempts_root=Path(APP["volumes"]["runs"]["mount"]) / "_probes",
                        commit=runs_vol.commit)
    res.setdefault("modal", {})["function"] = "rgb_probe"
    return res


@app.function(volumes=VOLUMES, gpu=FN["train_gpu"]["gpu"], cpu=FN["train_gpu"]["cpu"],
              memory=FN["train_gpu"]["memory_mib"],
              timeout=FN["train_gpu"]["default_timeout_s"], **COMMON)
def train_gpu(plan: dict[str, Any]) -> dict[str, Any]:
    """RGB study: one training segment on one L4, deterministic kernels (D-078, A-03)."""
    from roboforge.compute.rgb_train import rgb_train_job

    runs_vol.reload()  # earlier segments of this run (resume, attempt cap)
    res = rgb_train_job(plan, runs_root=Path(APP["volumes"]["runs"]["mount"]),
                        commit=runs_vol.commit)
    res.setdefault("modal", {})["function"] = "train_gpu"
    return res


@app.function(volumes=VOLUMES, gpu=FN["train_gpu_pc"]["gpu"], cpu=FN["train_gpu_pc"]["cpu"],
              memory=FN["train_gpu_pc"]["memory_mib"],
              timeout=FN["train_gpu_pc"]["default_timeout_s"], **COMMON)
def train_gpu_pc(plan: dict[str, Any]) -> dict[str, Any]:
    """Level 2 (Session 9): one point-cloud training segment on one L4, deterministic kernels."""
    from roboforge.compute.pc_train import pc_train_job

    runs_vol.reload()  # earlier segments of this run (resume, attempt cap)
    res = pc_train_job(plan, runs_root=Path(APP["volumes"]["runs"]["mount"]),
                       commit=runs_vol.commit)
    res.setdefault("modal", {})["function"] = "train_gpu_pc"
    return res


FUNCTIONS = {"train_cpu": train_cpu, "gpu_probe": gpu_probe, "rgb_probe": rgb_probe,
             "train_gpu": train_gpu, "train_gpu_pc": train_gpu_pc}
NEEDS_DATA = {"train_cpu", "gpu_probe", "train_gpu", "train_gpu_pc"}  # rgb_probe reads no data


# --------------------------------------------------------------------------- local side
def _venv(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["uv", "run", "--frozen", "python", "-m", *args], cwd=REPO,
                          capture_output=True, text=True)


def _plan(module: str, cmd: list[str], out: Path) -> dict[str, Any]:
    p = _venv(module, *cmd, "--out", str(out), "--record")
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


def _ensure_volumes() -> None:
    """Create both Volumes if missing: ``app.run()`` looks every function's Volumes up."""
    for v in ("data", "runs"):
        modal.Volume.from_name(APP["volumes"][v]["name"], create_if_missing=True).hydrate()


def _upload_data(plan: dict[str, Any]) -> dict[str, Any]:
    """Create both Volumes if needed and upload the plan's processed file (paid path only).

    Runs **before** ``app.run()``: starting the app looks the Volumes up by name, so they
    must exist by then."""
    vol = modal.Volume.from_name(APP["volumes"]["data"]["name"], create_if_missing=True)
    vol.hydrate()
    modal.Volume.from_name(APP["volumes"]["runs"]["name"], create_if_missing=True).hydrate()
    d = plan.get("data") or {}
    rels = [d.get("processed_relpath", "processed/robomimic_v1.5_ph/lift/lowdim_v1.hdf5")]
    if d.get("rgb_relpath"):
        rels.append(d["rgb_relpath"])
    if d.get("pc_relpath"):
        rels.append(d["pc_relpath"])
    out = {}
    for rel in rels:
        local = REPO / "data" / rel
        size = local.stat().st_size
        # Large RGB files are uploaded once: skip a file already on the Volume with the same
        # size (the job re-checks every file's SHA-256 before training).
        try:
            same = any(e.path.lstrip("/") == rel and e.size == size
                       for e in vol.listdir(str(Path(rel).parent)))
        except Exception:  # missing directory
            same = False
        if not same:
            with vol.batch_upload(force=True) as batch:
                batch.put_file(str(local), "/" + rel)
        out[rel] = {"bytes": size, "uploaded": not same}
    return {"files": out}


def _live_apps() -> list[dict[str, Any]] | None:
    """This project's Modal apps that are not stopped, or ``None`` if Modal cannot be asked."""
    p = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json"],
                       capture_output=True, text=True)
    if p.returncode:
        return None
    try:
        rows = json.loads(p.stdout)
    except json.JSONDecodeError:
        return None
    return [r for r in rows if r.get("description") == APP["name"] and r.get("state") != "stopped"]


def _stop_app(app_id: str | None) -> None:
    """Stop a detached app that outlived its call, so the single-flight guard frees up."""
    if app_id and any(r["app_id"] == app_id for r in (_live_apps() or [])):
        subprocess.run([sys.executable, "-m", "modal", "app", "stop", "--yes", app_id])


def _await_call(call_id: str) -> Any:
    """The detached call's result, riding out client network outages (D-085)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "launch_wait", REPO / "src" / "roboforge" / "compute" / "launch_wait.py")
    lw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lw)
    return lw.await_result(lambda t: modal.FunctionCall.from_id(call_id).get(timeout=t),
                           log=lambda m: print(m, flush=True))


def _launch(plan_path: Path, approval: str) -> int:
    plan = json.loads(plan_path.read_text())
    # Single flight (D-085): a detached app keeps running when its client dies, so a new
    # launch must never start while any app of this project is still up.
    live = _live_apps()
    if live is None or live:
        print(f"REFUSED: {'cannot list Modal apps' if live is None else 'apps still live'}: "
              f"{live}", file=sys.stderr)
        return 2
    auth = _venv("roboforge.compute.plan", "authorize", "--plan", str(plan_path),
                 "--approval", approval)
    print(auth.stdout, auth.stderr)
    if auth.returncode:
        return 2
    if plan["function"] not in FUNCTIONS:
        raise SystemExit(f"unknown function {plan['function']!r}; known: {sorted(FUNCTIONS)}")
    fn = FUNCTIONS[plan["function"]]
    result: dict[str, Any]
    app_id = None  # read while the app runs: app.app_id is unset again after it stops
    t0 = time.time()
    try:
        if plan["function"] in NEEDS_DATA:
            upload = _upload_data(plan)
        else:
            _ensure_volumes()
            upload = None
        # Detached (D-085, KI-047): a client network loss no longer stops the remote job.
        with modal.enable_output(), app.run(detach=True):
            app_id = app.app_id
            call = fn.with_options(timeout=plan["timeout_s"]).spawn(plan)
            call_id = call.object_id
            plan_path.with_suffix(".call.json").write_text(json.dumps(
                {"function_call_id": call_id, "app_id": app_id,
                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
            print(f"function_call: {call_id}", flush=True)
            result = _await_call(call_id)
            result.setdefault("modal", {}).update(app_id=app_id, upload=upload,
                                                  function_call_id=call_id)
    except BaseException as exc:  # incl. KeyboardInterrupt (Ctrl-C, or re-raised remotely)
        result = {"status": "error", "detail": f"{type(exc).__name__}: {exc}",
                  "modal": {"app_id": app_id}}
        reraise = None if isinstance(exc, Exception) else exc
    else:
        reraise = None
        _stop_app(app_id)  # only after a result: an unfinished detached job keeps running
    return _finish(plan_path, result, t0, reraise)


def _finish(plan_path: Path, result: dict[str, Any], t0: float,
            reraise: BaseException | None) -> int:
    """Write and record a launch's result (every case, so no launch is left without one)."""
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


def _attach(plan_path: Path) -> int:
    """Re-attach to a detached launch whose client died (D-085): wait for the recorded call,
    then write and record its result. The ledger keeps the last result of a launch."""
    call = json.loads(plan_path.with_suffix(".call.json").read_text())
    t0 = time.time()
    try:
        result = _await_call(call["function_call_id"])
        result.setdefault("modal", {}).update(app_id=call["app_id"],
                                              function_call_id=call["function_call_id"],
                                              reattached=True)
    except Exception as exc:
        result = {"status": "error", "detail": f"{type(exc).__name__}: {exc}",
                  "modal": {"app_id": call["app_id"], "reattached": True}}
    else:
        _stop_app(call["app_id"])
    return _finish(plan_path, result, t0, None)


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


def _fetch_rgb(run_id: str, dest_root: Path, with_resume: bool) -> int:
    """Download one RGB-study (or Level 2) run directory into ``runs/``; verify every hash.

    Both studies share the run layout and the verifier (``rgb_train verify``).

    The resume file (~0.4 GB) stays on the Volume unless ``with_resume``. Files already
    present locally with the same size are not downloaded again."""
    vol = modal.Volume.from_name(APP["volumes"]["runs"]["name"])
    for entry in vol.iterdir(run_id, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        if entry.path.split("/")[1:2] == ["resume"] and not with_resume:
            continue
        dest = dest_root / entry.path
        if dest.exists() and dest.stat().st_size == entry.size and \
                "checkpoints/" in entry.path:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            vol.read_file_into_fileobj(entry.path, fh)
        tmp.replace(dest)
    p = _venv("roboforge.compute.rgb_train", "verify", "--run-dir", str(dest_root / run_id))
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
    tg = sub.add_parser("train-gpu")
    tg.add_argument("--family", choices=["bc_chunk", "diffusion"], required=True)
    tg.add_argument("--seed", required=True)
    tg.add_argument("--prefix", required=True)
    tg.add_argument("--mode", choices=["new", "resume"], default="new")
    tg.add_argument("--timeout-s")
    tg.add_argument("--set", dest="overrides", action="append", default=[])
    tp = sub.add_parser("train-gpu-pc")
    tp.add_argument("--seed", required=True)
    tp.add_argument("--prefix", required=True)
    tp.add_argument("--mode", choices=["new", "resume"], default="new")
    tp.add_argument("--timeout-s")
    tp.add_argument("--set", dest="overrides", action="append", default=[])
    rp = sub.add_parser("rgb-probe")
    rp.add_argument("--timeout-s")
    rp.add_argument("--steps", default="30")
    for s in (t, pr, rp, tg, tp):
        s.add_argument("--budget-approved", metavar="APPROVAL_ID",
                       help="launch for real under this recorded owner approval")
        s.add_argument("--plan-dir", type=Path, default=REPO / "logs" / "tmp" / "modal_plans")
    f = sub.add_parser("fetch")
    f.add_argument("--run-id", required=True)
    f.add_argument("--staging", type=Path, default=REPO / "logs" / "tmp" / "modal_fetch")
    for name in ("fetch-rgb", "fetch-pc"):
        fr = sub.add_parser(name)
        fr.add_argument("--run-id", required=True)
        fr.add_argument("--with-resume", action="store_true")
    at = sub.add_parser("attach", help="re-attach to a detached launch whose client died")
    at.add_argument("--plan", type=Path, required=True)
    c = sub.add_parser("cancel")
    c.add_argument("--app-id", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        return _fetch(args.run_id, args.staging)
    if args.cmd in ("fetch-rgb", "fetch-pc"):
        return _fetch_rgb(args.run_id, REPO / "runs", args.with_resume)
    if args.cmd == "attach":
        return _attach(args.plan)
    if args.cmd == "cancel":
        return subprocess.run([sys.executable, "-m", "modal", "app", "stop", "--yes",
                               args.app_id]).returncode
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out = args.plan_dir / f"{args.cmd.replace('-', '_')}_{stamp}.json"
    module = "roboforge.compute.plan"
    if args.cmd == "train":
        cmd = ["train", "--config", args.config, "--prefix", args.prefix, "--mode", args.mode]
        for k in ("task", "seed", "timeout_s", "simulate_preemption_s", "stop_at_step"):
            if getattr(args, k):
                cmd += [f"--{k.replace('_', '-')}", str(getattr(args, k))]
        for o in args.overrides:
            cmd += ["--set", o]
    elif args.cmd == "train-gpu":
        module = "roboforge.compute.rgb_train"
        cmd = ["plan", "--family", args.family, "--seed", args.seed, "--prefix", args.prefix,
               "--mode", args.mode] + (["--timeout-s", args.timeout_s] if args.timeout_s
                                       else [])
        for o in args.overrides:
            cmd += ["--set", o]
    elif args.cmd == "train-gpu-pc":
        module = "roboforge.compute.pc_train"
        cmd = ["plan", "--seed", args.seed, "--prefix", args.prefix, "--mode", args.mode] + (
            ["--timeout-s", args.timeout_s] if args.timeout_s else [])
        for o in args.overrides:
            cmd += ["--set", o]
    elif args.cmd == "probe":
        cmd = ["probe", "--steps", args.steps] + (["--timeout-s", args.timeout_s]
                                                 if args.timeout_s else [])
    else:
        module = "roboforge.compute.rgb_probe"
        cmd = ["plan", "--steps", args.steps] + (["--timeout-s", args.timeout_s]
                                                if args.timeout_s else [])
    plan = _plan(module, cmd, out)
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
