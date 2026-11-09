"""RGB training runs: plans (local), segments (remote or local) and run manifests (Session 8).

D-078, approval A-03. One run is one RGB config at one seed. It trains in **segments**: each
launch of ``train_gpu`` runs one segment until ``max_steps``, its deadline or a preemption
signal. Then it writes ``resume/latest.pt`` and the next launch (``mode: resume``) continues.

Run directory ``runs/<run_id>/`` (the Modal runs Volume, or local ``runs/`` for CPU smokes):
* ``manifest.json`` (``roboforge.rgb_run/1``) holds the config, the data hashes, the status,
  every cadence checkpoint (step, SHA-256, inference digest, validation loss) and every
  segment;
* ``segments.jsonl`` adds one line per started segment: launch ID, attempt, host, cloud,
  region, GPU, driver and determinism flags (the amendment's provenance);
* ``metrics.jsonl`` holds loss and gradient norm every ``log_every`` steps;
* ``checkpoints/step_<k>.pt`` and ``resume/latest.pt``.

The attempt cap holds as for ``train_cpu``: a launch that has already started
``max_attempts_per_launch`` segments returns without training.

    uv run --frozen python -m roboforge.compute.rgb_train plan --family diffusion --seed 0 \
        --prefix s8 --timeout-s 3600 --out plan.json --record
    uv run --frozen python -m roboforge.compute.rgb_train local --family bc_chunk --seed 0 \
        --prefix s8smoke --device cpu --set train.max_steps=20 --set train.ckpt_every=10
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from roboforge.compute import cost_log
from roboforge.compute.plan import _base, _checks, summary
from roboforge.compute.pricing import load_modal_config
from roboforge.paths import REPO_ROOT

FUNCTION = "train_gpu"
RUN_SCHEMA = "roboforge.rgb_run/1"


def _overrides(items: list[str]) -> dict[str, Any]:
    import yaml

    out = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def rgb_manifest(relpath: str) -> dict[str, Any]:
    """The committed manifest of an installed RGB dataset."""
    name = Path(relpath).name.replace(".hdf5", "")          # rgb84_v1
    task = Path(relpath).parent.name
    p = REPO_ROOT / "data" / "manifests" / "rgb" / f"{task}_{name}.json"
    if not p.is_file():
        raise FileNotFoundError(f"{p}: RGB dataset not installed (make rgb-dataset-install)")
    return json.loads(p.read_text())


def run_context(cfg: Any, run_id: str) -> dict[str, Any]:
    """Normalizers and every data hash the run trains against (Level 1's, plus the RGB file)."""
    from roboforge.train.run import data_context

    ctx = data_context(cfg["task"], run_id)
    m = rgb_manifest(cfg["data"]["rgb_relpath"])
    if m["processed_content_sha256"] != ctx["hashes"]["processed_content_sha256"]:
        raise ValueError("RGB dataset pairs with a different low-dim dataset")
    ctx["hashes"].update(rgb_content_sha256=m["content_sha256"], rgb_file_sha256=m["file_sha256"],
                         rgb_relpath=m["relpath"])
    return ctx


def train_plan(family: str, seed: int, prefix: str, timeout_s: int | None, mode: str = "new",
               overrides: list[str] | None = None,
               events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Plan of one ``train_gpu`` launch (one segment of one run)."""
    from roboforge.data.pipeline import load_task_artifacts
    from roboforge.data.processed import processed_path, processed_relpath
    from roboforge.provenance import sha256_file
    from roboforge.train.rgb import RGBConfig, config_path

    ov = {"seed": seed, **_overrides(overrides or [])}
    path = config_path(family)
    cfg = RGBConfig.load(path, ov)
    rid = cfg.run_id(prefix)
    mcfg = load_modal_config()
    plan = _base(mcfg, FUNCTION, timeout_s or mcfg["functions"][FUNCTION]["default_timeout_s"])
    tm = load_task_artifacts(cfg["task"])["task_manifest"]["processed"]
    rm = rgb_manifest(cfg["data"]["rgb_relpath"])
    plan.update(mode=mode, run_id=rid, prefix=prefix, family=family,
                config_path=path.relative_to(REPO_ROOT).as_posix(), overrides=ov,
                config=cfg.raw, config_sha256=cfg.sha256(),
                data={"task": cfg["task"],
                      "processed_relpath": processed_relpath(cfg["task"], tm["converter_version"]),
                      "processed_content_sha256": tm["content_sha256"],
                      "processed_file_sha256": sha256_file(processed_path(
                          cfg["task"], tm["converter_version"])),
                      "rgb_relpath": rm["relpath"], "rgb_content_sha256": rm["content_sha256"],
                      "rgb_file_sha256": rm["file_sha256"]})
    plan["blocking"] = _checks(mode, rid, events if events is not None else
                               cost_log.read_events())
    return plan


def segment_host() -> dict[str, Any]:
    """Where this segment runs: the amendment's per-segment provenance."""
    import subprocess

    import torch

    smi = None
    if torch.cuda.is_available():
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True)
        smi = r.stdout.strip() or r.stderr.strip()
    env = {k: os.environ.get(k) for k in ("MODAL_REGION", "MODAL_CLOUD_PROVIDER",
                                          "MODAL_TASK_ID", "MODAL_IMAGE_ID")}
    return {"hostname": os.uname().nodename, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "nvidia_smi": smi, **env}


def _write_manifest(run_dir: Path, man: dict[str, Any]) -> None:
    tmp = run_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(man, indent=1, default=str) + "\n")
    os.replace(tmp, run_dir / "manifest.json")


def run_segment(*, family: str, seed: int, prefix: str, overrides: dict[str, Any],
                runs_root: Path, device: str, launch_id: str, mode: str,
                deadline_s: float | None, max_attempts: int | None = None,
                stop_at_step: int | None = None, commit: Any = None,
                expected: dict[str, Any] | None = None) -> dict[str, Any]:
    """Train one segment of one run; returns the segment result."""
    from roboforge.data.rgb_dataset import load_rgb_dataset
    from roboforge.paths import data_root
    from roboforge.provenance import git_revision, sha256_file
    from roboforge.train.rgb import RGBConfig, RGBTrainer, config_path, load_rgb_windows

    t0 = time.perf_counter()
    commit = commit or (lambda: None)
    cfg = RGBConfig.load(config_path(family), {"seed": seed, **overrides})
    rid = cfg.run_id(prefix)
    run_dir = runs_root / rid
    seg_file = run_dir / "segments.jsonl"
    prior = [json.loads(x) for x in seg_file.read_text().splitlines()] if seg_file.exists() else []
    attempt = sum(1 for s in prior if s["launch_id"] == launch_id)
    if max_attempts is not None and attempt >= max_attempts:
        return {"status": "refused", "run_id": rid, "attempt": attempt,
                "container_wall_s": time.perf_counter() - t0,
                "detail": f"attempt cap: {attempt} segments already started by this launch"}
    if mode == "new" and prior and not any(s["launch_id"] == launch_id for s in prior):
        raise FileExistsError(f"{run_dir} already has segments; a new run never reuses it")
    if mode == "resume" and not (run_dir / "resume" / "latest.pt").is_file():
        raise FileNotFoundError(f"{run_dir}: nothing to resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = run_context(cfg, rid)
    h = ctx["hashes"]
    if expected is not None:  # the plan's hashes must be the ones on this machine
        for k in ("rgb_content_sha256", "processed_content_sha256"):
            if expected[k] != h[k]:
                raise ValueError(f"{k}: plan {expected[k]} != here {h[k]}")
    rgb_path = data_root() / h["rgb_relpath"]
    if sha256_file(rgb_path) != h["rgb_file_sha256"]:
        raise ValueError(f"{rgb_path}: file hash differs from the committed manifest")
    rgb = load_rgb_dataset(rgb_path, expected_sha256=h["rgb_content_sha256"])
    train = load_rgb_windows(cfg, "train", rgb)
    valid = load_rgb_windows(cfg, "valid", rgb)
    del rgb
    tr = RGBTrainer(cfg, run_dir, train, valid, ctx, device)
    resumed_from = tr.load_resume() if (run_dir / "resume" / "latest.pt").is_file() else 0
    seg = {"launch_id": launch_id, "attempt": attempt, "mode": mode, "device": device,
           "resumed_from_step": resumed_from, "utc_start": time.strftime(
               "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git_revision(),
           "flags": tr.flags, "host": segment_host()}
    with seg_file.open("a") as fh:
        fh.write(json.dumps(seg, default=str) + "\n")
    man_path = run_dir / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {
        "schema": RUN_SCHEMA, "run_id": rid, "family": cfg["family"], "task": cfg["task"],
        "seed": int(cfg["seed"]), "config": cfg.raw, "config_sha256": cfg.sha256(),
        "hashes": h, "checkpoints": [], "segments": [], "status": "running"}
    man["segments"].append(seg)
    _write_manifest(run_dir, man)
    commit()

    stop = {"signal": None}

    def _sig(signum: int, frame: Any) -> None:
        stop["signal"] = signum

    installed = []
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _sig)
            installed.append(int(s))
        except ValueError:  # not the main thread
            pass
    end = None if deadline_s is None else t0 + deadline_s
    metrics = (run_dir / "metrics.jsonl").open("a")

    def log(r: dict[str, Any]) -> None:
        metrics.write(json.dumps(r) + "\n")
        metrics.flush()

    def on_ckpt(ck: dict[str, Any]) -> None:
        man["checkpoints"].append(ck)
        _write_manifest(run_dir, man)
        commit()
        print(json.dumps({"rgb_checkpoint": ck}), flush=True)

    t_train = time.perf_counter()
    status = tr.run(should_stop=lambda: stop["signal"] is not None or
                    (end is not None and time.perf_counter() >= end),
                    stop_at_step=stop_at_step, log=log, on_checkpoint=on_ckpt)
    metrics.close()
    seg.update(status=status, end_step=tr.step, train_wall_s=time.perf_counter() - t_train,
               signal=stop["signal"], steps_per_s=(tr.step - resumed_from) /
               max(1e-9, time.perf_counter() - t_train))
    man["segments"][-1] = seg
    man["status"] = "complete" if status == "complete" else "interrupted"
    man["step"] = tr.step
    _write_manifest(run_dir, man)
    commit()
    return {"status": "complete" if status == "complete" else "interrupted", "run_id": rid,
            "attempt": attempt, "end_step": tr.step, "resumed_from_step": resumed_from,
            "segment": seg, "signals_installed": installed,
            "container_wall_s": time.perf_counter() - t0}


def rgb_train_job(plan: dict[str, Any], *, runs_root: Path, commit: Any = None) -> dict[str, Any]:
    """Remote side of a ``train_gpu`` launch."""
    from roboforge.compute.remote import _set_git_env

    _set_git_env(plan)
    return run_segment(family=plan["family"], seed=int(plan["overrides"]["seed"]),
                       prefix=plan["prefix"],
                       overrides={k: v for k, v in plan["overrides"].items() if k != "seed"},
                       runs_root=runs_root, device="cuda", launch_id=plan["launch_id"],
                       mode=plan["mode"], deadline_s=plan["deadline_s"],
                       max_attempts=int(plan["policy"]["max_attempts_per_launch"]),
                       commit=commit, expected=plan["data"])


def verify_run_dir(run_dir: Path) -> dict[str, Any]:
    """A fetched run directory: manifest present, every cadence checkpoint hash-equal."""
    from roboforge.provenance import sha256_file

    man = json.loads((Path(run_dir) / "manifest.json").read_text())
    bad = [c["path"] for c in man["checkpoints"]
           if sha256_file(Path(run_dir) / c["path"]) != c["sha256"]]
    if bad:
        raise ValueError(f"{run_dir}: checkpoint hash mismatch: {bad}")
    return {"run_id": man["run_id"], "status": man["status"], "step": man.get("step"),
            "checkpoints": len(man["checkpoints"]), "segments": len(man["segments"]),
            "verified": True}


def main(argv: list[str] | None = None) -> int:
    """CLI: ``plan`` (a launch plan), ``local`` (a segment here) or ``verify`` (a fetched run)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "local"):
        s = sub.add_parser(name)
        s.add_argument("--family", choices=["bc_chunk", "diffusion"], required=True)
        s.add_argument("--seed", type=int, required=True)
        s.add_argument("--prefix", required=True)
        s.add_argument("--set", dest="overrides", action="append", default=[])
        s.add_argument("--mode", choices=["new", "resume"], default="new")
    pl = sub.choices["plan"]
    pl.add_argument("--timeout-s", type=int)
    pl.add_argument("--out", type=Path, required=True)
    pl.add_argument("--record", action="store_true")
    lo = sub.choices["local"]
    lo.add_argument("--device", default="cpu")
    lo.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs")
    lo.add_argument("--stop-at-step", type=int)
    ve = sub.add_parser("verify")
    ve.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args(argv)
    if args.cmd == "verify":
        print(json.dumps(verify_run_dir(args.run_dir), indent=1))
        return 0
    if args.cmd == "plan":
        plan = train_plan(args.family, args.seed, args.prefix, args.timeout_s, args.mode,
                          args.overrides)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan, indent=1, sort_keys=True, default=str) + "\n")
        if args.record:
            cost_log.append_event("dry_run", {**summary(plan), "blocking": plan["blocking"],
                                              "cost_usd": 0.0})
        print(json.dumps({"plan": str(args.out), "run_id": plan["run_id"],
                          "estimate_upper_usd": round(plan["estimate"]["total_usd"], 4),
                          "blocking": plan["blocking"]}, indent=1))
        return 0
    import uuid

    res = run_segment(family=args.family, seed=args.seed, prefix=args.prefix,
                      overrides=_overrides(args.overrides), runs_root=args.runs_root,
                      device=args.device, launch_id=f"local-{uuid.uuid4().hex[:12]}",
                      mode=args.mode, deadline_s=None, stop_at_step=args.stop_at_step)
    print(json.dumps({k: res[k] for k in ("status", "run_id", "end_step",
                                          "resumed_from_step")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
