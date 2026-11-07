"""Cost and time ESTIMATES for one Modal job and for the main suite (Session 5).

    python -m roboforge.compute.estimate job --function train_cpu --max-minutes 60
    python -m roboforge.compute.estimate job --config configs/models/diffusion_unet.yaml \
        --gpu L4 --max-minutes 15
    python -m roboforge.compute.estimate suite --out reports/session5/cost/suite_projection.json

Every number is an estimate. ``job`` is the worst case for the stated minutes: list price for
the GPU, CPU and memory, the billed keep-alive and a build/start allowance. ``suite`` projects
the RESEARCH_PROTOCOL §6.3 main suite from **measured** inputs, each read from its evidence file
and recorded with that file's SHA-256 in the output:

* local single-thread training throughput: the Session 3–4 pilot run manifests;
* local training slowdown under concurrency: ``reports/session5/cost/train_concurrency.json``;
* canonical-container validation evaluation wall time and steps (Lift): the Session 3–4
  validation summaries;
* per-step simulation cost and parallel-container speed-up:
  ``reports/session5/rollout_profile/summary.json``;
* inference latency per decision in the container: ``reports/session{3,4}/cost_latency``;
* GPU throughput: the Gate A probe result, if one exists; otherwise a labelled scenario range.

Quantities with no measurement (Can episode lengths of trained policies, Modal x86 CPU speed
relative to the M1 Pro) enter as stated assumptions with low/high values. Each scenario
reports both.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

from roboforge.compute.pricing import Resources, cost_usd, launch_upper_bound, load_modal_config
from roboforge.paths import REPO_ROOT
from roboforge.provenance import git_revision, sha256_file

FAMILIES = ("bc_mlp", "bc_rnn", "bc_mlp_chunk", "diffusion_unet")
# The Gate A training smoke, trained on Modal CPU and downloaded (Session 5): its full-run
# rate, including offline metrics and checkpoints, gives the full-run/pure-step ratio.
MODAL_SMOKE_RUN = "s5modal2_diffusion_unet_lift_seed0_2fda84af"
PILOTS = {"bc_mlp": "s3_bc_mlp_lift_seed0_dbf312cd", "bc_rnn": "s3_bc_rnn_lift_seed0_2e7cffe3",
          "bc_mlp_chunk": "s3_bc_mlp_chunk_lift_seed0_bb8b4dd6",
          "diffusion_unet": "s4_diffusion_unet_lift_seed0_336e15e9"}
# Calls per control step that pay the measured latency: diffusion was timed per decision
# (one every T_a = 8 steps); the BC rows were timed per step (conservative for BC-MLP-chunk).
EXEC_HORIZON = {"bc_mlp": 1, "bc_rnn": 1, "bc_mlp_chunk": 1, "diffusion_unet": 8}
# Stated assumptions (no measurement exists yet); each scenario reports low and high.
ASSUME = {
    "can_steps_per_episode": {"low": 160, "high": 400,
                              "why": "no Can policy exists; low = most starts succeed near 1.4x "
                                     "the mean demo length (116), high = every episode runs to "
                                     "the 400-step horizon"},
    "modal_cpu_time_factor": {"low": 0.8, "high": 1.6,
                              "why": "fallback only, used if the Gate A probe's CPU profile or "
                                     "the Modal training smoke is missing: Modal x86_64 "
                                     "single-thread time relative to the M1 Pro"},
    "gpu_steps_per_s": {"low": {"diffusion_unet": 40, "bc_rnn": 60, "bc_mlp": 250,
                                "bc_mlp_chunk": 250},
                        "high": {"diffusion_unet": 150, "bc_rnn": 250, "bc_mlp": 800,
                                 "bc_mlp_chunk": 800},
                        "why": "kernel-launch-bound small networks at batch 100; unmeasured "
                               "until the Gate A probe"},
    "failed_run_reserve": {"value": 0.25, "why": "25 % extra training and evaluation compute "
                                                 "for failed or repeated runs"},
    "modal_start_overhead_s": {"value": 120, "why": "container start and data/code mount per "
                                                    "launch (estimate)"},
    "render_clips": {"value": 40, "frames": 400, "cameras": 2,
                     "why": "Session 10 website clips from selected rollouts (estimate)"},
}


def _src(path: Path) -> dict[str, str]:
    return {"path": Path(path).relative_to(REPO_ROOT).as_posix(), "sha256": sha256_file(path)}


def measured_inputs() -> dict[str, Any]:
    """Read every measured input from its evidence file."""
    sources: list[dict[str, str]] = []
    train = {}
    for fam, rid in PILOTS.items():
        mp = REPO_ROOT / "runs" / rid / "manifest.json"
        m = json.loads(mp.read_text())
        train[fam] = {"steps_per_s": m["timing"]["steps_per_s"],
                      "wall_s_pilot": m["timing"]["wall_s"],
                      "max_steps": m["config"]["train"]["max_steps"],
                      "ckpt_bytes": (REPO_ROOT / m["checkpoints"][-1]["path"]).stat().st_size}
        sources.append(_src(mp))
    lift_eval: dict[str, list[dict[str, float]]] = {f: [] for f in FAMILIES}
    for f in sorted(glob.glob(str(REPO_ROOT / "reports/session[34]/validation/*.json"))):
        s = json.loads(Path(f).read_text())
        fam = s["train_run_id"].split("_lift_")[0].split("_", 1)[1]
        lift_eval[fam].append({"wall_s": s["wall_s"], "steps": s["steps_total"]})
        sources.append(_src(Path(f)))
    latency = {}
    for f in ("reports/session3/cost_latency/linux-aarch64-container-1f685638f28c.json",
              "reports/session4/cost_latency/linux-aarch64-container-1f685638f28c.json"):
        d = json.loads((REPO_ROOT / f).read_text())
        sources.append(_src(REPO_ROOT / f))
        for row in d["rows"]:  # median observe + act per call, 1 thread, container
            if row["name"] in FAMILIES:
                latency[row["name"]] = float(row["inference"]["observe_plus_act_ms"]["median"])
    prof_p = REPO_ROOT / "reports/session5/rollout_profile/summary.json"
    prof = json.loads(prof_p.read_text())
    sources.append(_src(prof_p))
    conc_p = REPO_ROOT / "reports/session5/cost/train_concurrency.json"
    conc = json.loads(conc_p.read_text())
    sources.append(_src(conc_p))
    probe_p = REPO_ROOT / "reports/session5/modal/probe_result.json"
    probe = json.loads(probe_p.read_text()) if probe_p.is_file() else None
    if probe:
        sources.append(_src(probe_p))
    smoke_p = REPO_ROOT / "runs" / MODAL_SMOKE_RUN / "manifest.json"
    modal_smoke = None
    if smoke_p.is_file():
        m = json.loads(smoke_p.read_text())
        modal_smoke = {"run_id": MODAL_SMOKE_RUN, "steps_per_s": m["timing"]["steps_per_s"],
                       "n_segments": m["timing"]["n_segments"],
                       "max_steps": m["config"]["train"]["max_steps"]}
        sources.append(_src(smoke_p))
    return {"train": train, "lift_eval": lift_eval, "latency_ms_per_decision": latency,
            "rollout_profile": prof, "train_concurrency": conc, "gpu_probe": probe,
            "modal_smoke": modal_smoke,
            "sources": sources}


def _profile_row(prof: dict[str, Any], job: str, p: int) -> dict[str, Any]:
    return next(r for r in prof["levels"] if r["job"] == job and r["P"] == p)


def probe_steps_per_s(probe: dict[str, Any] | None, device: str,
                      family: str) -> float | None:
    """Measured pure-step rate of ``family`` on ``device`` in the Gate A probe, if that
    section succeeded."""
    sec = ((probe or {}).get("training_profile") or {}).get(device, {}).get(family)
    return float(sec["value"]["steps_per_s_median"]) if sec and sec.get("ok") else None


def gpu_steps_per_s(probe: dict[str, Any] | None, family: str) -> float | None:
    """Measured CUDA steps/s of ``family`` from the Gate A probe, if that section succeeded."""
    return probe_steps_per_s(probe, "cuda", family)


def full_run_ratio(inp: dict[str, Any]) -> float | None:
    """Modal full-run rate / pure-step rate, both measured for diffusion: the training
    smoke (offline metrics and checkpoints included) over the probe's CPU profile."""
    smoke = inp.get("modal_smoke")
    pure = probe_steps_per_s(inp.get("gpu_probe"), "cpu", "diffusion_unet")
    return smoke["steps_per_s"] / pure if smoke and pure else None


def suite_counts(tuning: str, steps: int, ckpt_every: int) -> dict[str, Any]:
    """Runs and evaluations of the main suite (4 families x 2 tasks)."""
    configs = {"max": 4, "none": 1}[tuning]
    runs_per = configs + 2  # each configuration at seed 0, then seeds 1-2 of the chosen one
    n_ck = steps // ckpt_every
    runs = len(FAMILIES) * 2 * runs_per
    return {"families": list(FAMILIES), "tasks": ["lift", "can"], "seeds": [0, 1, 2],
            "configs_per_family_task": configs, "runs_per_family_task": runs_per,
            "training_runs": runs, "steps_per_run": steps, "checkpoints_per_run": n_ck,
            "validation_evaluations": runs * n_ck, "test_evaluations": len(FAMILIES) * 2 * 3}


def project(inp: dict[str, Any], tuning: str, steps: int, ckpt_every: int,
            k_train: int, p_eval: int) -> dict[str, Any]:
    """Local wall time and core-hours for one suite variant (low/high)."""
    c = suite_counts(tuning, steps, ckpt_every)
    per = c["runs_per_family_task"] * 2  # runs per family over both tasks
    conc = inp["train_concurrency"]
    slow = conc["slowdown_at_k"][str(k_train)] if str(k_train) in conc["slowdown_at_k"] else \
        max(conc["slowdown_at_k"].values())
    train_core_s = {f: per * steps / inp["train"][f]["steps_per_s"] for f in FAMILIES}
    total_core_s = sum(train_core_s.values())
    reserve = 1 + ASSUME["failed_run_reserve"]["value"]
    prof = inp["rollout_profile"]
    lift_p1 = _profile_row(prof, "lift_diffusion", 1)
    can_p1 = _profile_row(prof, "can_random", 1)
    sim_ms_lift = 1e3 * (lift_p1["containers"][0]["profile"]["eval_wall_s"] -
                         lift_p1["containers"][0]["profile"]["timed_s"]["inference"]) / \
        lift_p1["containers"][0]["profile"]["n_steps"]
    can_prof = can_p1["containers"][0]["profile"]
    sim_ms_can = 1e3 * (can_prof["eval_wall_s"] - can_prof["timed_s"]["inference"]) / \
        can_prof["n_steps"]
    best = _profile_row(prof, "lift_diffusion", p_eval)
    speed = best["evaluations_per_hour"] / lift_p1["evaluations_per_hour"]
    out = {"counts": c, "k_train": k_train, "train_slowdown_at_k": slow, "p_eval": p_eval,
           "eval_parallel_speedup": speed, "sim_ms_per_step": {"lift": sim_ms_lift,
                                                               "can": sim_ms_can}}
    for bound in ("low", "high"):
        can_steps = ASSUME["can_steps_per_episode"][bound] * 50
        eval_s = 0.0
        for f in FAMILIES:
            lat = inp["latency_ms_per_decision"][f] / EXEC_HORIZON[f]
            lift = inp["lift_eval"][f]
            lift_wall = (max if bound == "high" else min)(e["wall_s"] for e in lift)
            can_wall = can_steps * (sim_ms_can + lat) / 1e3
            n_val = c["runs_per_family_task"] * c["checkpoints_per_run"]
            eval_s += n_val * (lift_wall + can_wall) + 3 * (lift_wall + can_wall)  # + test
        out[bound] = {"train_core_h": reserve * total_core_s / 3600,
                      "train_wall_h": reserve * total_core_s * slow / k_train / 3600,
                      "eval_container_h": reserve * eval_s / 3600,
                      "eval_wall_h": reserve * eval_s / speed / 3600}
        out[bound]["total_wall_h_sequential_phases"] = (out[bound]["train_wall_h"] +
                                                        out[bound]["eval_wall_h"])
    out["train_core_h_by_family"] = {f: reserve * v / 3600 for f, v in train_core_s.items()}
    return out


def modal_training_cost(inp: dict[str, Any], proj: dict[str, Any], cfg: dict[str, Any],
                        device: str) -> dict[str, Any]:
    """List-price cost of training the suite on Modal (evaluation stays local), low/high."""
    pricing, c = cfg["pricing"], proj["counts"]
    per = c["runs_per_family_task"] * 2
    reserve = 1 + ASSUME["failed_run_reserve"]["value"]
    start = ASSUME["modal_start_overhead_s"]["value"]
    out: dict[str, Any] = {"device": device}
    probe = inp.get("gpu_probe")
    ratio = full_run_ratio(inp)
    measured: list[str] = []
    for bound in ("low", "high"):
        usd, hours = 0.0, 0.0
        for f in FAMILIES:
            # Low bound: the measured pure-step rate. High bound: that rate times the measured
            # full-run/pure-step ratio (offline metrics, checkpoints, slower containers).
            sps = probe_steps_per_s(probe, device, f)
            if sps is not None and ratio is not None:
                rate = sps if bound == "low" else sps * ratio
                if bound == "low":
                    measured.append(f)
            elif device == "cpu":
                fac = ASSUME["modal_cpu_time_factor"]["low" if bound == "low" else "high"]
                rate = inp["train"][f]["steps_per_s"] / fac
            else:
                rate = ASSUME["gpu_steps_per_s"]["high" if bound == "low" else "low"][f]
            secs = c["steps_per_run"] / rate
            if device == "cpu":
                fn = cfg["functions"]["train_cpu"]
                res = Resources(secs + start, fn["cpu"], fn["memory_mib"] / 1024)
            else:
                fn = cfg["functions"]["gpu_probe"]
                res = Resources(secs + start, fn["cpu"], fn["memory_mib"] / 1024, fn["gpu"])
            usd += per * cost_usd(res, pricing)["total_usd"]
            hours += per * res.seconds / 3600
        build = cost_usd(Resources(cfg["build_allowance"]["seconds"],
                                   cfg["build_allowance"]["cpu"],
                                   cfg["build_allowance"]["memory_gib"]), pricing)["total_usd"]
        out[bound] = {"usd": reserve * usd + build, "container_h": reserve * hours,
                      "sequential_wall_h_max_containers_1": reserve * hours}
    out["full_run_ratio"] = ratio
    out["rate_source"] = (
        f"measured on Modal ({', '.join(measured)}): Gate A probe pure-step rate (low) and "
        f"x {ratio:.3f} full-run ratio of the Modal training smoke (high)"
        + ("" if len(measured) == len(FAMILIES) else "; other families: stated assumption")
        if measured else "stated assumption (no Modal measurement)")
    return out


def storage(inp: dict[str, Any], proj: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Checkpoint and record storage (local disk; Modal Volume is free below 1 TiB)."""
    c = proj["counts"]
    per = c["runs_per_family_task"] * 2
    ck = sum(per * c["checkpoints_per_run"] * inp["train"][f]["ckpt_bytes"] for f in FAMILIES)
    rec_dir = REPO_ROOT / "runs" / ("s4_diffusion_unet_lift_seed0_336e15e9__val_step0015000__"
                                    "linux-aarch64-container-1f685638f28c__ps0r0")
    rec_bytes = sum(p.stat().st_size for p in rec_dir.rglob("*") if p.is_file())
    steps = json.loads((rec_dir / "manifest.json").read_text())["steps_total"]
    per_step = rec_bytes / steps
    n_eval = c["validation_evaluations"] + c["test_evaluations"]
    rec = {b: n_eval / 2 * per_step * (3000 + ASSUME["can_steps_per_episode"][b] * 50)
           for b in ("low", "high")}
    gib = {b: (ck + rec[b]) / 2**30 for b in ("low", "high")}
    return {"checkpoint_gib": ck / 2**30, "records_gib": {b: v / 2**30 for b, v in rec.items()},
            "total_gib": gib, "modal_volume_usd_per_month": 0.0 if max(gib.values()) <
            cfg["pricing"]["volume_free_gib"] else None,
            "record_bytes_per_step_measured": per_step}


def rendering(inp: dict[str, Any]) -> dict[str, Any]:
    """OSMesa rendering time for the website clips (CPU; estimate)."""
    r = _profile_row(inp["rollout_profile"], "lift_diffusion", 1)["containers"][0]["profile"]
    a = ASSUME["render_clips"]
    frames = a["value"] * a["frames"] * a["cameras"]
    return {"frames": frames, "seconds_per_frame_measured": r["rendering"]["median_s"],
            "core_h": frames * r["rendering"]["median_s"] / 3600,
            "backend": r["rendering"]["backend"]}


def suite(out: Path) -> dict[str, Any]:
    """All scenarios of the main-suite projection."""
    cfg = load_modal_config()
    inp = measured_inputs()
    conc = inp["train_concurrency"]
    k = int(conc["recommended_k"])
    p_eval = max((r["P"] for r in inp["rollout_profile"]["levels"]
                  if r["job"] == "lift_diffusion"),
                 key=lambda p: _profile_row(inp["rollout_profile"], "lift_diffusion",
                                            p)["evaluations_per_hour"])
    # (tuning, steps, checkpoint cadence). The 5k-cadence variants exist because the BC-MLP
    # pilot peaked at its 5k checkpoint, which a 10k cadence never evaluates.
    variants = {"main_100k_full_tuning": ("max", 100_000, 10_000),
                "main_100k_ckpt5k_full_tuning": ("max", 100_000, 5_000),
                "main_50k_ckpt5k_full_tuning": ("max", 50_000, 5_000),
                "main_100k_no_further_tuning": ("none", 100_000, 10_000),
                "reduced_50k_no_further_tuning": ("none", 50_000, 10_000)}
    scen = {}
    for name, (tun, steps, ck) in variants.items():
        pr = project(inp, tun, steps, ck, k, p_eval)
        scen[name] = {"local": pr, "modal_cpu_training": modal_training_cost(inp, pr, cfg, "cpu"),
                      "modal_gpu_training": modal_training_cost(inp, pr, cfg, "cuda"),
                      "storage": storage(inp, pr, cfg)}
    res = {"schema": "roboforge.suite_projection/1",
           "kind": "ESTIMATES from measured inputs and stated assumptions (not a bill)",
           "pricing": {"fetched_utc": cfg["pricing"]["fetched_utc"],
                       "source": cfg["pricing"]["source"],
                       "monthly_credits_usd": cfg["pricing"]["monthly_credits_usd"]},
           "gate_a_probe_upper_bound": launch_upper_bound(cfg, "gpu_probe", 900),
           "assumptions": ASSUME, "inputs": {k2: v for k2, v in inp.items() if k2 != "sources"},
           "sources": inp["sources"], "rendering": rendering(inp), "scenarios": scen,
           "git": git_revision()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    return res


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job")
    j.add_argument("--function", choices=["train_cpu", "gpu_probe"])
    j.add_argument("--config", type=Path, help="(informational) the run config")
    j.add_argument("--gpu", help="price this GPU instead of the function's (e.g. T4, L4)")
    j.add_argument("--max-minutes", type=float, required=True)
    s = sub.add_parser("suite")
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg = load_modal_config()
    if a.cmd == "job":
        fn = a.function or ("gpu_probe" if a.gpu else "train_cpu")
        ub = launch_upper_bound(cfg, fn, a.max_minutes * 60)
        if a.gpu:
            f = cfg["functions"][fn]
            ub["job"] = cost_usd(Resources(a.max_minutes * 60 + cfg["policy"]
                                           ["scaledown_window_s"], f["cpu"],
                                           f["memory_mib"] / 1024, a.gpu), cfg["pricing"])
            ub["total_usd"] = ub["job"]["total_usd"] + ub["build_allowance"]["total_usd"]
        ub["config"] = str(a.config) if a.config else None
        print(json.dumps(ub, indent=1))
        return 0
    res = suite(a.out)
    for name, sc in res["scenarios"].items():
        lo, hi = sc["local"]["low"], sc["local"]["high"]
        print(f"{name}: local train {lo['train_wall_h']:.1f}-{hi['train_wall_h']:.1f} h wall, "
              f"eval {lo['eval_wall_h']:.1f}-{hi['eval_wall_h']:.1f} h wall, $0 | "
              f"Modal CPU training ${sc['modal_cpu_training']['low']['usd']:.2f}-"
              f"{sc['modal_cpu_training']['high']['usd']:.2f} | Modal GPU training "
              f"${sc['modal_gpu_training']['low']['usd']:.2f}-"
              f"{sc['modal_gpu_training']['high']['usd']:.2f} "
              f"({sc['modal_gpu_training']['rate_source']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
