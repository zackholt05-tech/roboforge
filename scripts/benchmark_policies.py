"""Baseline training cost (from run manifests) and inference latency (microbenchmark).

    ROBOFORGE_EVIDENCE_SESSION=3 infra/containers/run_sim_cpu.sh \
        python scripts/benchmark_policies.py --train-runs runs/<id1> runs/<id2>

**Training cost** is read from each immutable training manifest: wall and CPU seconds, steps
per second and samples per second. These are the host's numbers; training runs on local CPU.

**Inference latency** is measured on the platform running this script. Load the final
checkpoint with the stated torch thread count, then:

* collect the observations of a 400-step no-op episode from validation start 0
  (``--max-decisions N`` keeps the first N);
* time ``observe`` + ``act`` per decision (normalization, window build, forward), after a
  warm-up;
* time the network forward alone.

For a stochastic policy (diffusion) a decision is one seeded sample (noise seed of the
benchmark start, ``policy_seed`` 0, repeat 0). ``--diffusion-samplers ddim:10 ddpm:100``
additionally times other reverse samplers on the same weights by swapping the sampler; those
rows are latency measurements only and never evaluated policies.

The rollout records also hold the in-loop ``planning_latency_ms`` of every decision. The
report goes to ``<out-dir>/<platform_id>.json`` (default
``reports/session$ROBOFORGE_EVIDENCE_SESSION/cost_latency``, D-039).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from roboforge.eval.learned import LearnedPolicy  # noqa: E402
from roboforge.eval.policies import NoOpPolicy  # noqa: E402
from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative  # noqa: E402
from roboforge.provenance import evaluation_platform, git_revision  # noqa: E402
from roboforge.sim.env_adapter import RoboForgeEnv  # noqa: E402
from roboforge.sim.eval_states import load_suite  # noqa: E402
from roboforge.train.run import read_manifest  # noqa: E402


def observations(task: str) -> list:
    """Observations of a 400-step no-op episode from validation start 0."""
    env = RoboForgeEnv(task)
    try:
        obs, _ = env.reset(load_suite(task, "validation")[0])
        out = [obs]
        noop = NoOpPolicy(env.action_spec)
        for _ in range(env.horizon):
            obs, _, term, trunc, _ = env.step(noop.act(obs)[0])
            out.append(obs)
            if term or trunc:
                break
    finally:
        env.close()
    return out


def stats_ms(x: list[float]) -> dict[str, float]:
    """Summary statistics in milliseconds."""
    a = 1000.0 * np.asarray(x)
    return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "mean": float(a.mean()), "n": int(len(a))}


def latency(policy: LearnedPolicy, obs: list, repeats: int,
            state_id: str) -> dict[str, dict[str, float]]:
    """Per-decision ``observe + act`` and forward-only (one network evaluation) timings."""
    meta = type("E", (), {"task": policy.task, "initial_state": {"state_id": state_id},
                          **{k: policy.hashes[k] for k in (
                              "observation_spec_sha256", "action_spec_sha256",
                              "task_spec_sha256")}})
    g = torch.Generator().manual_seed(0)
    decide, forward = [], []
    for r in range(repeats + 1):
        policy.reset_episode(meta)
        for o in obs:
            t0 = time.perf_counter()
            policy.observe(o)
            policy.act(o)
            dt = time.perf_counter() - t0
            oh, ov = policy.window()
            x, v = torch.from_numpy(oh)[None], torch.from_numpy(ov)[None]
            t1 = time.perf_counter()
            with torch.no_grad():
                if policy.stochastic:  # one U-Net evaluation (a sample makes K of them)
                    from roboforge.models.diffusion import unet_backend

                    m = policy.model
                    with unet_backend():
                        m.unet(torch.randn((1, m.pred_horizon, m.action_dim), generator=g),
                               torch.tensor(m.sampler.timesteps[0]), m.encoder(x, v))
                else:
                    policy.model(x, v)
            df = time.perf_counter() - t1
            if r:  # the first pass is warm-up
                decide.append(dt)
                forward.append(df)
    return {"observe_plus_act_ms": stats_ms(decide), "forward_only_ms": stats_ms(forward)}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train-runs", type=Path, nargs="+", required=True)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-decisions", type=int, default=None,
                   help="time only the first N frames' decisions per pass (default: all)")
    p.add_argument("--diffusion-samplers", nargs="*", default=[], metavar="NAME:STEPS",
                   help="extra reverse samplers to time on stochastic policies, e.g. ddpm:100")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    args.out_dir = args.out_dir or evidence_reports_dir("cost_latency")
    platform = evaluation_platform()
    obs_cache: dict[str, list] = {}
    rows = []
    for run in args.train_runs:
        tm = read_manifest(run)
        last = tm["checkpoints"][-1]
        policy = LearnedPolicy.from_checkpoint(REPO_ROOT / last["path"], last["sha256"],
                                               num_threads=args.num_threads)
        task = tm["config"]["task"]
        obs_cache.setdefault(task, observations(task)[:args.max_decisions])
        state_id = load_suite(task, "validation")[0].state_id
        t = tm["timing"]
        steps, batch = tm["config"]["train"]["max_steps"], tm["config"]["train"]["batch_size"]
        rows.append({
            "train_run_id": tm["run_id"], "name": tm["config"]["name"], "task": task,
            "param_count": tm["param_count"], "horizons": tm["config"]["horizons"],
            "training": {"platform_id": tm["platform"]["platform_id"],
                         "cpu": tm["platform"]["cpu"], "num_threads": tm["num_threads"],
                         "steps": steps, "batch_size": batch, "wall_s": t["wall_s"],
                         "cpu_s": t["cpu_s"], "steps_per_s": t["steps_per_s"],
                         "samples_per_s": steps * batch / t["wall_s"],
                         "note": "wall time includes offline-loss evaluation and checkpoint "
                                 "writes"},
            "inference": {"checkpoint": last, "num_threads": torch.get_num_threads(),
                          "decisions_per_pass": len(obs_cache[task]), "repeats": args.repeats,
                          "sampler": (policy.model.sampler.to_dict() if policy.stochastic
                                      else None),
                          **latency(policy, obs_cache[task], args.repeats, state_id)}})
        if policy.stochastic:
            from roboforge.models.noise_schedule import make_sampler

            configured = policy.model.sampler
            variants = []
            for spec in args.diffusion_samplers:
                name, steps = spec.split(":")
                policy.model.sampler = make_sampler(policy.model.schedule, name, int(steps),
                                                    policy.model.cfg.clip_sample)
                variants.append({"sampler": policy.model.sampler.to_dict(),
                                 "label": "latency only: same weights, swapped sampler; "
                                          "not an evaluated configuration",
                                 **latency(policy, obs_cache[task], args.repeats, state_id)})
            policy.model.sampler = configured
            rows[-1]["inference"]["sampler_variants"] = variants
    report = {"schema": "roboforge.cost_latency/1", "platform": platform, "git": git_revision(),
              "rows": rows}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{platform['platform_id']}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    for r in rows:
        i, tr = r["inference"], r["training"]
        for v in i.get("sampler_variants", []):
            print(f"{r['name']:13s} [{v['sampler']['sampler']}-"
                  f"{v['sampler']['num_inference_steps']}] act median "
                  f"{v['observe_plus_act_ms']['median']:.3f} ms (latency only)")
        print(f"{r['name']:13s} params {r['param_count']:,d}; train {tr['steps_per_s']:.0f} "
              f"steps/s ({tr['wall_s']:.0f} s); act median {i['observe_plus_act_ms']['median']:.3f}"
              f" ms p95 {i['observe_plus_act_ms']['p95']:.3f} ms; forward median "
              f"{i['forward_only_ms']['median']:.3f} ms")
    print(f"-> {repo_relative(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
