"""X2-b latency benchmark of the selected X2 checkpoints (X2_harder_task §4; D-127).

Runs on the canonical container, inside :func:`roboforge.x2.binding.x2_bindings`. For every
checkpoint of ``reports/session16/x2/suite/x2_test_configuration.json`` (4 families x seeds
0-2): load it with 1 torch thread, collect the observations of a no-op episode (500 steps, the
X2 horizon) from X2 validation start 0, and time ``observe`` + ``act`` per decision and the
network forward alone with the frozen timing functions of ``scripts/benchmark_policies.py``
(first pass warm-up). Diffusion decisions are seeded samples (policy seed 0, repeat 0). No
evaluation and no held-out start: the policy never acts in the simulator here::

    ROBOFORGE_EVIDENCE_SESSION=16 infra/containers/run_sim_cpu.sh python scripts/x2b_benchmark.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import torch  # noqa: E402

from roboforge.eval.learned import LearnedPolicy  # noqa: E402
from roboforge.eval.policies import NoOpPolicy  # noqa: E402
from roboforge.paths import REPO_ROOT, repo_relative  # noqa: E402
from roboforge.provenance import evaluation_platform, git_revision, write_json  # noqa: E402
from roboforge.sim.env_adapter import RoboForgeEnv  # noqa: E402
from roboforge.sim.eval_states import load_suite  # noqa: E402
from roboforge.x2 import states as x2s  # noqa: E402
from roboforge.x2.binding import x2_bindings  # noqa: E402
from roboforge.x2.suite import x2b_config  # noqa: E402


def _frozen_benchmark():
    """The frozen ``scripts/benchmark_policies.py`` as a module (its timing functions)."""
    spec = importlib.util.spec_from_file_location(
        "benchmark_policies", REPO_ROOT / "scripts" / "benchmark_policies.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def observations(task: str) -> tuple[list, str]:
    """Observations of a no-op episode from X2 validation start 0, and that start's ID."""
    start = load_suite(task, x2s.VALIDATION_SUITE)[0]
    env = RoboForgeEnv(task)
    try:
        obs, _ = env.reset(start)
        out = [obs]
        noop = NoOpPolicy(env.action_spec)
        for _ in range(env.horizon):
            obs, _, term, trunc, _ = env.step(noop.act(obs)[0])
            out.append(obs)
            if term or trunc:
                break
    finally:
        env.close()
    return out, start.state_id


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args(argv)
    bench = _frozen_benchmark()
    out_dir = REPO_ROOT / x2b_config()["out_dir"]
    tc = json.loads((out_dir / "x2_test_configuration.json").read_text())
    rows = []
    with x2_bindings():
        obs, state_id = observations(tc["task"])
        for fam, v in tc["families"].items():
            for r in v["runs"]:
                path = (REPO_ROOT / "runs" / "x2" / r["train_run_id"] / "checkpoints"
                        / f"step_{int(r['selected_step']):07d}.pt")
                policy = LearnedPolicy.from_checkpoint(path, r["checkpoint_sha256"],
                                                       num_threads=1, policy_seed=0, repeat=0)
                rows.append({"family": fam, "seed": r["seed"], "train_run_id": r["train_run_id"],
                             "selected_step": r["selected_step"],
                             "checkpoint_sha256": r["checkpoint_sha256"],
                             "num_threads": torch.get_num_threads(),
                             "decisions_per_pass": len(obs), "repeats": args.repeats,
                             "sampler": (policy.model.sampler.to_dict() if policy.stochastic
                                         else None),
                             **bench.latency(policy, obs, args.repeats, state_id)})
    platform = evaluation_platform()
    report = {"schema": "roboforge.x2b_latency/1", "decisions": x2b_config()["decisions"],
              "label": "inference latency of the selected X2 checkpoints (measurement only)",
              "platform": platform, "git": git_revision(), "start": state_id, "rows": rows}
    out = out_dir / "latency" / f"{platform['platform_id']}.json"
    write_json(out, report)
    for r in rows:
        print(f"{r['family']:15s} seed {r['seed']}  observe+act median "
              f"{r['observe_plus_act_ms']['median']:.2f} ms  forward "
              f"{r['forward_only_ms']['median']:.2f} ms")
    print(repo_relative(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
