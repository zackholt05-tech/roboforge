"""Where closed-loop evaluation time goes: physics, features, inference, recording, rendering.

    ROBOFORGE_EVIDENCE_SESSION=5 infra/containers/run_sim_cpu.sh \
        python scripts/profile_rollouts.py \
        --task lift --train-run runs/<id> --step 15000 --n-states 50 --out <json>
    ... --task can --policy random_uniform --n-states 10 --out <json>

**Engineering profile, VALIDATION starts only** (Session 5). It runs ``run_episode`` exactly as
an evaluation does, with timers wrapped around the adapter's methods. The wrappers only
measure time; they change no input or output. Each episode's ``physics_sha256`` is compared
with the recorded canonical evaluation of the same checkpoint (``--reference``), which shows
that neither the instrumentation nor parallel load changed the physics. Nothing is written to
``runs/``, and no outcome is reported or used for selection: outcomes are recorded only to
compare them with the reference.

Timed parts of every control step:

* ``physics``: robosuite ``env.step``, the OSC controller plus 25 MuJoCo substeps;
* ``features``: the policy observation from the post-step state (scratch ``MjData``);
* ``success``: the task predicate (``mj_kinematics``);
* ``inference``: ``policy.act`` and ``policy.observe``;
* ``recording``: per-frame kinematics and state capture for the record;
* ``reset``: restoring the stored snapshot.

``rendering`` is timed separately: ``--render-frames`` offscreen frames of two cameras at
256x256 from one start. State-based evaluation never renders.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import numpy as np  # noqa: E402

from roboforge.eval.learned import LearnedPolicy  # noqa: E402
from roboforge.eval.policies import RandomPolicy  # noqa: E402
from roboforge.eval.rollout import run_episode  # noqa: E402
from roboforge.provenance import evaluation_platform, git_revision  # noqa: E402
from roboforge.sim.env_adapter import RoboForgeEnv  # noqa: E402
from roboforge.sim.eval_states import load_suite  # noqa: E402
from roboforge.sim.kinematics import KinematicsExtractor  # noqa: E402
from roboforge.train.run import read_manifest  # noqa: E402

PARTS = ("physics", "features", "success", "inference", "recording", "reset")


def _timed(obj: Any, name: str, bucket: str, acc: dict[str, float]) -> None:
    """Replace ``obj.name`` by a wrapper adding its wall time to ``acc[bucket]``."""
    fn = getattr(obj, name)

    @functools.wraps(fn)
    def wrapper(*a: Any, **k: Any) -> Any:
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            acc[bucket] += time.perf_counter() - t

    setattr(obj, name, wrapper)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", choices=["lift", "can"], required=True)
    p.add_argument("--train-run", type=Path)
    p.add_argument("--step", type=int)
    p.add_argument("--policy", choices=["checkpoint", "random_uniform"], default="checkpoint")
    p.add_argument("--n-states", type=int, default=50)
    p.add_argument("--render-frames", type=int, default=10)
    p.add_argument("--reference", type=Path,
                   help="recorded evaluation manifest (runs/<eval_run_id>/manifest.json)")
    p.add_argument("--tag", default="")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    t_start = time.perf_counter()
    env = RoboForgeEnv(args.task)
    acc: dict[str, float] = defaultdict(float)
    if args.policy == "checkpoint":
        tm = read_manifest(args.train_run)
        ck = {c["step"]: c for c in tm["checkpoints"]}[args.step]
        policy = LearnedPolicy.from_checkpoint(ck["path"], expected_sha256=ck["sha256"])
        codec = policy.codec(env.action_spec)
        _timed(policy, "act", "inference", acc)
        if hasattr(policy, "observe"):
            _timed(policy, "observe", "inference", acc)
        label = f"{tm['run_id']} step {args.step}"
    else:
        policy, codec, label = RandomPolicy(env.action_spec, policy_seed=0), None, "random"
        _timed(policy, "act", "inference", acc)
    _timed(env._env, "step", "physics", acc)  # robosuite: controller + 25 substeps
    _timed(env, "_observation", "features", acc)
    _timed(env, "get_success", "success", acc)
    _timed(env, "reset", "reset", acc)
    # The extractor is rebuilt at every reset, so its method is timed on the class (in this
    # profiling process only).
    _timed(KinematicsExtractor, "frame", "recording", acc)
    _timed(env, "current_state", "recording", acc)
    ref = {}
    if args.reference:
        ref = {e["state_id"]: e["physics_sha256"]
               for e in json.loads(args.reference.read_text())["episodes"]}
    eps = []
    t_eval = time.perf_counter()
    for init in load_suite(args.task, "validation")[:args.n_states]:
        t = time.perf_counter()
        rec = run_episode(env, policy, init, codec=codec)
        eps.append({"state_id": init.state_id, "steps": int(rec.outcome["steps"]),
                    "wall_s": time.perf_counter() - t,
                    "physics_sha256_matches_reference":
                        (rec.meta["physics_sha256"] == ref[init.state_id]) if ref else None})
    eval_wall = time.perf_counter() - t_eval
    timed = {k: acc[k] for k in PARTS}
    n_steps = sum(e["steps"] for e in eps)
    render = {}
    if args.render_frames:
        init = load_suite(args.task, "validation")[0]
        env.reset(init)
        cams = ["agentview", "frontview"]
        times = []
        for _ in range(args.render_frames):
            for c in cams:
                t = time.perf_counter()
                env.render(c, 256, 256)
                times.append(time.perf_counter() - t)
        render = {"backend": os.environ.get("MUJOCO_GL"), "cameras": cams, "size": [256, 256],
                  "n_frames": len(times), "median_s": float(np.median(times)),
                  "mean_s": float(np.mean(times)), "first_s": times[0]}
    env.close()
    out = {"schema": "roboforge.rollout_profile/1",
           "kind": "engineering profile on validation starts (not a result; not selection)",
           "tag": args.tag, "task": args.task, "policy": label, "n_episodes": len(eps),
           "n_steps": n_steps, "eval_wall_s": eval_wall,
           "process_wall_s": time.perf_counter() - t_start,
           "steps_per_s": n_steps / eval_wall, "timed_s": timed,
           "untimed_s": eval_wall - sum(timed.values()),
           "per_step_ms": {k: 1e3 * v / max(n_steps, 1) for k, v in timed.items()},
           "share": {k: v / eval_wall for k, v in timed.items()},
           "physics_sha256_all_match": (all(e["physics_sha256_matches_reference"] for e in eps)
                                        if ref else None),
           "rendering": render, "episodes": eps, "platform": evaluation_platform(),
           "git": git_revision(), "cpu_count": os.cpu_count()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("tag", "task", "n_steps", "eval_wall_s",
                                          "steps_per_s", "share",
                                          "physics_sha256_all_match")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
