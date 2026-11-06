"""Checkpoint reload equivalence in the closed loop, plus the observe-hook regression check.

    # on the host, once per evidence session: extract the Session 2 sanity reference hashes
    ROBOFORGE_EVIDENCE_SESSION=4 python scripts/check_checkpoint_equivalence.py \
        --write-sanity-reference
    # on the canonical platform
    ROBOFORGE_EVIDENCE_SESSION=4 infra/containers/run_sim_cpu.sh \
        python scripts/check_checkpoint_equivalence.py --train-run runs/<id> --step <N>

Checks, on whatever platform the script runs on (it records which):

A. two independent strict ``weights_only`` loads of the checkpoint have bitwise-equal
   parameters;
B. closed-loop episodes on the first ``--n-states`` validation starts give identical
   ``physics_sha256`` and identical raw policy actions with either load;
C. a fresh Python process loading the checkpoint reproduces the same ``physics_sha256``;
D. if a validation evaluation of this checkpoint on this platform exists, its recorded
   ``physics_sha256`` for those starts is reproduced;
E. on the canonical platform only: the Session 2 random and no-op sanity episodes on
   validation start 0 still give their recorded ``physics_sha256``. This shows that the
   changes to the rollout runner (the ``observe`` hook, D-032; ``policy_episode``, D-041)
   left non-learned policies bitwise unchanged;
F. stochastic policies only: the two loads draw bitwise-identical seeded samples on
   validation-demo windows (same generator seed), and the noise seed recorded in each
   episode equals the protocol rule for ``(policy_seed, state_id, repeat)``.

Stochastic policies run with ``--policy-seed`` / ``--repeat`` (default 0 / 0, the
validation setting). The report goes to ``<out-dir>`` (default
``reports/session$ROBOFORGE_EVIDENCE_SESSION/checkpoint_equivalence``, D-039). Check E
reads ``<out-dir>/session2_sanity_reference.json``, which ``--write-sanity-reference``
writes on the host from ``reports/session2`` (the container does not mount that directory).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from roboforge.data.sequence import SequenceDataset  # noqa: E402
from roboforge.eval.learned import LearnedPolicy, noise_seed  # noqa: E402
from roboforge.eval.policies import NoOpPolicy, RandomPolicy  # noqa: E402
from roboforge.eval.rollout import run_episode  # noqa: E402
from roboforge.eval.validation import canonical_platform_id, eval_run_id  # noqa: E402
from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative  # noqa: E402
from roboforge.provenance import evaluation_platform, git_revision, sha256_file  # noqa: E402
from roboforge.sim.env_adapter import RoboForgeEnv  # noqa: E402
from roboforge.sim.eval_states import load_suite  # noqa: E402
from roboforge.train.data import WindowTensors  # noqa: E402
from roboforge.train.run import read_manifest  # noqa: E402

REFERENCE_NAME = "session2_sanity_reference.json"


def episodes(policy: LearnedPolicy, task: str, n: int) -> list[dict[str, Any]]:
    """Run the first ``n`` validation starts; return per-episode hashes."""
    env = RoboForgeEnv(task)
    codec = policy.codec(env.action_spec)
    out = []
    try:
        for init in load_suite(task, "validation")[:n]:
            rec = run_episode(env, policy, init, codec=codec)
            out.append({"state_id": init.state_id, "physics_sha256": rec.meta["physics_sha256"],
                        "raw_actions_sha256": _sha(rec.arrays["raw_policy_action"]),
                        "success": rec.outcome["success"], "steps": rec.outcome["steps"],
                        "policy_episode": rec.meta.get("policy_episode")})
    finally:
        env.close()
    return out


def _sha(a: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_reference(out_dir: Path) -> int:
    """Extract the Session 2 sanity hashes the container cannot read (it mounts only the
    current evidence session)."""
    rows = []
    for f in sorted((REPO_ROOT / "reports" / "session2" / "sanity_eval").glob("*.json")):
        d = json.loads(f.read_text())
        e = d["episodes"][0]
        rows.append({"task": d["task"], "policy": d["policy"]["name"],
                     "policy_seed": d["policy"]["params"].get("policy_seed"),
                     "state_id": e["state_id"], "physics_sha256": e["physics_sha256"],
                     "platform_id": d["platform"]["platform_id"],
                     "source": {"path": repo_relative(f), "sha256": sha256_file(f)}})
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = out_dir / REFERENCE_NAME
    ref.write_text(json.dumps({"schema": "roboforge.sanity_reference/1", "rows": rows},
                              indent=1) + "\n")
    print(f"wrote {len(rows)} reference rows -> {repo_relative(ref)}")
    return 0


def sanity_regression(reference: Path) -> dict[str, Any]:
    """Re-run the Session 2 sanity episodes (validation start 0) and compare hashes."""
    ref = json.loads(reference.read_text())["rows"]
    rows = []
    for r in ref:
        env = RoboForgeEnv(r["task"])
        try:
            policy = (RandomPolicy(env.action_spec, r["policy_seed"])
                      if r["policy"] == "random_uniform" else NoOpPolicy(env.action_spec))
            init = load_suite(r["task"], "validation")[0]
            assert init.state_id == r["state_id"]
            got = run_episode(env, policy, init).meta["physics_sha256"]
        finally:
            env.close()
        rows.append({**{k: r[k] for k in ("task", "policy", "state_id", "platform_id")},
                     "expected": r["physics_sha256"], "got": got,
                     "equal": got == r["physics_sha256"]})
    return {"reference": repo_relative(reference), "rows": rows,
            "passed": bool(rows) and all(x["equal"] for x in rows)}


def seeded_samples(p1: LearnedPolicy, p2: LearnedPolicy, task: str,
                   n_windows: int = 64) -> dict[str, Any]:
    """Check F: both loads sample identical chunks from identical generators."""
    kw = {"obs_horizon": p1.obs_horizon, "pred_horizon": p1.pred_horizon}
    data = WindowTensors.from_dataset(SequenceDataset.from_committed(task, "valid", **kw))
    b = data.batch(np.arange(min(n_windows, len(data))))
    outs = [p.model.sample(b["obs_history"], b["obs_valid"], torch.Generator().manual_seed(0))
            for p in (p1, p2)]
    return {"n_windows": len(b["obs_history"]), "generator_seed": 0,
            "samples_sha256": _sha(outs[0].numpy()), "equal": bool(torch.equal(*outs))}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train-run", type=Path)
    p.add_argument("--step", type=int, help="checkpoint step (default: the last)")
    p.add_argument("--n-states", type=int, default=3)
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: reports/session$ROBOFORGE_EVIDENCE_SESSION/"
                        "checkpoint_equivalence")
    p.add_argument("--write-sanity-reference", action="store_true")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    out_dir = args.out_dir or evidence_reports_dir("checkpoint_equivalence")
    if args.write_sanity_reference:
        return write_reference(out_dir)
    tm = read_manifest(args.train_run)
    ck = {c["step"]: c for c in tm["checkpoints"]}
    step = args.step or max(ck)
    path, sha = REPO_ROOT / ck[step]["path"], ck[step]["sha256"]
    task = tm["config"]["task"]
    seeds = {"policy_seed": args.policy_seed, "repeat": args.repeat}
    if args.child:
        print(json.dumps(episodes(LearnedPolicy.from_checkpoint(path, sha, **seeds), task,
                                  args.n_states)))
        return 0
    p1 = LearnedPolicy.from_checkpoint(path, sha, **seeds)
    p2 = LearnedPolicy.from_checkpoint(path, sha, **seeds)
    s1, s2 = p1.model.state_dict(), p2.model.state_dict()
    a = {"params_bitwise_equal": all(torch.equal(s1[k], s2[k]) for k in s1)}
    e1, e2 = episodes(p1, task, args.n_states), episodes(p2, task, args.n_states)
    b = {"episodes": e1, "equal": e1 == e2}
    child = subprocess.run([sys.executable, __file__, "--child", "--train-run",
                            str(args.train_run), "--step", str(step), "--n-states",
                            str(args.n_states), "--policy-seed", str(args.policy_seed),
                            "--repeat", str(args.repeat)],
                           capture_output=True, text=True, check=True)
    e3 = json.loads(child.stdout.strip().splitlines()[-1])
    c = {"equal": e3 == e1}
    platform = evaluation_platform()
    eval_dir = REPO_ROOT / "runs" / eval_run_id(tm["run_id"], step, platform["platform_id"],
                                                stochastic=p1.stochastic, **seeds)
    d: dict[str, Any] = {"eval_run": repo_relative(eval_dir), "available": eval_dir.is_dir()}
    if d["available"]:
        recorded = {s: json.loads((eval_dir / "episodes" / s / "record.json").read_text())[
            "physics_sha256"] for s in (e["state_id"] for e in e1)}
        d["equal"] = all(recorded[e["state_id"]] == e["physics_sha256"] for e in e1)
    canonical = platform["platform_id"] == canonical_platform_id()
    e = (sanity_regression(out_dir / REFERENCE_NAME) if canonical
         else {"skipped": "not the canonical platform"})
    f: dict[str, Any] = {"skipped": "deterministic policy"}
    if p1.stochastic:
        f = seeded_samples(p1, p2, task)
        f["noise_seeds_follow_rule"] = all(
            ep["policy_episode"]["noise_seed"] == noise_seed(args.policy_seed, ep["state_id"],
                                                             args.repeat) for ep in e1)
        f["passed"] = f["equal"] and f["noise_seeds_follow_rule"]
    passed = (a["params_bitwise_equal"] and b["equal"] and c["equal"]
              and d.get("equal", True) and e.get("passed", True) and f.get("passed", True))
    report = {"schema": "roboforge.checkpoint_equivalence/1", "train_run_id": tm["run_id"],
              "checkpoint": {"step": step, "path": ck[step]["path"], "sha256": sha},
              "platform": platform, "canonical_platform": canonical, "git": git_revision(),
              "A_two_loads_params": a, "B_two_loads_closed_loop": b,
              "C_fresh_process_closed_loop": c, "D_matches_recorded_validation_eval": d,
              "E_session2_sanity_regression": e, "F_seeded_samples": f,
              "stochastic_policy": seeds if p1.stochastic else None, "passed": passed}
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_ps{args.policy_seed}r{args.repeat}" if p1.stochastic else ""
    out = out_dir / f"{tm['run_id']}_step{step:07d}_{platform['platform_id']}{tag}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"[{tm['run_id']} step {step} on {platform['platform_id']}] A {a} | B {b['equal']} | "
          f"C {c['equal']} | D {d.get('equal', 'n/a')} | E {e.get('passed', 'skipped')} | "
          f"F {f.get('passed', 'n/a')} -> "
          f"{'PASS' if passed else 'FAIL'} ({repo_relative(out)})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
