"""X1 engineering checks before any reported rollout (D-107; X1_action_clamp.md §3.5).

**Not reported results.** These checks run before the preregistration and before the first
clamped rollout on a robustness set. Results go to ``logs/session<N>/x1_checks/``, which no
analysis reads. There are four modes.

``noop`` runs in the canonical container. It takes one frozen checkpoint and a few
**development starts**: seeds 13,000,000 + i, outside every reserved block, partition
``eval_adhoc``. Each start runs twice, once through the frozen ``LearnedPolicy`` path and once
through the clamp wrapper with the clamp on.
- For a diffusion checkpoint, the check passes when every episode has the same
  ``physics_sha256`` and the same raw and executed actions, and the clamp changed no entry.
- For a BC checkpoint, it records how often the clamp acts, and checks that every executed
  action lies inside the clamp bounds.

    infra/containers/run_sim_cpu.sh python scripts/check_x1_engineering.py noop \
        --train-run runs/<id> --step N --expected-sha256 <sha> [--n-starts 5]

``compare`` runs on the host. Its input is a clamp-disabled summary written by
``scripts/eval_x1_clamp.py --no-clamp``. It finds the Session 7 summary of the same checkpoint
and set. It reads that summary only after its SHA-256 equals the value recorded in
``reports/session7/robustness/analysis.json``, whose own SHA-256 is pinned below. It then
requires, for all 50 starts, the same per-episode ``physics_sha256``, success, first-success
step and step count.

``isolation`` runs on the host. It writes a snapshot: the names of the top-level entries of
``runs/`` plus a content digest of every file in the 60 Session 7 H4 record directories.

``isolation-compare`` compares two snapshots. It requires the 60 digests to be equal, and the
only new top-level entries to be ``x1_clamp`` and ``x1_checks``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SESSION7_ANALYSIS = ("reports/session7/robustness/analysis.json",
                     "8cca66a8a8a8cecf0b734355167936e81983c821c625a23456523365409c9451")
DEV_SEED_BASE = 13_000_000
EPISODE_KEYS = ("physics_sha256", "success", "first_success_step", "steps", "stop_reason")


def _paths() -> tuple[Path, Path]:
    from roboforge.paths import REPO_ROOT, evidence_session

    return REPO_ROOT, REPO_ROOT / "logs" / f"session{evidence_session()}" / "x1_checks"


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as fh:  # exclusive create: a check result is written once
        fh.write(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(f"-> {path}")


def pinned_json(root: Path, rel: str, sha256: str) -> dict[str, Any]:
    """Read a JSON file only after its SHA-256 equals the pinned value."""
    from roboforge.provenance import sha256_file

    got = sha256_file(root / rel)
    if got != sha256:
        raise SystemExit(f"{rel}: SHA-256 {got} != pinned {sha256}")
    return json.loads((root / rel).read_text())


def session7_rows(root: Path) -> list[dict[str, Any]]:
    """Every H4 evaluation row of the hash-pinned Session 7 analysis (60: Lift 36, Can 24)."""
    a = pinned_json(root, *SESSION7_ANALYSIS)
    return [r for t in a["tasks"].values() for r in t["h4"]["evaluations"]]


def cmd_noop(args: argparse.Namespace) -> int:
    """Clamp on vs the frozen path on development starts (canonical container)."""
    from roboforge.sim.backend import configure_mujoco_gl

    configure_mujoco_gl()
    from roboforge.eval.learned import LearnedPolicy
    from roboforge.eval.rollout import run_episode
    from roboforge.exploratory.clamp import ClampedPolicy, clamp_bounds
    from roboforge.provenance import evaluation_platform, git_revision
    from roboforge.sim.env_adapter import RoboForgeEnv
    from roboforge.sim.eval_states import check_seed_allowed
    from roboforge.sim.snapshot import generate
    from roboforge.train.run import read_manifest

    root, out_dir = _paths()
    tm = read_manifest(args.train_run)
    ck = {c["step"]: c for c in tm["checkpoints"]}[args.step]
    if ck["sha256"] != args.expected_sha256:
        raise SystemExit("checkpoint SHA-256 differs from the expected value")
    path = root / ck["path"]
    task = tm["config"]["task"]
    bounds = clamp_bounds(task)
    lo, hi = np.asarray(bounds["low"]), np.asarray(bounds["high"])
    env = RoboForgeEnv(task)
    episodes = []
    try:
        for i in range(args.n_starts):
            seed = DEV_SEED_BASE + i
            partition = check_seed_allowed(task, seed)
            if partition != "eval_adhoc":
                raise SystemExit(f"seed {seed} maps to {partition}, not a development start")
            init = generate(env.raw_env, task, seed, f"{task}_x1dev_{seed}", partition)
            frozen = LearnedPolicy.from_checkpoint(path, expected_sha256=ck["sha256"])
            rec_f = run_episode(env, frozen, init, run_id="x1_noop_frozen",
                                codec=frozen.codec(env.action_spec))
            wrapped = ClampedPolicy(LearnedPolicy.from_checkpoint(
                path, expected_sha256=ck["sha256"]), clamp=True)
            rec_c = run_episode(env, wrapped, init, run_id="x1_noop_clamped",
                                codec=wrapped.codec(env.action_spec))
            af, ac = rec_f.arrays, rec_c.arrays
            n = min(len(af["executed_action"]), len(ac["executed_action"]))
            counts = rec_c.meta["policy_episode"]["x1_clamp"]["counts"]
            ex = ac["executed_action"]
            episodes.append({
                "seed": seed, "state_id": init.state_id, "partition": partition,
                "steps_frozen": int(len(af["executed_action"])), "steps_clamped": int(len(ex)),
                "success_frozen": rec_f.outcome["success"],
                "success_clamped": rec_c.outcome["success"],
                "physics_sha256_equal": rec_f.meta["physics_sha256"] ==
                rec_c.meta["physics_sha256"],
                "raw_actions_equal": bool(np.array_equal(af["raw_policy_action"],
                                                         ac["raw_policy_action"])),
                "executed_actions_equal": bool(np.array_equal(af["executed_action"],
                                                              ac["executed_action"])),
                "first_divergent_step": (None if np.array_equal(
                    af["executed_action"][:n], ac["executed_action"][:n]) else int(np.argmax(
                        np.any(af["executed_action"][:n] != ac["executed_action"][:n], axis=1)))),
                "raw_within_normalized_bound": bool(np.all(np.abs(ac["raw_policy_action"]) <= 1)),
                "executed_within_clamp_bounds": bool(np.all((ex >= lo) & (ex <= hi))),
                "clamp_counts": counts})
    finally:
        env.close()
    stochastic = bool(frozen.stochastic)
    if stochastic:
        passed = all(e["physics_sha256_equal"] and e["raw_actions_equal"] and
                     e["executed_actions_equal"] and e["clamp_counts"]["n_actions_clamped"] == 0
                     and e["raw_within_normalized_bound"] for e in episodes)
        criterion = ("diffusion: identical physics, raw and executed actions with and without "
                     "the clamp, and zero clamped entries")
    else:
        passed = all(e["executed_within_clamp_bounds"] for e in episodes)
        criterion = "BC: every executed action of the clamped run lies inside the clamp bounds"
    doc = {"check": "noop", "label": "X1 engineering check (development starts; not a result)",
           "train_run_id": tm["run_id"], "step": args.step, "checkpoint_sha256": ck["sha256"],
           "task": task, "stochastic": stochastic, "bounds": bounds,
           "platform": evaluation_platform(), "git": git_revision(),
           "criterion": criterion, "passed": passed, "episodes": episodes}
    _write(out_dir / f"noop_{tm['run_id']}_step{args.step:07d}.json", doc)
    print(f"noop {tm['run_id']} step {args.step}: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def cmd_compare(args: argparse.Namespace) -> int:
    """A clamp-disabled summary vs the hash-checked Session 7 summary of the same evaluation."""
    from roboforge.provenance import sha256_file

    root, out_dir = _paths()
    new = json.loads(Path(args.summary).read_text())
    if new["x1_clamp"]["enabled"]:
        raise SystemExit("compare needs a clamp-disabled (--no-clamp) summary")
    rows = [r for r in session7_rows(root) if r["eval_run_id"] + "__x1-noclamp" ==
            new["eval_run_id"]]
    if len(rows) != 1:
        raise SystemExit(f"no unique Session 7 row for {new['eval_run_id']}")
    row = rows[0]
    if sha256_file(root / row["summary"]) != row["summary_sha256"]:
        raise SystemExit(f"{row['summary']}: SHA-256 differs from the Session 7 analysis")
    old = json.loads((root / row["summary"]).read_text())
    eo = {e["state_id"]: e for e in old["episodes"]}
    en = {e["state_id"]: e for e in new["episodes"]}
    diffs = [{"state_id": s, **{k: [eo[s].get(k), en.get(s, {}).get(k)] for k in EPISODE_KEYS
                                if eo[s].get(k) != en.get(s, {}).get(k)}}
             for s in sorted(eo) if any(eo[s].get(k) != en.get(s, {}).get(k)
                                        for k in EPISODE_KEYS)]
    same = {"state_ids": sorted(eo) == sorted(en), "successes": old["successes"] ==
            new["successes"], "suite_manifest": old["suite_manifest_content_sha256"] ==
            new["suite_manifest_content_sha256"], "checkpoint": old["checkpoint"]["sha256"] ==
            new["checkpoint"]["sha256"], "platform": old["platform"]["platform_id"] ==
            new["platform"]["platform_id"], "episodes": not diffs}
    passed = all(same.values()) and new["verify_passed"] and new["canonical_platform"]
    doc = {"check": "reproduce", "label": "X1 engineering check (not a result)",
           "session7_summary": row["summary"], "session7_summary_sha256": row["summary_sha256"],
           "session7_analysis": {"path": SESSION7_ANALYSIS[0], "sha256": SESSION7_ANALYSIS[1]},
           "x1_summary": str(Path(args.summary).resolve().relative_to(root)),
           "x1_summary_sha256": sha256_file(args.summary),
           "successes": [old["successes"], new["successes"]], "n_episodes": len(eo),
           "compared_keys": list(EPISODE_KEYS), "equal": same, "differences": diffs,
           "verify_passed": new["verify_passed"], "x1_git": new["git"],
           "criterion": "every start: equal physics_sha256, success, first-success step, steps",
           "passed": passed}
    _write(out_dir / f"reproduce_{new['task']}.json", doc)
    print(f"reproduce {new['eval_run_id']}: {'PASS' if passed else 'FAIL'} "
          f"({old['successes']} vs {new['successes']}, {len(diffs)} differing episodes)")
    return 0 if passed else 1


def _dir_digest(d: Path) -> dict[str, Any]:
    h, n = hashlib.sha256(), 0
    for f in sorted(p for p in d.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(d)).encode() + b"\0")
        h.update(hashlib.sha256(f.read_bytes()).digest())
        n += 1
    return {"files": n, "digest": h.hexdigest()}


def cmd_isolation(args: argparse.Namespace) -> int:
    """Snapshot of runs/: top-level names + content digests of the Session 7 H4 records."""
    from roboforge.provenance import git_revision

    root, out_dir = _paths()
    digests = {}
    for r in session7_rows(root):
        s = json.loads((root / r["summary"]).read_text())
        digests[s["records_dir"]] = _dir_digest(root / s["records_dir"])
    doc = {"check": "isolation", "tag": args.tag, "git": git_revision(),
           "runs_top_level": sorted(p.name for p in (root / "runs").iterdir()),
           "session7_h4_record_dirs": digests}
    _write(out_dir / f"isolation_{args.tag}.json", doc)
    print(f"{len(doc['runs_top_level'])} top-level entries; {len(digests)} Session 7 dirs")
    return 0


def cmd_isolation_compare(args: argparse.Namespace) -> int:
    """Two isolation snapshots: Session 7 dirs unchanged; only X1 directories added."""
    root, out_dir = _paths()
    b = json.loads(Path(args.before).read_text())
    a = json.loads(Path(args.after).read_text())
    added = sorted(set(a["runs_top_level"]) - set(b["runs_top_level"]))
    removed = sorted(set(b["runs_top_level"]) - set(a["runs_top_level"]))
    changed = sorted(k for k in b["session7_h4_record_dirs"]
                     if b["session7_h4_record_dirs"][k] != a["session7_h4_record_dirs"].get(k))
    passed = (not removed and not changed and set(added) <= {"x1_clamp", "x1_checks"}
              and len(b["session7_h4_record_dirs"]) == 60)
    doc = {"check": "isolation-compare", "before": args.before, "after": args.after,
           "added_top_level": added, "removed_top_level": removed,
           "changed_session7_dirs": changed, "n_session7_dirs": len(b["session7_h4_record_dirs"]),
           "passed": passed}
    _write(out_dir / f"isolation_compare_{Path(args.after).stem}.json", doc)
    print(f"isolation: {'PASS' if passed else 'FAIL'} added={added} changed={len(changed)}")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    n = sub.add_parser("noop")
    n.add_argument("--train-run", type=Path, required=True)
    n.add_argument("--step", type=int, required=True)
    n.add_argument("--expected-sha256", required=True)
    n.add_argument("--n-starts", type=int, default=5)
    c = sub.add_parser("compare")
    c.add_argument("--summary", type=Path, required=True)
    i = sub.add_parser("isolation")
    i.add_argument("--tag", required=True)
    ic = sub.add_parser("isolation-compare")
    ic.add_argument("--before", required=True)
    ic.add_argument("--after", required=True)
    a = p.parse_args(argv)
    return {"noop": cmd_noop, "compare": cmd_compare, "isolation": cmd_isolation,
            "isolation-compare": cmd_isolation_compare}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
