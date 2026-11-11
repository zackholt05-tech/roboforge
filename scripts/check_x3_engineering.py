"""X3 engineering checks before the first X3 rollout (DECISIONS D-111; not results).

    ROBOFORGE_EVIDENCE_SESSION=14 uv run --frozen python scripts/check_x3_engineering.py \
        reproduce
    ROBOFORGE_EVIDENCE_SESSION=14 uv run --frozen python scripts/check_x3_engineering.py \
        snapshot --tag before

**reproduce.** The X3 evaluator in its check mode (``scripts/eval_x3.py --check``, validation
suite only) re-runs two frozen Can checkpoints on the 50 validation starts, in the canonical
container, at most 2 at a time through the host-wide slots:
- BC-MLP seed 0 at its selected step 55,000;
- diffusion seed 0 at its selected step 65,000 (this checks the noise seeding).
Each must reproduce its committed Session 7 validation summary exactly, with equal per-episode
state ID, ``physics_sha256``, success, first-success step, step count and stop reason on all 50
starts. The Session 7 summary is read only after its SHA-256 equals the value its committed
selection record lists. Records go to ``runs/x3_checks/``, results to
``logs/session<N>/x3_checks/``; the analysis never reads either.

**snapshot.** The top-level listing of ``runs/`` and a digest of the frozen Can held-out record
directories (names and file sizes), taken before and after the X3 evaluations. The only new
top-level entries may be ``x3`` and ``x3_checks``, and the digest must not change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT, evidence_session, repo_relative
from roboforge.provenance import git_revision, sha256_file
from roboforge.replication.x3 import RUNS_SUBDIR, x3_eval_run_id
from roboforge.replication.x3_jobs import PLATFORM, frozen_checkpoints
from roboforge.slots import ContainerSlots

# The Session 7 selection records of the two targets (committed; each lists the SHA-256 of
# every validation summary it read).
TARGETS = (("bc_mlp", "reports/session7/selection/s7_bc_mlp_cos_can_seed0_ae8152e3.json"),
           ("diffusion_unet",
            "reports/session7/selection/s7_diffusion_unet_cos_can_seed0_99251da3.json"))
EPISODE_KEYS = ("state_id", "physics_sha256", "success", "first_success_step", "steps",
                "stop_reason")
ALLOWED_NEW = {RUNS_SUBDIR[False], RUNS_SUBDIR[True]}


def logs_dir() -> Path:
    return REPO_ROOT / "logs" / f"session{evidence_session()}" / "x3_checks"


def target(family: str, rel: str) -> dict[str, Any]:
    """One target: the frozen checkpoint, and its hash-checked Session 7 validation summary."""
    sel = json.loads((REPO_ROOT / rel).read_text())
    ch = sel["chosen"]
    (fz,) = [c for c in frozen_checkpoints() if c["family"] == family and c["seed"] == 0]
    if (fz["train_run_id"], fz["step"], fz["checkpoint_sha256"]) != (
            sel["train_run_id"], int(ch["step"]), ch["sha256"]):
        raise SystemExit(f"{rel}: does not match the freeze's {family} seed 0 checkpoint")
    (src,) = [s for s in sel["sources"] if f"__val_step{int(ch['step']):07d}__" in s["path"]]
    path = REPO_ROOT / src["path"]
    if sha256_file(path) != src["sha256"]:
        raise SystemExit(f"{src['path']}: SHA-256 differs from {rel}")
    return {**fz, "s7_summary": src["path"], "s7_summary_sha256": src["sha256"],
            "eval_run_id": x3_eval_run_id(fz["train_run_id"], fz["step"], PLATFORM,
                                          stochastic=family == "diffusion_unet", check=True)}


def compare(t: dict[str, Any]) -> dict[str, Any]:
    """Episode-by-episode comparison of a check summary with its Session 7 summary."""
    mine = json.loads((logs_dir() / f"{t['eval_run_id']}.json").read_text())
    ref = json.loads((REPO_ROOT / t["s7_summary"]).read_text())
    a = [tuple(e[k] for k in EPISODE_KEYS) for e in mine["episodes"]]
    b = [tuple(e[k] for k in EPISODE_KEYS) for e in ref["episodes"]]
    mism = [x[0] for x, y in zip(a, b, strict=False) if x != y]
    ok = (len(a) == len(b) == 50 and not mism and mine["suite"] == "validation"
          and mine["canonical_platform"] and mine["verify_passed"]
          and mine["successes"] == ref["successes"]
          and mine["checkpoint"]["sha256"] == t["checkpoint_sha256"])
    return {"family": t["family"], "train_run_id": t["train_run_id"], "step": t["step"],
            "check_summary": repo_relative(logs_dir() / f"{t['eval_run_id']}.json"),
            "s7_summary": t["s7_summary"], "s7_summary_sha256": t["s7_summary_sha256"],
            "n_episodes": [len(a), len(b)], "successes": [mine["successes"], ref["successes"]],
            "mismatching_episodes": mism, "passed": ok}


def reproduce() -> int:
    """Run both checks in containers, then compare them on the host."""
    if git_revision()["dirty"]:
        raise SystemExit("working tree is dirty: refusing to launch")
    targets = [target(f, rel) for f, rel in TARGETS]
    pool, running = ContainerSlots(), []
    for t in targets:
        if (logs_dir() / f"{t['eval_run_id']}.json").exists():
            continue
        slot = None
        while slot is None:
            slot = pool.acquire(t["eval_run_id"])
            if slot is None:
                time.sleep(10)
        cmd = ["infra/containers/run_sim_cpu.sh", "python", "scripts/eval_x3.py", "--check",
               "--train-run", f"runs/{t['train_run_id']}", "--step", str(t["step"]),
               "--family", t["family"], "--seed", "0",
               "--expected-sha256", t["checkpoint_sha256"]]
        log = logs_dir() / f"{t['eval_run_id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log.open("a"),
                                stderr=subprocess.STDOUT,
                                env={**os.environ,
                                     "ROBOFORGE_EVIDENCE_SESSION": str(evidence_session())})
        pool.attach(slot, proc.pid)
        running.append((proc, slot))
    for proc, slot in running:
        proc.wait()
        pool.release(slot)
    results = [compare(t) for t in targets]
    out = {"schema": "roboforge.x3_engineering_check/1", "label": "engineering check, not a "
           "result", "git": git_revision(), "checks": results,
           "passed": all(r["passed"] for r in results)}
    (logs_dir() / "reproduce.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in results:
        print(f"[{r['family']} seed 0 step {r['step']}] successes {r['successes']}; "
              f"mismatching episodes {len(r['mismatching_episodes'])}; "
              f"{'PASS' if r['passed'] else 'FAIL'}")
    return 0 if out["passed"] else 1


def frozen_digest() -> dict[str, Any]:
    """Names and sizes of every file in the frozen Can held-out record directories."""
    h, n = hashlib.sha256(), 0
    for c in frozen_checkpoints():
        for d in sorted((REPO_ROOT / "runs").glob(f"{c['train_run_id']}__heldout*")):
            for f in sorted(d.rglob("*")):
                if f.is_file():
                    h.update(f"{f.relative_to(REPO_ROOT)}\t{f.stat().st_size}\n".encode())
                    n += 1
    return {"n_files": n, "sha256": h.hexdigest()}


def snapshot(tag: str) -> int:
    """Record (``before``) or compare (``after``) the runs/ listing and the frozen digest."""
    path = logs_dir() / "isolation_before.json"
    snap = {"top_level": sorted(p.name for p in (REPO_ROOT / "runs").iterdir()),
            "frozen_heldout_records": frozen_digest()}
    if tag == "before":
        if path.exists():
            raise SystemExit(f"{repo_relative(path)} exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snap, indent=1) + "\n")
        print(f"before: {len(snap['top_level'])} entries, frozen digest "
              f"{snap['frozen_heldout_records']}")
        return 0
    before = json.loads(path.read_text())
    new = sorted(set(snap["top_level"]) - set(before["top_level"]))
    gone = sorted(set(before["top_level"]) - set(snap["top_level"]))
    ok = (set(new) <= ALLOWED_NEW and not gone
          and snap["frozen_heldout_records"] == before["frozen_heldout_records"])
    res = {"new_top_level": new, "removed_top_level": gone,
           "frozen_heldout_records": [before["frozen_heldout_records"],
                                      snap["frozen_heldout_records"]], "passed": ok}
    (logs_dir() / "isolation_after.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"after: new {new}, removed {gone}, frozen digest unchanged "
          f"{snap['frozen_heldout_records'] == before['frozen_heldout_records']}: "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("reproduce")
    s = sub.add_parser("snapshot")
    s.add_argument("--tag", choices=["before", "after"], required=True)
    a = p.parse_args(argv)
    return reproduce() if a.cmd == "reproduce" else snapshot(a.tag)


if __name__ == "__main__":
    sys.exit(main())
