"""X2-c engineering check before the freeze: the X2-c evaluator reproduces X2-b (not a result).

    ROBOFORGE_EVIDENCE_SESSION=17 uv run --frozen python scripts/x2c_check.py

The X2-c evaluator in its check mode (``scripts/x2c_eval.py --check``: X2 validation starts
only, records in ``runs/x2c_checks/``) re-runs two of the 12 selected checkpoints of the
committed X2-b test configuration on the 50 X2 validation starts, in the canonical container,
both at once through the host-wide slots:

- diffusion seed 0 at step 100,000 (an X2-a pilot evaluation; checks the noise seeding);
- BC-MLP-chunk seed 1 at step 5,000 (an X2-b evaluation).

Each must reproduce its committed validation summary exactly: equal state ID,
``physics_sha256``, success, first-success step, step count and stop reason on all 50 starts,
the same success count, the 500-step horizon and a passed audit. The reference summaries and
the test configuration are read only after their committed SHA-256 checks. Results go to
``logs/session<N>/x2c_checks/reproduce.json``; the X2-c analysis never reads them. No X2
held-out start is stepped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any

from roboforge.paths import REPO_ROOT, evidence_session, repo_relative
from roboforge.provenance import git_revision, sha256_file, tracked_and_unmodified
from roboforge.slots import ContainerSlots
from roboforge.x2 import heldout as xh

TARGETS = (("diffusion_unet", 0, "reports/session15/x2/pilot/evals"),
           ("bc_mlp_chunk", 1, "reports/session16/x2/suite/evals"))
PLATFORM = "linux-aarch64-container-1f685638f28c"
EPISODE_KEYS = ("state_id", "physics_sha256", "success", "first_success_step", "steps",
                "stop_reason")


def logs_dir():
    """``logs/session<N>/x2c_checks``."""
    return REPO_ROOT / "logs" / f"session{evidence_session()}" / "x2c_checks"


def targets() -> list[dict[str, Any]]:
    """The two checkpoints and their committed X2-b validation summaries."""
    for rel in (xh.TEST_CONFIGURATION,):
        ok, why = tracked_and_unmodified(rel)
        if not ok or sha256_file(REPO_ROOT / rel) != xh.TEST_CONFIGURATION_SHA256:
            raise SystemExit(f"{rel}: not the committed X2-b test configuration ({why})")
    tc = json.loads((REPO_ROOT / xh.TEST_CONFIGURATION).read_text())["families"]
    out = []
    for fam, seed, src in TARGETS:
        (r,) = [x for x in tc[fam]["runs"] if int(x["seed"]) == seed]
        ref = f"{src}/{r['eval_run_id']}.json"
        ok, why = tracked_and_unmodified(ref)
        if not ok:
            raise SystemExit(f"{ref}: {why}")
        out.append({"family": fam, "seed": seed, "train_run_id": r["train_run_id"],
                    "step": int(r["selected_step"]), "checkpoint_sha256": r["checkpoint_sha256"],
                    "reference": ref, "reference_sha256": sha256_file(REPO_ROOT / ref),
                    "eval_run_id": xh.x2c_eval_run_id(r["train_run_id"], int(r["selected_step"]),
                                                      PLATFORM,
                                                      stochastic=fam == "diffusion_unet",
                                                      check=True)})
    return out


def compare(t: dict[str, Any]) -> dict[str, Any]:
    """Episode-by-episode comparison of a check summary with its X2-b summary."""
    path = logs_dir() / f"{t['eval_run_id']}.json"
    mine = json.loads(path.read_text())
    ref = json.loads((REPO_ROOT / t["reference"]).read_text())
    a = [tuple(e[k] for k in EPISODE_KEYS) for e in mine["episodes"]]
    b = [tuple(e[k] for k in EPISODE_KEYS) for e in ref["episodes"]]
    mism = [x[0] for x, y in zip(a, b, strict=False) if x != y]
    ok = (len(a) == len(b) == 50 and not mism and mine["schema"] == xh.CHECK_SCHEMA
          and mine["suite"] == ref["suite"] and mine["canonical_platform"]
          and mine["verify_passed"] and mine["successes"] == ref["successes"]
          and int(mine["episode_rules"]["horizon"]) == xh.HORIZON
          and mine["episode_rules"] == ref["episode_rules"]
          and mine["checkpoint"]["sha256"] == t["checkpoint_sha256"])
    return {"family": t["family"], "seed": t["seed"], "train_run_id": t["train_run_id"],
            "step": t["step"], "check_summary": repo_relative(path),
            "check_summary_sha256": sha256_file(path), "reference": t["reference"],
            "reference_sha256": t["reference_sha256"], "n_episodes": [len(a), len(b)],
            "successes": [mine["successes"], ref["successes"]],
            "episode_rules": mine["episode_rules"], "mismatching_episodes": mism,
            "passed": ok}


def main() -> int:
    """Run both checks in containers, then compare them on the host."""
    if git_revision()["dirty"]:
        raise SystemExit("working tree is dirty: refusing to launch")
    ts = targets()
    pool, running = ContainerSlots(), []
    for t in ts:
        if (logs_dir() / f"{t['eval_run_id']}.json").exists():
            continue
        slot = None
        while slot is None:
            slot = pool.acquire(t["eval_run_id"])
            if slot is None:
                time.sleep(10)
        cmd = ["infra/containers/run_sim_cpu.sh", "python", "scripts/x2c_eval.py", "--check",
               "--train-run", f"runs/x2/{t['train_run_id']}", "--step", str(t["step"]),
               "--family", t["family"], "--seed", str(t["seed"]),
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
    results = [compare(t) for t in ts]
    out = {"schema": "roboforge.x2c_engineering_check/1",
           "label": "engineering check, not a result", "git": git_revision(),
           "checks": results, "passed": all(r["passed"] for r in results)}
    (logs_dir() / "reproduce.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in results:
        print(f"[{r['family']} seed {r['seed']} step {r['step']}] successes {r['successes']}; "
              f"mismatching episodes {len(r['mismatching_episodes'])}; "
              f"{'PASS' if r['passed'] else 'FAIL'}")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
