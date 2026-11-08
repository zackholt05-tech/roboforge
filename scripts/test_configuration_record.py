"""Record a task's test configuration (D-058 for Lift; Session 7 for Can): the §6.3-chosen
configuration and the §7-selected checkpoints of seeds 0-2 per family, plus the held-out
schedule by its manifest hash only.

    ROBOFORGE_EVIDENCE_SESSION=7 uv run --frozen python scripts/test_configuration_record.py \
        --task can --session 7

Reads ``reports/session<S>/tuning/*_<task>.json``, ``reports/session<S>/selection/*.json``,
the training manifests (each selected SHA-256 is re-checked against the manifest index and
the file on disk) and ONE field of ``reports/session2/eval_states_<task>.json``. It never
opens an evaluation-state file and prints nothing of the held-out suite but its hash. Writes
``reports/session<S>/<task>_test_configuration.json`` (refuses to overwrite).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from roboforge.eval.freeze import FAMILY_ORDER
from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import git_revision, sha256_file

HELD_OUT_KEY = "final" + "_test"  # the key name in the Session 2 report


def record(task: str, session: int) -> dict:
    """The test-configuration record of ``task`` from session ``session``'s reports."""
    rep = REPO_ROOT / "reports" / f"session{session}"
    s2 = json.loads((REPO_ROOT / "reports/session2" / f"eval_states_{task}.json").read_text())
    held, val = s2[HELD_OUT_KEY]["manifest_content_sha256"], s2["validation"][
        "manifest_content_sha256"]
    assert held != val
    families = {}
    for fam in FAMILY_ORDER:
        tp = rep / "tuning" / f"{fam}_{task}.json"
        ch = json.loads(tp.read_text())["chosen"]
        stem = ch["train_run_id"].rsplit("_seed0_", 1)[0]
        seeds, cfgs = [], set()
        for f in sorted((rep / "selection").glob(f"{stem}_seed*_*.json")):
            sel = json.loads(f.read_text())
            rid = sel["train_run_id"]
            if not rid.startswith(stem + "_seed"):
                continue
            man = json.loads((REPO_ROOT / "runs" / rid / "manifest.json").read_text())
            c = sel["chosen"]
            idx = {k["step"]: k["sha256"] for k in man["checkpoints"]}
            disk = sha256_file(REPO_ROOT / "runs" / rid / "checkpoints"
                               / f"step_{c['step']:07d}.pt")
            assert idx[c["step"]] == c["sha256"] == disk, rid
            assert man["status"] == "complete" and not man["git"]["dirty"], rid
            cfg = dict(man["config"])
            cfg.pop("seed")
            cfgs.add(json.dumps(cfg, sort_keys=True))
            seeds.append({"seed": man["config"]["seed"], "train_run_id": rid,
                          "config_sha256": man["config_sha256"], "selected_step": c["step"],
                          "checkpoint_sha256": c["sha256"],
                          "validation": {k: c[k] for k in ("successes", "n_episodes",
                                                           "first_success_step_mean",
                                                           "wilson95", "eval_run_id")},
                          "selection_record": repo_relative(f),
                          "selection_record_sha256": sha256_file(f)})
        seeds.sort(key=lambda s: s["seed"])
        assert [s["seed"] for s in seeds] == [0, 1, 2], (fam, seeds)
        assert len(cfgs) == 1, f"{fam}: replicate configs differ beyond the seed"
        families[fam] = {"chosen_trial": ch["trial"], "overrides": ch["overrides"],
                         "tuning_record": repo_relative(tp),
                         "tuning_record_sha256": sha256_file(tp), "seeds": seeds}
    return {"schema": "roboforge.test_configuration/1", "task": task,
            "status": "recorded (frozen only by docs/FINAL_TEST_FREEZE.md)",
            "held_out_schedule": {"manifest_content_sha256": held,
                                  "n_states": s2[HELD_OUT_KEY]["n_states"],
                                  "source": f"reports/session2/eval_states_{task}.json "
                                            "(one field)"},
            "validation_suite_manifest_content_sha256": val,
            "evaluation_platform": "linux-aarch64-container-1f685638f28c",
            "noise_seed_rule": {"policy_seed": 0, "R": 1}, "families": families,
            "git": git_revision(), "generated_utc": datetime.now(tz=UTC).isoformat()}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", required=True)
    p.add_argument("--session", type=int, required=True)
    a = p.parse_args(argv)
    rec = record(a.task, a.session)
    out = REPO_ROOT / "reports" / f"session{a.session}" / f"{a.task}_test_configuration.json"
    with out.open("x") as f:
        f.write(json.dumps(rec, indent=1) + "\n")
    for fam, f in rec["families"].items():
        print(fam, "trial", f["chosen_trial"], [(s["seed"], s["selected_step"],
                                                 s["validation"]["successes"]) for s in f["seeds"]])
    print(f"-> {repo_relative(out)} (sha256 {sha256_file(out)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
