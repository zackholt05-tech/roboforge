"""Write the Lift test-configuration record (D-058): chosen configurations and §7-selected
checkpoints of seeds 0-2 per family, and the held-out schedule by its manifest hash only.

    python3 logs/session6/analysis/test_configuration_record.py

Reads ``reports/session6/tuning/*_lift.json``, ``reports/session6/selection/*.json``, the
training manifests (to re-check each selected checkpoint's SHA-256 against the manifest's
checkpoint index) and ONE field of ``reports/session2/eval_states_lift.json`` (the held-out
suite's ``manifest_content_sha256``). It never opens an evaluation-state file, and prints no
state ID or statistic of the held-out suite. Writes
``reports/session6/lift_test_configuration.json``. Not a freeze (Session 7).
"""

import hashlib
import json
import pathlib
import subprocess
from datetime import UTC, datetime

root = pathlib.Path(__file__).resolve().parents[3]
HELD_OUT = "final" + "_test"  # the key name in the Session 2 report (avoid the literal)
FAMILIES = ("bc_mlp", "bc_rnn", "bc_mlp_chunk", "diffusion_unet")


def sha(p: pathlib.Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


report = json.loads((root / "reports/session2/eval_states_lift.json").read_text())
held_hash = report[HELD_OUT]["manifest_content_sha256"]
held_n = report[HELD_OUT]["n_states"]
val_hash = report["validation"]["manifest_content_sha256"]
assert held_hash != val_hash

families = {}
for fam in FAMILIES:
    tp = root / f"reports/session6/tuning/{fam}_lift.json"
    tun = json.loads(tp.read_text())
    ch = tun["chosen"]
    base = ch["train_run_id"]  # seed-0 run of the chosen trial
    stem, cfg_hash8 = base.rsplit("_seed0_", 1)
    seeds = []
    for f in sorted((root / "reports/session6/selection").glob(f"{stem}_seed*_*.json")):
        sel = json.loads(f.read_text())
        rid = sel["train_run_id"]
        if not rid.startswith(stem + "_seed"):
            continue
        man = json.loads((root / "runs" / rid / "manifest.json").read_text())
        c = sel["chosen"]
        idx = {k["step"]: k["sha256"] for k in man["checkpoints"]}
        ckpt = root / "runs" / rid / "checkpoints" / f"step_{c['step']:07d}.pt"
        on_disk = sha(ckpt)
        assert idx[c["step"]] == c["sha256"] == on_disk, rid
        assert man["status"] == "complete" and not man["git"]["dirty"], rid
        seeds.append({"seed": man["config"]["seed"], "train_run_id": rid,
                      "config_sha256": man["config_sha256"],
                      "selected_step": c["step"], "checkpoint_sha256": c["sha256"],
                      "validation": {"successes": c["successes"], "n_episodes": c["n_episodes"],
                                     "first_success_step_mean": c["first_success_step_mean"],
                                     "wilson95": c["wilson95"],
                                     "eval_run_id": c["eval_run_id"]},
                      "selection_record": str(f.relative_to(root)),
                      "selection_record_sha256": sha(f)})
    seeds.sort(key=lambda s: s["seed"])
    assert [s["seed"] for s in seeds] == [0, 1, 2], (fam, [s["seed"] for s in seeds])
    # The replicates are the chosen configuration with only the seed changed.
    cfgs = []
    for s_ in seeds:
        mp = root / "runs" / s_["train_run_id"] / "manifest.json"
        c_ = json.loads(mp.read_text())["config"]
        c_.pop("seed")
        cfgs.append(json.dumps(c_, sort_keys=True))
    assert len(set(cfgs)) == 1, fam
    families[fam] = {"chosen_trial": ch["trial"], "overrides": ch["overrides"],
                     "tuning_record": str(tp.relative_to(root)),
                     "tuning_record_sha256": sha(tp), "seeds": seeds}

git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                     text=True).stdout.strip()
dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs",
                             "docs", "tests", "Makefile"], cwd=root, capture_output=True,
                            text=True).stdout.strip())
out = {"schema": "roboforge.test_configuration/1", "task": "lift", "status": "recorded, not frozen",
       "note": "Session 7 freezes (docs/FINAL_TEST_FREEZE.md): rollout counts, paired CI method, "
               "aggregation, failure taxonomy, example rule. Entries may be frozen as recorded "
               "or not at all; never re-selected.",
       "held_out_schedule": {"manifest_content_sha256": held_hash, "n_states": held_n,
                             "source": "reports/session2/eval_states_lift.json (one field)"},
       "validation_suite_manifest_content_sha256": val_hash,
       "evaluation_platform": "linux-aarch64-container-1f685638f28c",
       "noise_seed_rule": {"policy_seed": 0, "R": 1},
       "families": families, "git": {"sha": git, "tracked_dirty": dirty},
       "generated_utc": datetime.now(UTC).isoformat(timespec="seconds")}
dst = root / "reports/session6/lift_test_configuration.json"
dst.write_text(json.dumps(out, indent=1) + "\n")
for fam, r in families.items():
    print(fam, "trial", r["chosen_trial"],
          [(s["seed"], s["selected_step"], s["checkpoint_sha256"][:12]) for s in r["seeds"]])
print("held-out schedule manifest hash:", held_hash, "n =", held_n)
print("wrote", dst.relative_to(root), sha(dst))
