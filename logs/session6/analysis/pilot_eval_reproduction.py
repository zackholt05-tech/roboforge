"""Session 6 trial-1 evaluations at 5k-30k vs the Session 3/4 pilot evaluations of the same
(bitwise-equal, see pilot_prefix_equivalence.json) checkpoints on the same canonical platform.

Compares, per checkpoint step, the per-episode (state_id, success, steps, stop_reason,
physics_sha256) tuples of the two validation summaries. Reads summaries only.

    python3 logs/session6/analysis/pilot_eval_reproduction.py > logs/session6/analysis/pilot_eval_reproduction.log
"""
import json
import pathlib

root = pathlib.Path(__file__).resolve().parents[3]
eq = json.loads((root / "logs/session6/analysis/pilot_prefix_equivalence.json").read_text())
PLAT = "linux-aarch64-container-1f685638f28c"
SESS = {"s3": "session3", "s4": "session4"}
out, all_ok = {}, True
for fam, rec in eq.items():
    pilot, t1 = rec["pilot"], rec["trial1"]
    suffix = "__ps0r0" if fam == "diffusion_unet" else ""
    rows = []
    for st in (5000, 10000, 15000, 20000, 25000, 30000):
        def load(run, sess):
            p = root / "reports" / sess / "validation" / f"{run}__val_step{st:07d}__{PLAT}{suffix}.json"
            return json.loads(p.read_text())
        a, b = load(pilot, SESS[pilot[:2]]), load(t1, "session6")
        key = lambda s: sorted((e["state_id"], e["success"], e["steps"], e["stop_reason"],
                                e["physics_sha256"]) for e in s["episodes"])
        same = key(a) == key(b)
        same_ckpt = a["checkpoint"]["sha256"] == b["checkpoint"]["sha256"]
        rows.append({"step": st, "episodes_identical": same,
                     "checkpoint_file_sha256_equal": same_ckpt,  # files embed the config (max_steps differs), KI-029
                     "successes": [sum(e["success"] for e in a["episodes"]),
                                   sum(e["success"] for e in b["episodes"])],
                     "suite_hash_equal": a["suite_manifest_content_sha256"]
                     == b["suite_manifest_content_sha256"]})
        all_ok &= same
    out[fam] = {"pilot": pilot, "trial1": t1, "steps": rows}
    print(fam, pilot, "->", t1)
    for r in rows:
        print("  ", r)
print("ALL 24 CHECKPOINT EVALUATIONS REPRODUCED EPISODE-FOR-EPISODE:", all_ok)
(root / "logs/session6/analysis/pilot_eval_reproduction.json").write_text(
    json.dumps({"all_identical": all_ok, "families": out}, indent=1) + "\n")
